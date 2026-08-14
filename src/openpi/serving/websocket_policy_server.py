import asyncio
import concurrent.futures
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.policies import policy as _policy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openpi-policy",
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                payload = await websocket.recv()
                start_time = time.monotonic()
                deserialize_started = time.monotonic()
                request = msgpack_numpy.unpackb(payload)
                request_deserialization_ms = (time.monotonic() - deserialize_started) * 1000.0

                queued_at = time.monotonic()
                action, queue_ms, infer_ms = await asyncio.get_running_loop().run_in_executor(
                    self._inference_executor,
                    self._infer_queued,
                    request,
                    queued_at,
                )

                action["server_timing"] = {
                    "request_deserialization_ms": request_deserialization_ms,
                    "queue_ms": queue_ms,
                    "infer_ms": infer_ms,
                    "total_before_response_ms": (time.monotonic() - start_time) * 1000.0,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                serialize_started = time.monotonic()
                response = packer.pack(action)
                action["server_timing"]["response_serialization_ms"] = (
                    time.monotonic() - serialize_started
                ) * 1000.0
                response = packer.pack(action)
                await websocket.send(response)
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _infer_queued(self, request, queued_at: float) -> tuple[dict, float, float]:
        infer_started = time.monotonic()
        queue_ms = (infer_started - queued_at) * 1000.0
        if isinstance(request, dict) and request.get("request_type") == "rtc_v1":
            action = self._infer_rtc(request)
        else:
            action = self._policy.infer(request)
        infer_ms = (time.monotonic() - infer_started) * 1000.0
        return action, queue_ms, infer_ms

    def _infer_rtc(self, request: dict) -> dict:
        id_fields = {
            key: request.get(key)
            for key in ("request_id", "plan_id", "timeline_version", "checkpoint_id")
        }
        try:
            infer_rtc = getattr(self._policy, "infer_rtc", None)
            if infer_rtc is None:
                raise _policy.RtcPolicyError("served policy has no RTC inference method")
            result = infer_rtc(request)
            return {"ok": True, **id_fields, **result}
        except (_policy.RtcPolicyError, TypeError, ValueError) as exc:
            logger.warning("Rejected RTC request %s: %s", id_fields.get("request_id"), exc)
            return {
                "ok": False,
                **id_fields,
                "error_code": "invalid_rtc_request",
                "error": str(exc),
            }


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
