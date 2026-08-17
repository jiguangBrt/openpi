import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        started = time.monotonic()
        serialization_started = started
        data = self._packer.pack(obs)
        request_serialization_ms = (time.monotonic() - serialization_started) * 1000.0
        transport_started = time.monotonic()
        self._ws.send(data)
        response = self._ws.recv()
        transport_round_trip_ms = (time.monotonic() - transport_started) * 1000.0
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        decode_started = time.monotonic()
        result = msgpack_numpy.unpackb(response)
        response_decode_ms = (time.monotonic() - decode_started) * 1000.0
        if isinstance(result, dict):
            server_timing = result.get("server_timing", {})
            server_path_ms = 0.0
            if isinstance(server_timing, dict):
                for key in (
                    "request_deserialization_ms",
                    "queue_ms",
                    "infer_ms",
                    "response_serialization_ms",
                ):
                    value = server_timing.get(key)
                    if isinstance(value, int | float):
                        server_path_ms += max(0.0, float(value))
            result["client_timing"] = {
                "request_serialization_ms": request_serialization_ms,
                "transport_round_trip_ms": transport_round_trip_ms,
                "network_round_trip_estimate_ms": max(0.0, transport_round_trip_ms - server_path_ms),
                "response_decode_ms": response_decode_ms,
                "total_ms": (time.monotonic() - started) * 1000.0,
            }
        return result

    @override
    def reset(self) -> None:
        pass
