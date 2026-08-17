import time

import numpy as np

from openpi.policies import policy as _policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class _FakeRtcPolicy:
    def infer(self, request):
        del request
        return {"actions": np.zeros((10, 16))}

    def infer_rtc(self, request):
        return {"actions": np.zeros((10, 16))}


class _RejectingRtcPolicy:
    def infer_rtc(self, request):
        del request
        raise _policy.RtcPolicyError("bad prefix")


def _request():
    return {
        "request_id": "request",
        "plan_id": "plan",
        "timeline_version": 2,
        "checkpoint_id": 3,
    }


def test_rtc_envelope_preserves_ids():
    server = WebsocketPolicyServer(_FakeRtcPolicy())
    result = server._infer_rtc(_request())  # noqa: SLF001
    assert result["ok"]
    assert result["request_id"] == "request"
    assert result["actions"].shape == (10, 16)


def test_invalid_rtc_request_is_structured():
    server = WebsocketPolicyServer(_RejectingRtcPolicy())
    result = server._infer_rtc(_request())  # noqa: SLF001
    assert not result["ok"]
    assert result["error_code"] == "invalid_rtc_request"
    assert "bad prefix" in result["error"]


def test_inference_executor_reports_queue_and_inference_time():
    server = WebsocketPolicyServer(_FakeRtcPolicy())

    result, queue_ms, infer_ms = server._infer_queued(_request(), time.monotonic() - 0.001)  # noqa: SLF001

    assert result["actions"].shape == (10, 16)
    assert queue_ms >= 0.5
    assert infer_ms >= 0.0
