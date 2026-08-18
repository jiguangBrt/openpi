"""Non-robot WebSocket smoke test for a MarvinPro RTC policy server."""

from __future__ import annotations

import argparse
import uuid

import numpy as np
from openpi_client import websocket_client_policy


def _observation(prompt: str) -> dict:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "state": np.zeros(16, dtype=np.float32),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": prompt,
    }


def _check_actions(result: dict, horizon: int, *, label: str) -> None:
    actions = np.asarray(result.get("actions"))
    if actions.shape != (horizon, 16) or not np.isfinite(actions).all():
        raise RuntimeError(f"{label} returned invalid actions {actions.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-predicted-delay", type=int, default=4)
    parser.add_argument("--prompt", default="Stack all three red cones into one stable stack.")
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = policy.get_server_metadata().get("rtc", {})
    expected_metadata = {
        "protocol": "rtc_v1",
        "action_horizon": args.horizon,
        "native_action_dim": 16,
        "model_action_dim": 32,
        "execution_horizon": args.execution_horizon,
        "max_predicted_delay": args.max_predicted_delay,
        "prefix_attention_schedule": "exp",
    }
    if metadata != expected_metadata:
        raise RuntimeError(f"unexpected RTC metadata: {metadata!r}")

    observation = _observation(args.prompt)
    normal_result = policy.infer(observation)
    _check_actions(normal_result, args.horizon, label="normal inference")

    prefix_horizon = args.horizon - args.execution_horizon
    request = {
        "request_type": "rtc_v1",
        "request_id": uuid.uuid4().hex,
        "plan_id": "smoke-plan",
        "timeline_version": 1,
        "checkpoint_id": 1,
        "observation": observation,
        "old_remaining_actions_absolute": np.zeros((prefix_horizon, 16), dtype=np.float32),
        "d_pred": min(2, args.max_predicted_delay),
        "s": args.execution_horizon,
        "schedule": "exp",
        "beta": 5.0,
    }
    rtc_result = policy.infer(request)
    if not rtc_result.get("ok"):
        raise RuntimeError(f"valid RTC request was rejected: {rtc_result!r}")
    _check_actions(rtc_result, args.horizon, label="RTC inference")

    invalid_request = {
        **request,
        "request_id": uuid.uuid4().hex,
        "old_remaining_actions_absolute": np.zeros((prefix_horizon + 1, 16), dtype=np.float32),
    }
    invalid_result = policy.infer(invalid_request)
    if invalid_result.get("ok") or invalid_result.get("error_code") != "invalid_rtc_request":
        raise RuntimeError(f"invalid RTC request was not rejected structurally: {invalid_result!r}")

    post_rejection_result = policy.infer(observation)
    _check_actions(post_rejection_result, args.horizon, label="post-rejection inference")
    print(
        "MarvinPro RTC smoke passed: "
        f"H={args.horizon} s={args.execution_horizon} d_max={args.max_predicted_delay} "
        f"normal={np.asarray(normal_result['actions']).shape} "
        f"rtc={np.asarray(rtc_result['actions']).shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
