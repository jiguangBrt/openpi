# MarvinPro RTC Remote GPU Test Checklist

This checklist covers tests intentionally not run on the local laptop. Run it on the remote inference server with the
MarvinPro checkpoint and enough GPU memory before enabling a real RTC merge on the robot.

## Scope already checked locally

- Official exponential prefix weights for `d_pred=2`, `s=6`, `H=10`.
- Reverse-time VJP correction sign, zero-weight behavior, and finite `t=0/1` endpoints.
- MarvinPro absolute action prefix through joint-delta conversion, normalization, 16-to-32 padding, and the complete
  output inverse transform.
- `rtc_v1` WebSocket dispatch, ID preservation, and structured request rejection.

These checks do not prove that the full Pi0/Pi0.5 RTC sampler compiles or meets the delay budget on the remote GPU.

## 1. Environment and unit tests

Record the OpenPI commit, checkpoint path, GPU model, driver, CUDA version, and output of `nvidia-smi`.

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/models/pi0_test.py \
  src/openpi/policies/policy_test.py \
  src/openpi/serving/websocket_policy_server_test.py \
  src/openpi/transforms_test.py \
  -m 'not manual'
```

Expected: all selected tests pass. CPU is intentional for deterministic unit tests.

## 2. Full checkpoint sampler and JIT

Select an otherwise idle GPU with enough memory. The first result is compile-only and must never be sent to a robot.

```bash
cd /mnt/reacher-fast/openpi_ur_pp_202607/repo

CUDA_VISIBLE_DEVICES=5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python - <<'PY'
import time
import numpy as np

from openpi.policies import marvinpro_policy
from openpi.policies import policy_config
from openpi.training import config

checkpoint = "checkpoints/pi05_marvinpro_red_cones/marvinpro_red_cones_40k_gpu67/39999"
policy = policy_config.create_trained_policy(
    config.get_config("pi05_marvinpro_red_cones"),
    checkpoint,
)
assert policy.metadata["rtc"]["protocol"] == "rtc_v1"

observation = marvinpro_policy.make_marvinpro_example()
state = np.asarray(observation["state"], dtype=np.float32)
prefix = np.repeat(state[None, :], 4, axis=0)

for d_pred in (4, 4, 1, 2, 3, 4):
    request = {
        "request_type": "rtc_v1",
        "request_id": f"direct-{d_pred}",
        "plan_id": "remote-test",
        "timeline_version": 1,
        "checkpoint_id": 1,
        "observation": observation,
        "old_remaining_actions_absolute": prefix,
        "d_pred": d_pred,
        "s": 6,
        "schedule": "exp",
        "beta": 5.0,
        "warmup": d_pred == 4,
    }
    started = time.monotonic()
    result = policy.infer_rtc(request)
    wall_ms = (time.monotonic() - started) * 1000
    actions = np.asarray(result["actions"])
    assert actions.shape == (10, 16)
    assert np.isfinite(actions).all()
    assert set(result["rtc_timing"]) == {
        "preprocess_ms", "denoise_ms", "postprocess_ms", "total_ms"
    }
    print(d_pred, wall_ms, result["rtc_timing"])
PY
```

Pass criteria:

- The first call compiles; the second `d_pred=4` call reuses the compiled executable.
- Changing `d_pred` from 1 through 4 does not trigger a shape-dependent recompile.
- Every result is finite `(10, 16)` and includes all timing fields.
- No GPU OOM, JAX tracing error, NaN, or Inf occurs.

## 3. WebSocket protocol and latency

Start the updated server:

```bash
CUDA_VISIBLE_DEVICES=5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "Stack all three red cones into one stable stack." \
  policy:checkpoint \
  --policy.config=pi05_marvinpro_red_cones \
  --policy.dir=checkpoints/pi05_marvinpro_red_cones/marvinpro_red_cones_40k_gpu67/39999
```

From a second process, send two discarded warmups followed by at least 20 valid `rtc_v1` requests. Verify:

- metadata advertises `rtc_v1`, H=10, execution horizon 6, native action dim 16, model action dim 32, and maximum
  `d_pred=4`;
- response IDs exactly match request, and actions are finite `(10, 16)`;
- invalid shape, `d_pred=5`, wrong `s`, or non-exponential schedule returns `ok=false` without closing the socket;
- record wall, preprocess, denoise, postprocess, and server latency for every stable request;
- `ceil(7.5 * (max(last 20 stable wall seconds) + 0.05)) <= 4`.

Do not proceed if the last condition fails. The deployment client must use synchronized mode for that server/checkpoint.

## 4. Deployment stages still required

Run in this order and attach logs to the test record:

1. Fake bridge with delayed and out-of-order RTC responses.
2. Offline recorded-observation replay against the remote server.
3. Real policy with `--rollout-schedule rtc --rtc-shadow`; no RTC result is merged.
4. One real chunk using `--rollout-schedule tracking`; verify no arm clipping.
5. Synchronized regression with the updated protocol.
6. Two RTC chunks at the conservative defaults.
7. Increase episode length only after every observation has checkpoint evidence and every merge has matching IDs with
   `d_actual <= d_pred <= 4`.

Record p50/p95/max latency, `d_pred`, `d_actual`, phase freezes, clipping, checkpoint error, merge-boundary error, and any
fallback reason. Any RTC failure must keep synchronized fallback latched for the rest of that episode.

## 2026-08-10 remote validation record

- Scope: `/mnt/reacher-fast/openpi_ur_pp_202607/repo`, starting from remote commit `57bcdb5`; the remote repository
  has an independent Git history, so the RTC content patch was context-checked and applied without replacing the
  remote files wholesale.
- Static/CPU: Ruff passed; the selected CPU suite passed with `21 passed, 2 deselected`.
- Direct checkpoint test on physical GPU 4: checkpoint load took 5.86 s and the first RTC JIT took 22.61 s. Stable
  `d_pred=1..4` calls took 108.13-111.58 ms, produced finite `(10, 16)` actions, and did not trigger shape-dependent
  recompilation.
- Persistent service: physical GPU 5, PID `3474022`, port 8000, log
  `/mnt/reacher-fast/openpi_ur_pp_202607/logs/rtc_policy_gpu5.log`; GPU 4 remained unused after the direct test.
- WebSocket RTC: metadata matched `rtc_v1`, H=10, native/model dims 16/32, execution horizon 4, and maximum delay 4.
  Across 20 requests on one persistent connection, server inference was 104.14-116.96 ms and laptop wall latency was
  p50 246.57 ms, p95 328.72 ms, max 341.28 ms. The conservative delay bound was 3. A later fresh-connection request
  took 400.75 ms wall / 123.80 ms server, which still gives the allowed maximum bound of 4 after the 50 ms margin.
- Invalid prefix shape, `d_pred=5`, `s=3`, and non-exponential schedule all returned `ok=false` without closing the
  socket; a valid request after those errors succeeded.
- Legacy vanilla inference remained compatible: first compile took 7.52 s; the next call took 302.78 ms wall and
  85.60 ms server, with finite `(10, 16)` actions.
- Not covered here: robot shadow/tracking/synchronized/two-chunk merge stages and their physical checkpoint evidence.

## 2026-08-11 execution horizon 6 validation record

- Scope: remote commit `b0d6366` plus the four listed uncommitted `s=6` protocol/test/documentation files. No model
  checkpoint, weight, unrelated repository path, GPU process, or tmux session was modified.
- Static/CPU: the selected suite passed with `21 passed, 2 deselected`; Ruff passed. The matching deployment client
  passed all 53 unit tests.
- Direct checkpoint test on physical GPU 5: first cache/JIT call took 3954.34 ms. Stable `d_pred=1..4` calls took
  109.30-116.16 ms, returned finite `(10,16)` actions, and did not recompile per delay value.
- Persistent service runs in `marvinpro_rtc:0.0` on GPU 5 and port 8000. Metadata advertises execution horizon 6,
  prefix `(4,16)`, and maximum delay 4. An old `s=4/(6,16)` request was rejected structurally; a following valid
  request on the same connection succeeded.
- The first 20-request wall-latency sample had transport outliers (`p95=808.58 ms`, `max=1230.51 ms`) despite server
  `p95=120.52 ms`, so it failed the delay bound. The immediate stable repeat measured 225.34-413.38 ms wall and
  105.34-125.27 ms server, which gives `d_pred=4` with the 50 ms guard. Robot validation must restart with continuous
  shadow and must fall back rather than merge if the current warmup estimate exceeds four steps.
