# pi0.5-RECAP on Marvin Pro

This is an approximate implementation of the public RECAP algorithm on the
released pi0.5 architecture. It is not the unreleased official pi0.6* model.

## Fixed task contract

- Control/data rate: 15 Hz.
- Maximum episode duration: 240 seconds (`Tmax=3600` frames). Collect with
  `--episode-seconds 230` so the recorded frames plus the in-flight drain tail
  stay under the cap.
- Success: the three red cones form one stable stack, both grippers have
  released it, and the result remains stable for 2 seconds.
- Failure terminal reasons: `timeout`, `safety_stop`, `out_of_workspace`,
  `operator_abort`, or `task_failure`.
- A RECAP episode is one complete reset-to-terminal rollout. An H20 chunk, RTC
  request, checkpoint, recovery, or merge is never a separate episode.

## What the 100 episodes mean

The 100 Iteration-0 episodes are a held-out baseline evaluation set. They must
not be used by value or policy training. The training rollouts are separate:
300 autonomous episodes for Iteration 0, followed by 300 from Policy 1.

For each of the 100 evaluation episodes:

1. Put the robot at the agreed start pose and reset the three cones according
   to the evaluation randomization protocol.
2. Start one rollout client run (Terminal 4 below), switch Apex Input Mode to
   Custom, then authorize motion with the required uppercase `E`. The client
   notifies the resident collector over localhost TCP and recording starts
   automatically at that moment — no collector-side action is needed to start.
3. Let the policy run until success or a terminal failure. On success, wait the
   required 2 seconds and switch Apex Input Mode to None. At 230 seconds the
   client ends the rollout and holds; switch Input Mode to None. Use the same
   handoff for an early safety or workspace failure.
4. Rule the episode on the collector terminal: `s` commits as success, `f`
   commits as `task_failure`, `d` discards. A commit appends exactly one
   immutable outcome row to `<dataset_root>/outcomes.jsonl`; a discard writes
   nothing and the episode index is reused by the next commit, so committed
   indices stay contiguous. If the client aborts on a safety gate, the
   collector discards the episode automatically without asking for a ruling.
5. Reset the scene. Wait until the collector prints its `[IDLE]` line before
   authorizing the next rollout: a commit's video encoding blocks the collector
   for tens of seconds, and an episode_start received while it is busy ruling
   or encoding is ignored or delayed, which would drop the start of the next
   episode.

Episode IDs are assigned by the collector (`ep_<index:06d>`); the campaign
attributes `source` and `policy_iteration` are fixed once at collector start
(`--source evaluation --policy-iteration 0` for this set, `--source autonomous`
for training rollouts). Campaigns are separated by dataset root directory, and
the `run_dir` field of each outcome row links back to the deploy-side client
log of the same rollout.

## Current RTC-compatible recorder

The recorder is `/home/jh/tianji_tools/marvinpro_collector`, a two-process
split: `ros_bridge` runs on the controller in the apex environment (read-only
BEST_EFFORT subscriptions, `/tj` robot topics plus the undistorted quad camera
stream) and pushes one snapshot per camera frame over TCP :7331; the collector
runs on the dev machine (uv env) and does stateful H264 decoding, the
deployment quad split, 16-dim state/action assembly, wall-clock resampling to
the 15 Hz grid, and LeRobot v2.1 writing. The motion bridge remains the RTC
bridge on port 7332 from the deploy repo `/home/jh/Openpi_deploy`.

In remote mode the collector is resident: the rollout client (Terminal 4)
notifies it of episode boundaries over localhost TCP `127.0.0.1:7931`, and
every committed episode appends to one accumulating LeRobot v2.1 dataset
(`--out`). Restarting the collector with the same `--out` resumes and continues
the episode numbering; an existing dataset is validated at startup and never
deleted. The collector also appends the outcome sidecar automatically (see
"Outcome sidecar" below). See the collector repo README for the full protocol,
state machine, and quality gates.

The two bridges can run together because the collector bridge is read-only and
uses a different TCP port. Always run their doctor checks before enabling
motion. Convenience wrappers for all four terminals live in
`/home/jh/Openpi_deploy/quickstarts/recap_t1..t4_*.sh` (gitignored local
helpers; the raw commands below are the source of truth).

### Terminal 1: collector data bridge (resident)

```bash
cd /home/jh/tianji_tools/marvinpro_collector
./scripts/run_bridge_on_controller.sh
```

### Terminal 2: campaign collector, remote mode (resident, needs a real TTY)

```bash
cd /home/jh/tianji_tools
uv run marvin-collector record \
  --bridge-host 6.6.7.100 \
  --repo-id marvinpro/recap_eval_i0 \
  --out /home/jh/tianji_tools/data/recap_eval_i0 \
  --remote-control-port 7931 \
  --source evaluation --policy-iteration 0
```

Use `--source autonomous` for the training rollouts and a different dataset
root per campaign. The terminal must stay interactive: rulings (`s`/`f`/`d`)
are keyboard-driven, and a headless collector strands completed episodes in
the pending-ruling state.

### Terminal 3: RTC motion bridge (resident)

```bash
cd /home/jh/Openpi_deploy
RUN_DIR=logs/recap_bridge_$(date +%Y%m%d_%H%M%S) && mkdir -p "$RUN_DIR"
./scripts/run_bridge_on_controller.sh \
  --local-log "$RUN_DIR/bridge.log" --allow-motion --publish-hz 100
```

### Terminal 4: one RTC episode per run (fresh RUN_DIR each time)

```bash
cd /home/jh/OpenPI_UR/openpi
RUN_DIR=/home/jh/Openpi_deploy/logs/recap_redcones_$(date +%Y%m%d_%H%M%S) && mkdir -p "$RUN_DIR"
PYTHONPATH=/home/jh/Openpi_deploy/src uv run python -m marvinpro_deploy.rollout_client \
  --robot-host 6.6.7.100 --policy-host 192.168.50.73 --execute \
  --episode-seconds 230 --rollout-schedule rtc \
  --rtc-continuous --rtc-late-result-policy discard \
  --playback-mode interpolated --control-hz 100 --model-hz 15 \
  --playback-time-scale 1.5 --execute-steps 20 \
  --max-rtc-recoveries 20 --max-stuck-replans 2 \
  --policy-connect-timeout 5 --policy-request-timeout 5 \
  --exit-mode-timeout 60 \
  --log-level DEBUG --console-log-level WARNING \
  --log-file "$RUN_DIR/client.log" \
  --record-notify-host 127.0.0.1
```

`--record-notify-host 127.0.0.1` links the client to the collector: the client
sends `episode_start` after the motion gate is confirmed and the operator types
`E`, and `episode_end` (`completed` / `operator_stopped` / `aborted`) at
teardown. Aborted episodes are auto-discarded by the collector without a
ruling prompt. Do not use `--yes` for campaign collection: motion
authorization must remain an explicit per-episode operation.

Time scale: production collection since 2026-09-05 uses `--playback-time-scale
1.5` (15 Hz knots played at a 10 Hz wall rate). The policy-server leg is
currently WiFi, and on 2026-09-04 the measured full-path latency (p50 ~230 ms,
p95 ~270-320 ms, worst spike 621 ms) exceeded the 15 Hz feasibility budget
(`d_max=4` needs p95 <= 216 ms including the 50 ms guard), so RTC spent 50-70%
of every episode in blocking synchronized fallback and recovery holds. Those
stall frames are poison for RECAP: the value target is a remaining-frames
regression and the advantage is a per-frame `V(t+15)-V(t)` difference, so
random freezes inject label noise, while a uniform slowdown is a consistent,
learnable mapping. 10 Hz raises the budget to ~350 ms, which the WiFi link
meets. Switch back to `--playback-time-scale 1` only on a wired policy link,
and never mix time scales within one dataset root.

The RTC delay/merge guidance in the deploy repo is independent of RECAP's
paper-level policy guidance beta. Positive-only RECAP inference uses beta 1
without a second CFG branch and does not change the RTC merge settings.

### Outcome sidecar (automatic)

The collector appends one row to `<dataset_root>/outcomes.jsonl` for every
committed episode; the row fields satisfy `recap_manifest.EpisodeOutcome`
(`episode_index`, `episode_id`, `task`, `run_dir`, `started_at`/`ended_at`,
`measured_avg_camera_fps`, `success`, `terminal_reason`, `source`,
`policy_iteration`, `num_frames`, `fps`). Ruling `s` writes `success`, ruling
`f` writes `task_failure`; `d` and client-aborted episodes write nothing.
`recap_manifest.py add` remains available for hand-built legacy manifests but
is not part of the normal flow.

Validate the campaign with:

```bash
cd /home/wangyihan/openpi_260821/repo
uv run python scripts/recap_manifest.py validate \
  --manifest "$RECAP_OUTCOMES" \
  --expected-episodes 100
```

The collector prints a per-episode frame-validation report before each ruling.
Do not commit an episode whose validation failed: press `d`, record the
attempt as a collection-system failure outside the policy evaluation, fix the
source problem, and reset the scene. The discarded index is reused by the next
commit, so the campaign stays contiguous.

## Offline RECAP pipeline

The outcome manifest and value predictions remain sidecars; source images and
Parquet files are not rewritten.

`ReCAPSpec.max_episode_seconds` defaults to 120; under the 240-second task
contract both build scripts below must receive `--max-episode-seconds 240`.
Per-frame rewards renormalize to `-1/max_episode_frames` automatically, so a
failed episode's return still sums to -1 and the 201-bin value discretization
is unchanged. Episodes longer than the spec value are rejected loudly by
`episode_rewards`, so a missing flag fails the build instead of corrupting data.

```bash
# Build supervised return bins from the labeled train manifest.
uv run python scripts/build_recap_value_targets.py \
  --outcomes "$RECAP_OUTCOMES" \
  --output "$RECAP_DATASET/meta/recap/value_targets.npz" \
  --max-episode-seconds 240

# Train V1 from the same H20 Iteration-0 checkpoint.
uv run python scripts/train_recap_value.py \
  --base-checkpoint "$H20_I0" \
  --dataset-root "$RECAP_DATASET" \
  --outcomes "$RECAP_OUTCOMES" \
  --targets "$RECAP_DATASET/meta/recap/value_targets.npz" \
  --output checkpoints/recap_value_v1/params

# Export one value expectation per exact episode/frame key.
uv run python scripts/predict_recap_values.py \
  --value-params checkpoints/recap_value_v1/params \
  --base-checkpoint "$H20_I0" --dataset-root "$RECAP_DATASET" \
  --output artifacts/recap_v1_values.npz

# Compute return bins, N=15 advantages, and positive/negative policy labels.
uv run python scripts/build_recap_sidecar.py \
  --outcomes "$RECAP_OUTCOMES" --values artifacts/recap_v1_values.npz \
  --validation-metrics checkpoints/recap_value_v1/validation_metrics.json \
  --output "$RECAP_DATASET/meta/recap/advantages.npz" \
  --baseline-success-rate 0.42 \
  --max-episode-seconds 240
```

The value trainer makes a deterministic, success-stratified 80/20 split by
whole episode. It writes `validation_metrics.json` next to `params`, including
success/failure AUROC and successful-trajectory remaining-time MAE. Sidecar
generation stops when AUROC is below 0.65, either validation class is absent,
or autonomous advantage labels collapse to all-positive/all-negative.

The public RLinf example uses a different Step-4 policy variant: negative
samples are unconditional, positive samples use 10% dropout, and inference is
two-branch CFG. This implementation intentionally follows this project's fixed
contract instead: explicit positive/negative prompts, 30% condition dropout,
and positive-only single-branch inference with beta 1.

Before policy training set the H20 checkpoint, aggregated dataset, sidecar, and
the exact two-epoch step count:

```bash
export OPENPI_RECAP_BASE_CHECKPOINT="$H20_I0"
export OPENPI_RECAP_DATASET_ROOT="$RECAP_DATASET"
export OPENPI_RECAP_SIDECAR_PATH="$RECAP_DATASET/meta/recap/advantages.npz"
export OPENPI_RECAP_POLICY_STEPS=<ceil(total_frames*2/8)>
uv run scripts/train.py pi05_marvinpro_red_cones_h20_recap \
  --exp-name policy1 --overwrite
```

Policy 2 must repeat the value and policy training from the same H20 Iteration-0
checkpoint after aggregating all demonstrations and Iteration 0/1 autonomous
episodes. Evaluation episodes never enter `$RECAP_DATASET`.
