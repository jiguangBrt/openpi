# pi0.5-RECAP on Marvin Pro

This is an approximate implementation of the public RECAP algorithm on the
released pi0.5 architecture. It is not the unreleased official pi0.6* model.

## Fixed task contract

- Control/data rate: 15 Hz.
- Maximum episode duration: 120 seconds (`Tmax=1800` frames).
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
2. Start one dataset recording, then authorize one RTC rollout.
3. Let the policy run until success or a terminal failure. On success, wait the
   required 2 seconds and switch Apex Input Mode to None. At 120 seconds the
   client holds; switch Input Mode to None. Use the same handoff for an early
   safety or workspace failure.
4. Commit the recorded episode and append exactly one immutable outcome row.
5. Reset the scene. Do not retry an episode under the same ID because its first
   attempt failed.

Use IDs such as `eval-i0-000` through `eval-i0-099`. Training IDs should use a
different campaign prefix, for example `train-i0-000` through `train-i0-299`.

## Current RTC-compatible recorder

`/home/jh/TianJi_data_collector/MarvinPro_data_collector` writes the required
LeRobot v2.1 samples at the camera rate. It has been updated for the current
RTC stack: `/tj` robot topics, port 7331, continuous H264 decoding, the same
quad-camera crop as deployment, and measured gripper position normalized from
`0.0..1.25` to `0..1`. The motion bridge remains the RTC bridge on port 7332.

The two bridges can run together because the collector bridge is read-only and
uses a different TCP port. Always run their doctor checks before enabling
motion.

### Terminal A: RTC motion bridge

```bash
cd /home/jh/TianJi_data_collector/MarvinPro_deploy
export RUN_DIR="$PWD/logs/recap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
./scripts/run_bridge_on_controller.sh \
  --local-log "$RUN_DIR/bridge.log" --allow-motion --publish-hz 100
```

### Terminal B: read-only LeRobot collector bridge

```bash
cd /home/jh/TianJi_data_collector/MarvinPro_data_collector
./scripts/run_bridge_on_controller.sh --doctor --duration 8
./scripts/run_bridge_on_controller.sh
```

### Terminal C: one persistent campaign recorder

The output directory must not already exist. Keep this process open for all
episodes in one campaign because LeRobot 0.1.0 cannot append across runs.

```bash
cd /home/jh/OpenPI_UR/openpi
export COLLECTOR_DIR=/home/jh/TianJi_data_collector/MarvinPro_data_collector
export DATASET_ROOT=/home/jh/OpenPI_UR/datasets/recap_eval_i0
PYTHONPATH="$COLLECTOR_DIR/src" uv run python -m marvinpro_collector.cli record \
  --bridge-host 6.6.7.100 --task 1 \
  --repo-id marvinpro/recap_eval_i0 --out "$DATASET_ROOT"
```

At the RTC client's final confirmation, press Enter in the collector first,
then enter the required uppercase `E` in the RTC terminal. After the terminal
condition and Input Mode None handoff, press `s` in the collector. Record the
printed committed frame count before resetting the scene.

### Terminal D: one RTC episode

Use the exact RTC parameters that have passed the staged robot checklist. The
120-second task limit is the only duration change here.

```bash
cd /home/jh/OpenPI_UR/openpi
export DEPLOY_DIR=/home/jh/TianJi_data_collector/MarvinPro_deploy
export RUN_DIR=/home/jh/TianJi_data_collector/MarvinPro_deploy/logs/recap_current
mkdir -p "$RUN_DIR"
PYTHONPATH="$DEPLOY_DIR/src" uv run python -m marvinpro_deploy.rollout_client \
  --robot-host 6.6.7.100 --policy-host 192.168.50.73 --execute \
  --episode-seconds 120 --rollout-schedule rtc \
  --rtc-continuous --rtc-late-result-policy discard \
  --playback-mode interpolated --control-hz 100 --model-hz 15 \
  --playback-time-scale 3 --execute-steps 20 \
  --max-rtc-recoveries 3 --max-stuck-replans 2 \
  --policy-connect-timeout 5 --policy-request-timeout 2 \
  --log-level DEBUG --console-log-level WARNING \
  --log-file "$RUN_DIR/episode_000.log"
```

Do not use `--yes` for campaign collection. The collector's recording start and
the final motion authorization need to remain an explicit paired operation.
The RTC delay/merge guidance in MarvinPro_deploy is independent of RECAP's
paper-level policy guidance beta. Positive-only RECAP inference uses beta 1
without a second CFG branch and does not change the RTC merge settings.

### Add the outcome sidecar row

For a 527-frame successful evaluation episode:

```bash
cd /home/jh/OpenPI_UR/openpi
uv run python scripts/recap_manifest.py add \
  --manifest /home/jh/OpenPI_UR/datasets/recap_eval_i0.outcomes.jsonl \
  --episode-index 0 --episode-id eval-i0-000 \
  --source evaluation --policy-iteration 0 \
  --terminal-reason success --num-frames 527
```

For a full timeout use `--terminal-reason timeout --num-frames 1800`. Validate
the campaign with:

```bash
uv run python scripts/recap_manifest.py validate \
  --manifest /home/jh/OpenPI_UR/datasets/recap_eval_i0.outcomes.jsonl \
  --expected-episodes 100
```

Do not commit a collector episode whose frame validation failed. Record the
attempt as a collection-system failure outside the policy evaluation, fix the
source problem, reset, and allocate a new rollout ID.

## Offline RECAP pipeline

The outcome manifest and value predictions remain sidecars; source images and
Parquet files are not rewritten.

```bash
# Build supervised return bins from the labeled train manifest.
uv run python scripts/build_recap_value_targets.py \
  --outcomes "$RECAP_OUTCOMES" \
  --output "$RECAP_DATASET/meta/recap/value_targets.npz"

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
  --baseline-success-rate 0.42
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
