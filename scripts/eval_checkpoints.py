"""Offline evaluation for a sequence of JAX pi0 checkpoints.

The script evaluates the configured LeRobot test split with fixed random keys so
that checkpoint comparisons are paired and reproducible.  It reports both the
flow-matching objective used during training and open-loop action prediction
errors for the native (unpadded) action dimensions.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import math
import pathlib
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_marvinpro_red_cones")
    parser.add_argument("--checkpoint-root", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--test-episodes", type=int, nargs="+", default=list(range(103, 115)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument(
        "--metric-horizon",
        type=int,
        help="Number of leading action steps used for deployment-prefix metrics; defaults to the model horizon.",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def _make_test_config(args: argparse.Namespace) -> _config.TrainConfig:
    config = _config.get_config(args.config)
    base_config = dataclasses.replace(config.data.base_config, episodes=tuple(args.test_episodes))
    data_factory = dataclasses.replace(config.data, base_config=base_config)
    return dataclasses.replace(config, data=data_factory, batch_size=args.batch_size, num_workers=2)


def _collate(items):
    return jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *items)


def _make_batches(config: _config.TrainConfig, max_batches: int | None):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        multiprocessing_context="spawn" if config.num_workers else None,
        persistent_workers=config.num_workers > 0,
        collate_fn=_collate,
        drop_last=False,
    )
    batches = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batches.append(batch)
    return data_config, len(dataset), batches


def _device_put_batch(batch, data_sharding: jax.sharding.Sharding):
    batch = jax.tree.map(lambda value: jax.device_put(value, data_sharding), batch)
    return _model.Observation.from_dict(batch), batch["actions"]


def _resolve_metric_horizon(action_horizon: int, requested_horizon: int | None) -> int:
    metric_horizon = action_horizon if requested_horizon is None else requested_horizon
    if not 1 <= metric_horizon <= action_horizon:
        raise ValueError(f"metric horizon must be in 1..{action_horizon}, got {metric_horizon}")
    return metric_horizon


def _metric_names() -> tuple[str, ...]:
    return (
        "flow_loss_sum",
        "flow_loss_count",
        "norm_abs_sum",
        "norm_sq_sum",
        "norm_count",
        "joint_abs_sum",
        "joint_sq_sum",
        "joint_count",
        "gripper_abs_sum",
        "gripper_sq_sum",
        "gripper_count",
        "prefix_norm_abs_sum",
        "prefix_norm_sq_sum",
        "prefix_norm_count",
        "prefix_joint_abs_sum",
        "prefix_joint_sq_sum",
        "prefix_joint_count",
        "prefix_gripper_abs_sum",
        "prefix_gripper_sq_sum",
        "prefix_gripper_count",
        "first_joint_abs_sum",
        "first_joint_sq_sum",
        "first_joint_count",
        "first_gripper_abs_sum",
        "first_gripper_sq_sum",
        "first_gripper_count",
    )


def _finalize_metrics(sums: dict[str, float]) -> dict[str, float]:
    return {
        "test_flow_loss": sums["flow_loss_sum"] / sums["flow_loss_count"],
        "action_norm_mae": sums["norm_abs_sum"] / sums["norm_count"],
        "action_norm_rmse": math.sqrt(sums["norm_sq_sum"] / sums["norm_count"]),
        "joint_chunk_mae_rad": sums["joint_abs_sum"] / sums["joint_count"],
        "joint_chunk_rmse_rad": math.sqrt(sums["joint_sq_sum"] / sums["joint_count"]),
        "gripper_chunk_mae": sums["gripper_abs_sum"] / sums["gripper_count"],
        "gripper_chunk_rmse": math.sqrt(sums["gripper_sq_sum"] / sums["gripper_count"]),
        "action_prefix_norm_mae": sums["prefix_norm_abs_sum"] / sums["prefix_norm_count"],
        "action_prefix_norm_rmse": math.sqrt(sums["prefix_norm_sq_sum"] / sums["prefix_norm_count"]),
        "joint_prefix_mae_rad": sums["prefix_joint_abs_sum"] / sums["prefix_joint_count"],
        "joint_prefix_rmse_rad": math.sqrt(sums["prefix_joint_sq_sum"] / sums["prefix_joint_count"]),
        "gripper_prefix_mae": sums["prefix_gripper_abs_sum"] / sums["prefix_gripper_count"],
        "gripper_prefix_rmse": math.sqrt(sums["prefix_gripper_sq_sum"] / sums["prefix_gripper_count"]),
        "joint_first_mae_rad": sums["first_joint_abs_sum"] / sums["first_joint_count"],
        "joint_first_rmse_rad": math.sqrt(sums["first_joint_sq_sum"] / sums["first_joint_count"]),
        "gripper_first_mae": sums["first_gripper_abs_sum"] / sums["first_gripper_count"],
        "gripper_first_rmse": math.sqrt(sums["first_gripper_sq_sum"] / sums["first_gripper_count"]),
    }


def _write_results(output_dir: pathlib.Path, metadata: dict, results: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    json_path = output_dir / "checkpoint_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = output_dir / "checkpoint_metrics.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by {jax.device_count()} devices")

    config = _make_test_config(args)
    action_horizon = int(config.model.action_horizon)
    metric_horizon = _resolve_metric_horizon(action_horizon, args.metric_horizon)
    data_config, dataset_size, host_batches = _make_batches(config, args.max_batches)
    evaluated_samples = sum(next(iter(batch.values())).shape[0] for batch in host_batches)
    logging.info(
        "Loaded test episodes %s: %d/%d frames in %d batches",
        args.test_episodes,
        evaluated_samples,
        dataset_size,
        len(host_batches),
    )

    if data_config.norm_stats is None or "actions" not in data_config.norm_stats:
        raise ValueError("Action normalization statistics are required for physical-space metrics")
    action_stats = data_config.norm_stats["actions"]
    if action_stats.q01 is None or action_stats.q99 is None:
        raise ValueError("Quantile action statistics are required for this pi0.5 config")

    valid_dim = int(np.asarray(action_stats.q01).shape[-1])
    if valid_dim != 16:
        raise ValueError(f"Expected 16 native Marvin Pro action dimensions, got {valid_dim}")
    joint_indices = jnp.asarray([*range(7), *range(8, 15)])
    gripper_indices = jnp.asarray([7, 15])
    q01 = jnp.asarray(action_stats.q01)
    q99 = jnp.asarray(action_stats.q99)

    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    batches = host_batches

    first_step = args.steps[0]
    first_params = _model.restore_params(
        args.checkpoint_root / str(first_step) / "params", dtype=jnp.bfloat16, sharding=replicated
    )
    first_model = config.model.load(first_params)
    graphdef, first_state = nnx.split(first_model)
    del first_model, first_params

    @jax.jit
    def eval_batch(state, loss_rng, sample_rng, observation, target_actions):
        model = nnx.merge(graphdef, state)
        model.eval()
        flow_loss = model.compute_loss(loss_rng, observation, target_actions, train=False)
        predicted_actions = model.sample_actions(sample_rng, observation, num_steps=args.sample_steps)[..., :valid_dim]
        target_actions = target_actions[..., :valid_dim]

        norm_diff = predicted_actions - target_actions
        predicted_physical = (predicted_actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        target_physical = (target_actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        physical_diff = predicted_physical - target_physical
        joint_diff = physical_diff[..., joint_indices]
        gripper_diff = physical_diff[..., gripper_indices]
        prefix_norm_diff = norm_diff[:, :metric_horizon]
        prefix_joint_diff = joint_diff[:, :metric_horizon]
        prefix_gripper_diff = gripper_diff[:, :metric_horizon]
        first_joint_diff = joint_diff[:, 0]
        first_gripper_diff = gripper_diff[:, 0]

        return jnp.asarray(
            [
                jnp.sum(flow_loss),
                flow_loss.size,
                jnp.sum(jnp.abs(norm_diff)),
                jnp.sum(jnp.square(norm_diff)),
                norm_diff.size,
                jnp.sum(jnp.abs(joint_diff)),
                jnp.sum(jnp.square(joint_diff)),
                joint_diff.size,
                jnp.sum(jnp.abs(gripper_diff)),
                jnp.sum(jnp.square(gripper_diff)),
                gripper_diff.size,
                jnp.sum(jnp.abs(prefix_norm_diff)),
                jnp.sum(jnp.square(prefix_norm_diff)),
                prefix_norm_diff.size,
                jnp.sum(jnp.abs(prefix_joint_diff)),
                jnp.sum(jnp.square(prefix_joint_diff)),
                prefix_joint_diff.size,
                jnp.sum(jnp.abs(prefix_gripper_diff)),
                jnp.sum(jnp.square(prefix_gripper_diff)),
                prefix_gripper_diff.size,
                jnp.sum(jnp.abs(first_joint_diff)),
                jnp.sum(jnp.square(first_joint_diff)),
                first_joint_diff.size,
                jnp.sum(jnp.abs(first_gripper_diff)),
                jnp.sum(jnp.square(first_gripper_diff)),
                first_gripper_diff.size,
            ],
            dtype=jnp.float32,
        )

    metadata = {
        "config": args.config,
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "steps": args.steps,
        "test_episodes": args.test_episodes,
        "dataset_frames": dataset_size,
        "evaluated_frames": evaluated_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sample_steps": args.sample_steps,
        "model_action_horizon": action_horizon,
        "metric_horizon": metric_horizon,
        "devices": [str(device) for device in jax.devices()],
    }
    results = []
    state = first_state

    for step_index, step in enumerate(args.steps):
        if step_index:
            del state
            gc.collect()
            params = _model.restore_params(
                args.checkpoint_root / str(step) / "params", dtype=jnp.bfloat16, sharding=replicated
            )
            model = config.model.load(params)
            state = nnx.state(model)
            del model, params

        started = time.monotonic()
        metric_sums = np.zeros(len(_metric_names()), dtype=np.float64)
        for batch_index, host_batch in enumerate(batches):
            observation, target_actions = _device_put_batch(host_batch, data_sharding)
            batch_key = jax.random.fold_in(jax.random.key(args.seed), batch_index)
            loss_rng, sample_rng = jax.random.split(batch_key)
            values = eval_batch(state, loss_rng, sample_rng, observation, target_actions)
            metric_sums += np.asarray(values)
            if batch_index == 0:
                jax.block_until_ready(values)
                logging.info("Step %d compiled and started", step)
            elif (batch_index + 1) % 50 == 0 or batch_index + 1 == len(batches):
                jax.block_until_ready(values)
                logging.info("Step %d: %d/%d batches", step, batch_index + 1, len(batches))

        sums = dict(zip(_metric_names(), metric_sums, strict=True))
        result = {
            "checkpoint_step": step,
            "evaluated_frames": evaluated_samples,
            **_finalize_metrics(sums),
            "evaluation_seconds": time.monotonic() - started,
        }
        results.append(result)
        _write_results(args.output_dir, metadata, results)
        logging.info("Step %d metrics: %s", step, result)


if __name__ == "__main__":
    main()
