"""Compute MarvinPro action/state normalization without decoding video frames.

The policy normalization only uses ``observation.state`` and ``action``.  This
utility reads those columns directly from the LeRobot Parquet files, while
matching OpenPI's action horizon and episode-end padding semantics.
"""

import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import tyro

import openpi.shared.normalize as normalize


def _read_vector_column(table: pa.Table, key: str) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{key} must be a vector column, got shape {values.shape}")
    return values


def main(
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path,
    repo_id: str = "stack_cones_slow_260826_train",
    action_horizon: int = 20,
) -> None:
    parquet_files = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode Parquet files found below {dataset_root / 'data'}")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    delta_mask = np.array([True] * 7 + [False] + [True] * 7 + [False])
    total_frames = 0

    for parquet_file in parquet_files:
        table = parquet.read_table(parquet_file, columns=["observation.state", "action", "frame_index"])
        states = _read_vector_column(table, "observation.state")
        actions = _read_vector_column(table, "action")
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        if states.shape != actions.shape or states.shape[1] != 16:
            raise ValueError(f"Unexpected state/action shapes in {parquet_file}: {states.shape}, {actions.shape}")
        if frame_indices.shape != (len(states),):
            raise ValueError(f"Unexpected frame_index shape in {parquet_file}: {frame_indices.shape}")
        if not np.all(frame_indices[:-1] <= frame_indices[1:]):
            order = np.argsort(frame_indices)
            states, actions = states[order], actions[order]
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite state/action values in {parquet_file}")

        query_offsets = np.arange(action_horizon, dtype=np.int64)[None, :]
        query_indices = np.minimum(np.arange(len(actions), dtype=np.int64)[:, None] + query_offsets, len(actions) - 1)
        action_windows = actions[query_indices].copy()
        action_windows[..., delta_mask] -= states[:, None, delta_mask]

        state_stats.update(states)
        action_stats.update(action_windows)
        total_frames += len(states)

    destination = output_dir / repo_id
    normalize.save(destination, {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()})
    print(f"Wrote {total_frames} frames from {len(parquet_files)} episodes to {destination / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
