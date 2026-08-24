"""Build pi0.5-RECAP value targets and policy advantage labels.

Value predictions use a compact NPZ contract with equal-length vectors:
``episode_index``, ``frame_index``, and ``value``. Values must already be the
201-bin distribution expectation in [-1, 0].
"""

import argparse
import json
import pathlib

import numpy as np

from openpi.training import recap


def validate_value_metrics(path: pathlib.Path, *, minimum_auroc: float = 0.65) -> dict[str, object]:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read value validation metrics from {path}: {exc}") from exc
    auroc = metrics.get("success_failure_auroc")
    if isinstance(auroc, bool) or not isinstance(auroc, int | float) or not np.isfinite(auroc):
        raise ValueError("value validation has no finite success/failure AUROC; collect both classes")
    if float(auroc) < minimum_auroc:
        raise ValueError(f"value AUROC {float(auroc):.4f} is below the required {minimum_auroc:.2f}")
    if metrics.get("gate_passed") is not True:
        raise ValueError("value validation gate_passed is not true")
    return metrics


def build_sidecar(
    outcomes: list[recap.EpisodeOutcome],
    values_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    baseline_success_rate: float,
    spec: recap.ReCAPSpec,
) -> dict[str, float | int]:
    with np.load(values_path, allow_pickle=False) as data:
        required = {"episode_index", "frame_index", "value"}
        if missing := required - set(data.files):
            raise ValueError(f"value predictions are missing keys: {sorted(missing)}")
        episode_indices = np.asarray(data["episode_index"])
        frame_indices = np.asarray(data["frame_index"])
        predicted_values = np.asarray(data["value"], dtype=np.float32)
    if any(array.ndim != 1 for array in (episode_indices, frame_indices, predicted_values)):
        raise ValueError("value prediction arrays must be vectors")
    if not (len(episode_indices) == len(frame_indices) == len(predicted_values)):
        raise ValueError("value prediction arrays must have equal length")
    if not np.isfinite(predicted_values).all() or np.any((predicted_values < -1.0) | (predicted_values > 0.0)):
        raise ValueError("predicted values must be finite and in [-1, 0]")

    predictions: dict[tuple[int, int], float] = {}
    for episode_index, frame_index, value in zip(episode_indices, frame_indices, predicted_values, strict=True):
        key = (int(episode_index), int(frame_index))
        if key in predictions:
            raise ValueError(f"duplicate value prediction for episode/frame {key}")
        predictions[key] = float(value)

    rows: list[dict[str, object]] = []
    autonomous_advantages: list[np.ndarray] = []
    expected_keys: set[tuple[int, int]] = set()
    for outcome in sorted(outcomes, key=lambda item: item.episode_index):
        if outcome.source is recap.EpisodeSource.EVALUATION:
            raise ValueError(
                f"evaluation episode {outcome.episode_id} must remain held out and cannot enter a training sidecar"
            )
        rewards = recap.episode_rewards(outcome, spec)
        returns = recap.discounted_returns(rewards)
        keys = [(outcome.episode_index, frame_index) for frame_index in range(outcome.num_frames)]
        missing = [key for key in keys if key not in predictions]
        if missing:
            raise ValueError(f"missing {len(missing)} value predictions for episode {outcome.episode_id}")
        values = np.asarray([predictions[key] for key in keys], dtype=np.float32)
        advantages = recap.n_step_advantages(
            values,
            rewards,
            lookahead_frames=spec.advantage_lookahead_frames,
        )
        if outcome.source is recap.EpisodeSource.AUTONOMOUS:
            autonomous_advantages.append(advantages)
        expected_keys.update(keys)
        rows.extend(
            {
                "episode_index": outcome.episode_index,
                "frame_index": frame_index,
                "return_value": returns[frame_index],
                "return_bin": recap.value_bin_indices(returns[frame_index], spec),
                "predicted_value": values[frame_index],
                "advantage": advantages[frame_index],
            }
            for frame_index in range(outcome.num_frames)
        )

    extras = set(predictions) - expected_keys
    if extras:
        raise ValueError(f"value predictions contain {len(extras)} episode/frame keys absent from the manifest")
    if not autonomous_advantages:
        raise ValueError("at least one autonomous episode is required to determine the advantage threshold")

    threshold, percentile = recap.advantage_threshold(
        np.concatenate(autonomous_advantages),
        baseline_success_rate=baseline_success_rate,
    )
    advantages = np.asarray([row["advantage"] for row in rows], dtype=np.float32)
    indicators = recap.advantage_indicators(advantages, threshold)
    autonomous_mask = np.asarray(
        [
            outcome.source is recap.EpisodeSource.AUTONOMOUS
            for outcome in sorted(outcomes, key=lambda item: item.episode_index)
            for _ in range(outcome.num_frames)
        ],
        dtype=bool,
    )
    autonomous_indicators = indicators[autonomous_mask]
    if not autonomous_indicators.any() or autonomous_indicators.all():
        raise ValueError(
            "autonomous advantage labels collapsed to one class; do not start policy training"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        episode_index=np.asarray([row["episode_index"] for row in rows], dtype=np.int64),
        frame_index=np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
        return_value=np.asarray([row["return_value"] for row in rows], dtype=np.float32),
        return_bin=np.asarray([row["return_bin"] for row in rows], dtype=np.int32),
        predicted_value=np.asarray([row["predicted_value"] for row in rows], dtype=np.float32),
        advantage=advantages,
        advantage_indicator=indicators,
    )
    metadata: dict[str, float | int] = {
        "baseline_success_rate": baseline_success_rate,
        "advantage_percentile": percentile,
        "advantage_threshold": threshold,
        "positive_fraction_all": float(indicators.mean()),
        "positive_fraction_autonomous": float(autonomous_indicators.mean()),
        "num_positive_autonomous": int(autonomous_indicators.sum()),
        "num_negative_autonomous": int((~autonomous_indicators).sum()),
        "fps": spec.fps,
        "max_episode_seconds": spec.max_episode_seconds,
        "max_episode_frames": spec.max_episode_frames,
        "lookahead_frames": spec.advantage_lookahead_frames,
        "num_value_bins": spec.num_value_bins,
        "num_frames": len(rows),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=pathlib.Path, required=True)
    parser.add_argument("--values", type=pathlib.Path, required=True)
    parser.add_argument("--validation-metrics", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-success-rate", type=float, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-episode-seconds", type=int, default=120)
    args = parser.parse_args()

    validate_value_metrics(args.validation_metrics)
    spec = recap.ReCAPSpec(fps=args.fps, max_episode_seconds=args.max_episode_seconds)
    metadata = build_sidecar(
        recap.load_episode_outcomes(args.outcomes),
        args.values,
        args.output,
        baseline_success_rate=args.baseline_success_rate,
        spec=spec,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
