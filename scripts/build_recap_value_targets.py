"""Build frame-aligned 201-bin return targets from RECAP episode outcomes."""

import argparse
import json
import pathlib

import numpy as np

from openpi.training import recap


def build_targets(
    outcomes: list[recap.EpisodeOutcome],
    output_path: pathlib.Path,
    *,
    spec: recap.ReCAPSpec,
) -> dict[str, int]:
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    for outcome in sorted(outcomes, key=lambda item: item.episode_index):
        if outcome.source is recap.EpisodeSource.EVALUATION:
            raise ValueError(
                f"evaluation episode {outcome.episode_id} must remain held out and cannot enter value training"
            )
        episode_indices.append(np.full(outcome.num_frames, outcome.episode_index, dtype=np.int64))
        frame_indices.append(np.arange(outcome.num_frames, dtype=np.int64))
        returns.append(recap.discounted_returns(recap.episode_rewards(outcome, spec)))
    if not returns:
        raise ValueError("the outcome manifest is empty")

    episode_index = np.concatenate(episode_indices)
    frame_index = np.concatenate(frame_indices)
    return_value = np.concatenate(returns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        episode_index=episode_index,
        frame_index=frame_index,
        return_value=return_value,
        return_bin=recap.value_bin_indices(return_value, spec),
    )
    metadata = {
        "fps": spec.fps,
        "max_episode_seconds": spec.max_episode_seconds,
        "max_episode_frames": spec.max_episode_frames,
        "num_value_bins": spec.num_value_bins,
        "num_episodes": len(outcomes),
        "num_frames": len(return_value),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-episode-seconds", type=int, default=120)
    args = parser.parse_args()
    metadata = build_targets(
        recap.load_episode_outcomes(args.outcomes),
        args.output,
        spec=recap.ReCAPSpec(fps=args.fps, max_episode_seconds=args.max_episode_seconds),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
