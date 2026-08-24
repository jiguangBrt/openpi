"""Append and validate immutable RECAP episode outcome labels."""

import argparse
import pathlib

from openpi.training import recap


def _add(args: argparse.Namespace) -> None:
    existing = recap.load_episode_outcomes(args.manifest) if args.manifest.exists() else []
    outcome = recap.EpisodeOutcome(
        episode_index=args.episode_index,
        episode_id=args.episode_id,
        source=recap.EpisodeSource(args.source),
        policy_iteration=args.policy_iteration,
        success=args.terminal_reason == recap.TerminalReason.SUCCESS.value,
        terminal_reason=recap.TerminalReason(args.terminal_reason),
        num_frames=args.num_frames,
        fps=args.fps,
        started_at=args.started_at,
        ended_at=args.ended_at,
    )
    recap.write_episode_outcomes(args.manifest, [*existing, outcome])
    print(f"recorded {outcome.episode_id} ({outcome.terminal_reason.value}, {outcome.num_frames} frames)")


def _validate(args: argparse.Namespace) -> None:
    outcomes = recap.load_episode_outcomes(args.manifest)
    if args.expected_episodes is not None and len(outcomes) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} outcomes, found {len(outcomes)}")
    counts = {source.value: 0 for source in recap.EpisodeSource}
    successes = 0
    for outcome in outcomes:
        counts[outcome.source.value] += 1
        successes += int(outcome.success)
    print(f"valid outcomes={len(outcomes)} successes={successes} sources={counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    add = subparsers.add_parser("add")
    add.add_argument("--manifest", type=pathlib.Path, required=True)
    add.add_argument("--episode-index", type=int, required=True)
    add.add_argument("--episode-id", required=True)
    add.add_argument("--source", choices=[source.value for source in recap.EpisodeSource], required=True)
    add.add_argument("--policy-iteration", type=int, required=True)
    add.add_argument("--terminal-reason", choices=[reason.value for reason in recap.TerminalReason], required=True)
    add.add_argument("--num-frames", type=int, required=True)
    add.add_argument("--fps", type=int, default=15)
    add.add_argument("--started-at")
    add.add_argument("--ended-at")
    add.set_defaults(handler=_add)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=pathlib.Path, required=True)
    validate.add_argument("--expected-episodes", type=int)
    validate.set_defaults(handler=_validate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
