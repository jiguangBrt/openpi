"""RECAP data contracts, return targets, and advantage labels.

This module implements the public algorithmic pieces from RECAP without claiming
compatibility with the unreleased pi0.6* architecture or weights.
"""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import enum
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


class EpisodeSource(enum.StrEnum):
    DEMONSTRATION = "demonstration"
    AUTONOMOUS = "autonomous"
    EVALUATION = "evaluation"


class TerminalReason(enum.StrEnum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    SAFETY_STOP = "safety_stop"
    OUT_OF_WORKSPACE = "out_of_workspace"
    OPERATOR_ABORT = "operator_abort"
    TASK_FAILURE = "task_failure"


@dataclasses.dataclass(frozen=True)
class ReCAPSpec:
    """Task-time and distributional-value constants for pi0.5-RECAP."""

    fps: int = 15
    max_episode_seconds: int = 120
    num_value_bins: int = 201
    advantage_lookahead_seconds: float = 1.0
    conditioning_dropout_prob: float = 0.3

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        if self.num_value_bins < 2:
            raise ValueError("num_value_bins must be at least 2")
        if self.advantage_lookahead_seconds <= 0:
            raise ValueError("advantage_lookahead_seconds must be positive")
        if not 0.0 <= self.conditioning_dropout_prob <= 1.0:
            raise ValueError("conditioning_dropout_prob must be in [0, 1]")

    @property
    def max_episode_frames(self) -> int:
        return self.fps * self.max_episode_seconds

    @property
    def fail_penalty_frames(self) -> int:
        return self.max_episode_frames

    @property
    def advantage_lookahead_frames(self) -> int:
        return round(self.fps * self.advantage_lookahead_seconds)

    @property
    def value_bin_centers(self) -> np.ndarray:
        return np.linspace(-1.0, 0.0, self.num_value_bins, dtype=np.float32)


DEFAULT_SPEC = ReCAPSpec()


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    episode_index: int
    episode_id: str
    source: EpisodeSource
    policy_iteration: int
    success: bool
    terminal_reason: TerminalReason
    num_frames: int
    fps: int = DEFAULT_SPEC.fps
    started_at: str | None = None
    ended_at: str | None = None

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if self.policy_iteration < 0:
            raise ValueError("policy_iteration must be non-negative")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.success != (self.terminal_reason is TerminalReason.SUCCESS):
            raise ValueError("success must be true exactly when terminal_reason is 'success'")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeOutcome":
        return cls(
            episode_index=int(data["episode_index"]),
            episode_id=str(data["episode_id"]),
            source=EpisodeSource(data["source"]),
            policy_iteration=int(data["policy_iteration"]),
            success=bool(data["success"]),
            terminal_reason=TerminalReason(data["terminal_reason"]),
            num_frames=int(data["num_frames"]),
            fps=int(data.get("fps", DEFAULT_SPEC.fps)),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "source": self.source.value,
            "terminal_reason": self.terminal_reason.value,
        }


def load_episode_outcomes(path: str | pathlib.Path) -> list[EpisodeOutcome]:
    path = pathlib.Path(path)
    outcomes: list[EpisodeOutcome] = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                outcome = EpisodeOutcome.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid outcome at {path}:{line_number}: {exc}") from exc
            if outcome.episode_index in seen_indices:
                raise ValueError(f"duplicate episode_index {outcome.episode_index} in {path}")
            if outcome.episode_id in seen_ids:
                raise ValueError(f"duplicate episode_id {outcome.episode_id!r} in {path}")
            seen_indices.add(outcome.episode_index)
            seen_ids.add(outcome.episode_id)
            outcomes.append(outcome)
    return outcomes


def write_episode_outcomes(path: str | pathlib.Path, outcomes: Iterable[EpisodeOutcome]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(outcomes)
    _validate_unique_outcomes(rows)
    with path.open("w", encoding="utf-8") as file:
        for outcome in rows:
            file.write(json.dumps(outcome.to_dict(), sort_keys=True) + "\n")


def _validate_unique_outcomes(outcomes: Sequence[EpisodeOutcome]) -> None:
    indices = [outcome.episode_index for outcome in outcomes]
    ids = [outcome.episode_id for outcome in outcomes]
    if len(indices) != len(set(indices)):
        raise ValueError("episode_index values must be unique")
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be unique")


def episode_rewards(outcome: EpisodeOutcome, spec: ReCAPSpec = DEFAULT_SPEC) -> np.ndarray:
    """Return normalized per-frame rewards in [-1, 0]."""

    if outcome.fps != spec.fps:
        raise ValueError(f"episode {outcome.episode_id} has fps={outcome.fps}, expected {spec.fps}")
    if outcome.num_frames > spec.max_episode_frames:
        raise ValueError(
            f"episode {outcome.episode_id} has {outcome.num_frames} frames, "
            f"above the {spec.max_episode_seconds}s limit ({spec.max_episode_frames} frames)"
        )
    rewards = np.full(outcome.num_frames, -1.0 / spec.max_episode_frames, dtype=np.float32)
    rewards[-1] = 0.0 if outcome.success else -1.0
    return rewards


def discounted_returns(rewards: np.ndarray) -> np.ndarray:
    """Compute undiscounted, clipped returns for already-normalized rewards."""

    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1 or rewards.size == 0 or not np.isfinite(rewards).all():
        raise ValueError("rewards must be a non-empty finite vector")
    returns = np.cumsum(rewards[::-1], dtype=np.float64)[::-1]
    return np.clip(returns, -1.0, 0.0).astype(np.float32)


def value_bin_indices(values: np.ndarray, spec: ReCAPSpec = DEFAULT_SPEC) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    scaled = (np.clip(values, -1.0, 0.0) + 1.0) * (spec.num_value_bins - 1)
    return np.rint(scaled).astype(np.int32)


def value_bin_expectation(probabilities: np.ndarray, spec: ReCAPSpec = DEFAULT_SPEC) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape[-1] != spec.num_value_bins:
        raise ValueError(f"last probability dimension must be {spec.num_value_bins}, got {probabilities.shape[-1]}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("probabilities must have positive mass")
    normalized = probabilities / totals
    return np.sum(normalized * spec.value_bin_centers, axis=-1)


def n_step_advantages(
    values: np.ndarray,
    rewards: np.ndarray,
    *,
    lookahead_frames: int = DEFAULT_SPEC.advantage_lookahead_frames,
) -> np.ndarray:
    """Compute N-step advantages, omitting bootstrap whenever N crosses the terminal frame."""

    values = np.asarray(values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    if values.ndim != 1 or rewards.ndim != 1 or values.shape != rewards.shape or values.size == 0:
        raise ValueError("values and rewards must be non-empty vectors with equal shape")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError("values and rewards must be finite")
    if lookahead_frames <= 0:
        raise ValueError("lookahead_frames must be positive")

    cumulative = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(rewards, dtype=np.float64)])
    advantages = np.empty_like(values)
    terminal = len(values)
    for frame in range(terminal):
        end = min(frame + lookahead_frames, terminal)
        reward_sum = cumulative[end] - cumulative[frame]
        bootstrap = float(values[end]) if end < terminal else 0.0
        advantages[frame] = reward_sum + bootstrap - float(values[frame])
    return advantages


def advantage_threshold(advantages: np.ndarray, *, baseline_success_rate: float) -> tuple[float, int]:
    advantages = np.asarray(advantages, dtype=np.float32)
    if advantages.ndim != 1 or advantages.size == 0 or not np.isfinite(advantages).all():
        raise ValueError("advantages must be a non-empty finite vector")
    if not 0.0 <= baseline_success_rate <= 1.0:
        raise ValueError("baseline_success_rate must be in [0, 1]")
    percentile = 90 if baseline_success_rate >= 0.8 else 60
    return float(np.percentile(advantages, percentile)), percentile


def advantage_indicators(advantages: np.ndarray, threshold: float) -> np.ndarray:
    advantages = np.asarray(advantages, dtype=np.float32)
    if not np.isfinite(advantages).all() or not np.isfinite(threshold):
        raise ValueError("advantages and threshold must be finite")
    return advantages >= threshold


def split_episode_ids(
    episode_ids: Sequence[str], *, train_fraction: float = 0.8, seed: int = 0
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Make a deterministic episode-level split, never a frame-level split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    unique_ids = tuple(dict.fromkeys(episode_ids))
    if len(unique_ids) != len(episode_ids):
        raise ValueError("episode_ids must be unique")

    def score(episode_id: str) -> bytes:
        return hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()

    ordered = sorted(unique_ids, key=score)
    train_count = round(len(ordered) * train_fraction)
    if len(ordered) > 1:
        train_count = min(max(train_count, 1), len(ordered) - 1)
    return tuple(ordered[:train_count]), tuple(ordered[train_count:])


def split_episode_outcomes(
    outcomes: Sequence[EpisodeOutcome],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[tuple[EpisodeOutcome, ...], tuple[EpisodeOutcome, ...]]:
    """Deterministically split whole episodes, stratified by terminal success when possible."""

    _validate_unique_outcomes(outcomes)
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    for success in (False, True):
        group = [outcome.episode_id for outcome in outcomes if outcome.success is success]
        if not group:
            continue
        group_train, group_validation = split_episode_ids(group, train_fraction=train_fraction, seed=seed)
        train_ids.update(group_train)
        validation_ids.update(group_validation)
    train = tuple(outcome for outcome in outcomes if outcome.episode_id in train_ids)
    validation = tuple(outcome for outcome in outcomes if outcome.episode_id in validation_ids)
    if set(train_ids) & set(validation_ids) or len(train) + len(validation) != len(outcomes):
        raise AssertionError("episode split must be disjoint and exhaustive")
    return train, validation


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute tie-aware binary AUROC without an optional sklearn dependency."""

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("labels and scores must be equal-length finite vectors")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one positive and one negative")

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def evaluate_value_predictions(
    outcomes: Sequence[EpisodeOutcome],
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    predicted_values: np.ndarray,
    *,
    spec: ReCAPSpec = DEFAULT_SPEC,
    minimum_auroc: float = 0.65,
) -> dict[str, float | int | bool | None]:
    """Evaluate frame-aligned values on a held-out episode split."""

    _validate_unique_outcomes(outcomes)
    episode_indices = np.asarray(episode_indices)
    frame_indices = np.asarray(frame_indices)
    predicted_values = np.asarray(predicted_values, dtype=np.float32)
    arrays = (episode_indices, frame_indices, predicted_values)
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("prediction arrays must be equal-length vectors")
    if not np.isfinite(predicted_values).all() or np.any((predicted_values < -1.0) | (predicted_values > 0.0)):
        raise ValueError("predicted values must be finite and in [-1, 0]")

    prediction_by_key: dict[tuple[int, int], float] = {}
    for episode_index, frame_index, value in zip(episode_indices, frame_indices, predicted_values, strict=True):
        key = (int(episode_index), int(frame_index))
        if key in prediction_by_key:
            raise ValueError(f"duplicate value prediction for episode/frame {key}")
        prediction_by_key[key] = float(value)

    scores: list[float] = []
    labels: list[bool] = []
    successful_errors_seconds: list[float] = []
    expected_keys: set[tuple[int, int]] = set()
    for outcome in outcomes:
        returns = discounted_returns(episode_rewards(outcome, spec))
        for frame_index, target in enumerate(returns):
            key = (outcome.episode_index, frame_index)
            if key not in prediction_by_key:
                raise ValueError(f"missing value prediction for episode/frame {key}")
            prediction = prediction_by_key[key]
            expected_keys.add(key)
            scores.append(prediction)
            labels.append(outcome.success)
            if outcome.success:
                successful_errors_seconds.append(abs(prediction - float(target)) * spec.max_episode_seconds)

    extras = set(prediction_by_key) - expected_keys
    if extras:
        raise ValueError(f"value predictions contain {len(extras)} keys outside the evaluation split")
    try:
        auroc: float | None = binary_auroc(np.asarray(labels), np.asarray(scores))
    except ValueError:
        auroc = None
    remaining_time_mae = (
        float(np.mean(successful_errors_seconds)) if successful_errors_seconds else None
    )
    gate_passed = auroc is not None and auroc >= minimum_auroc
    return {
        "success_failure_auroc": auroc,
        "successful_remaining_time_mae_seconds": remaining_time_mae,
        "minimum_auroc": minimum_auroc,
        "gate_passed": gate_passed,
        "num_episodes": len(outcomes),
        "num_success_episodes": sum(outcome.success for outcome in outcomes),
        "num_failure_episodes": sum(not outcome.success for outcome in outcomes),
        "num_frames": len(scores),
    }


class ReCAPSidecar:
    """Strict `(episode_index, frame_index)` lookup for policy advantage labels."""

    REQUIRED_KEYS = ("episode_index", "frame_index", "advantage_indicator")

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            missing = set(self.REQUIRED_KEYS) - set(data.files)
            if missing:
                raise ValueError(f"RECAP sidecar {self.path} is missing keys: {sorted(missing)}")
            arrays = {key: np.asarray(data[key]) for key in self.REQUIRED_KEYS}
        lengths = {len(array) for array in arrays.values() if array.ndim == 1}
        if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
            raise ValueError(f"RECAP sidecar {self.path} arrays must be equal-length vectors")

        self._labels: dict[tuple[int, int], bool] = {}
        for episode_index, frame_index, indicator in zip(
            arrays["episode_index"], arrays["frame_index"], arrays["advantage_indicator"], strict=True
        ):
            key = (int(episode_index), int(frame_index))
            if key in self._labels:
                raise ValueError(f"duplicate RECAP sidecar key {key} in {self.path}")
            self._labels[key] = bool(indicator)

    def __len__(self) -> int:
        return len(self._labels)

    def indicator(self, episode_index: int, frame_index: int) -> bool:
        key = (int(episode_index), int(frame_index))
        try:
            return self._labels[key]
        except KeyError as exc:
            raise KeyError(f"RECAP sidecar {self.path} has no label for episode/frame {key}") from exc


class ReCAPValueTargets:
    """Strict `(episode_index, frame_index)` lookup for 201-bin return targets."""

    REQUIRED_KEYS = ("episode_index", "frame_index", "return_bin")

    def __init__(self, path: str | pathlib.Path, *, num_bins: int = DEFAULT_SPEC.num_value_bins):
        self.path = pathlib.Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            missing = set(self.REQUIRED_KEYS) - set(data.files)
            if missing:
                raise ValueError(f"RECAP value sidecar {self.path} is missing keys: {sorted(missing)}")
            arrays = {key: np.asarray(data[key]) for key in self.REQUIRED_KEYS}
        lengths = {len(array) for array in arrays.values() if array.ndim == 1}
        if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
            raise ValueError(f"RECAP value sidecar {self.path} arrays must be equal-length vectors")

        self._targets: dict[tuple[int, int], int] = {}
        for episode_index, frame_index, return_bin in zip(
            arrays["episode_index"], arrays["frame_index"], arrays["return_bin"], strict=True
        ):
            key = (int(episode_index), int(frame_index))
            target = int(return_bin)
            if not 0 <= target < num_bins:
                raise ValueError(f"return_bin {target} at {key} is outside [0, {num_bins})")
            if key in self._targets:
                raise ValueError(f"duplicate RECAP value sidecar key {key} in {self.path}")
            self._targets[key] = target

    def __len__(self) -> int:
        return len(self._targets)

    def target(self, episode_index: int, frame_index: int) -> int:
        key = (int(episode_index), int(frame_index))
        try:
            return self._targets[key]
        except KeyError as exc:
            raise KeyError(f"RECAP value sidecar {self.path} has no target for episode/frame {key}") from exc
