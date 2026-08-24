import dataclasses
import json

import numpy as np
import pytest

from openpi.training import recap


def _outcome(*, success: bool, num_frames: int = 3, source: str = "autonomous") -> recap.EpisodeOutcome:
    return recap.EpisodeOutcome(
        episode_index=0,
        episode_id="episode-000000",
        source=recap.EpisodeSource(source),
        policy_iteration=0,
        success=success,
        terminal_reason=recap.TerminalReason.SUCCESS if success else recap.TerminalReason.TIMEOUT,
        num_frames=num_frames,
    )


def test_default_spec_uses_120_second_limit() -> None:
    assert recap.DEFAULT_SPEC.max_episode_seconds == 120
    assert recap.DEFAULT_SPEC.max_episode_frames == 1800
    assert recap.DEFAULT_SPEC.fail_penalty_frames == 1800
    assert recap.DEFAULT_SPEC.advantage_lookahead_frames == 15


def test_success_and_failure_returns() -> None:
    success_rewards = recap.episode_rewards(_outcome(success=True))
    success_returns = recap.discounted_returns(success_rewards)
    np.testing.assert_allclose(success_returns, np.asarray([-2 / 1800, -1 / 1800, 0], dtype=np.float32))

    failure_rewards = recap.episode_rewards(_outcome(success=False))
    failure_returns = recap.discounted_returns(failure_rewards)
    np.testing.assert_array_equal(failure_returns, np.full(3, -1.0, dtype=np.float32))


def test_episode_over_120_seconds_is_rejected() -> None:
    with pytest.raises(ValueError, match="above the 120s limit"):
        recap.episode_rewards(_outcome(success=False, num_frames=1801))


def test_value_bin_mapping_and_expectation() -> None:
    values = np.asarray([-1.0, -0.5, 0.0])
    np.testing.assert_array_equal(recap.value_bin_indices(values), np.asarray([0, 100, 200]))
    probabilities = np.zeros((3, 201), dtype=np.float32)
    probabilities[np.arange(3), [0, 100, 200]] = 1
    np.testing.assert_allclose(recap.value_bin_expectation(probabilities), values)


def test_n_step_advantage_bootstraps_only_before_terminal() -> None:
    values = np.asarray([-0.5, -0.4, -0.3, -0.2], dtype=np.float32)
    rewards = np.asarray([-0.01, -0.01, -0.01, 0.0], dtype=np.float32)
    advantages = recap.n_step_advantages(values, rewards, lookahead_frames=2)
    assert advantages[0] == pytest.approx(0.18)
    assert advantages[1] == pytest.approx(0.18)
    assert advantages[2] == pytest.approx(0.29)
    assert advantages[3] == pytest.approx(0.2)


def test_advantage_threshold_switches_at_80_percent_success() -> None:
    advantages = np.arange(100, dtype=np.float32)
    threshold, percentile = recap.advantage_threshold(advantages, baseline_success_rate=0.79)
    assert percentile == 60
    assert recap.advantage_indicators(advantages, threshold).sum() == 40

    threshold, percentile = recap.advantage_threshold(advantages, baseline_success_rate=0.8)
    assert percentile == 90
    assert recap.advantage_indicators(advantages, threshold).sum() == 10


def test_episode_split_has_no_overlap() -> None:
    train, validation = recap.split_episode_ids([f"episode-{index}" for index in range(10)], seed=7)
    assert len(train) == 8
    assert len(validation) == 2
    assert not set(train) & set(validation)
    assert (train, validation) == recap.split_episode_ids([f"episode-{index}" for index in range(10)], seed=7)


def test_episode_outcome_split_is_stratified() -> None:
    outcomes = [
        dataclasses.replace(
            _outcome(success=index % 2 == 0),
            episode_index=index,
            episode_id=f"episode-{index}",
        )
        for index in range(10)
    ]
    train, validation = recap.split_episode_outcomes(outcomes, seed=7)
    assert len(train) == 8
    assert len(validation) == 2
    assert {outcome.success for outcome in validation} == {False, True}
    assert not {outcome.episode_id for outcome in train} & {outcome.episode_id for outcome in validation}


def test_outcome_round_trip_and_sidecar_alignment(tmp_path) -> None:
    manifest = tmp_path / "outcomes.jsonl"
    recap.write_episode_outcomes(manifest, [_outcome(success=True)])
    assert recap.load_episode_outcomes(manifest) == [_outcome(success=True)]
    row = json.loads(manifest.read_text())
    assert row["terminal_reason"] == "success"

    sidecar_path = tmp_path / "labels.npz"
    np.savez_compressed(
        sidecar_path,
        episode_index=np.asarray([0, 0]),
        frame_index=np.asarray([0, 1]),
        advantage_indicator=np.asarray([True, False]),
    )
    sidecar = recap.ReCAPSidecar(sidecar_path)
    assert sidecar.indicator(0, 0)
    assert not sidecar.indicator(0, 1)
    with pytest.raises(KeyError, match="no label"):
        sidecar.indicator(0, 2)

    targets_path = tmp_path / "targets.npz"
    np.savez_compressed(
        targets_path,
        episode_index=np.asarray([0, 0]),
        frame_index=np.asarray([0, 1]),
        return_bin=np.asarray([12, 200]),
    )
    targets = recap.ReCAPValueTargets(targets_path)
    assert targets.target(0, 0) == 12
    assert targets.target(0, 1) == 200


def test_binary_auroc_handles_ties() -> None:
    labels = np.asarray([False, True, False, True])
    assert recap.binary_auroc(labels, np.asarray([0.0, 1.0, 0.0, 1.0])) == 1.0
    assert recap.binary_auroc(labels, np.ones(4)) == 0.5


def test_value_validation_metrics_and_gate() -> None:
    outcomes = [
        dataclasses.replace(_outcome(success=True, num_frames=2), episode_index=0, episode_id="success"),
        dataclasses.replace(_outcome(success=False, num_frames=2), episode_index=1, episode_id="failure"),
    ]
    metrics = recap.evaluate_value_predictions(
        outcomes,
        episode_indices=np.asarray([0, 0, 1, 1]),
        frame_indices=np.asarray([0, 1, 0, 1]),
        predicted_values=np.asarray([-1 / 1800, 0.0, -1.0, -1.0]),
    )
    assert metrics["success_failure_auroc"] == 1.0
    assert metrics["successful_remaining_time_mae_seconds"] == pytest.approx(0.0)
    assert metrics["gate_passed"] is True
