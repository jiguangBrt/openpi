import json

import numpy as np
import pytest

from openpi.training import recap
from scripts import build_recap_sidecar
from scripts import build_recap_value_targets


def test_build_sidecar_synthetic_pipeline(tmp_path) -> None:
    outcomes = [
        recap.EpisodeOutcome(
            episode_index=0,
            episode_id="demo-000",
            source=recap.EpisodeSource.DEMONSTRATION,
            policy_iteration=0,
            success=True,
            terminal_reason=recap.TerminalReason.SUCCESS,
            num_frames=3,
        ),
        recap.EpisodeOutcome(
            episode_index=1,
            episode_id="rollout-000",
            source=recap.EpisodeSource.AUTONOMOUS,
            policy_iteration=0,
            success=False,
            terminal_reason=recap.TerminalReason.TIMEOUT,
            num_frames=3,
        ),
    ]
    values_path = tmp_path / "values.npz"
    np.savez_compressed(
        values_path,
        episode_index=np.repeat([0, 1], 3),
        frame_index=np.tile(np.arange(3), 2),
        value=np.asarray([-0.3, -0.2, -0.1, -0.9, -0.8, -0.7], dtype=np.float32),
    )
    output_path = tmp_path / "labels.npz"
    targets_path = tmp_path / "targets.npz"
    target_metadata = build_recap_value_targets.build_targets(
        outcomes,
        targets_path,
        spec=recap.DEFAULT_SPEC,
    )
    assert target_metadata["num_frames"] == 6
    targets = recap.ReCAPValueTargets(targets_path)
    assert len(targets) == 6

    metadata = build_recap_sidecar.build_sidecar(
        outcomes,
        values_path,
        output_path,
        baseline_success_rate=0.5,
        spec=recap.DEFAULT_SPEC,
    )
    assert metadata["max_episode_frames"] == 1800
    assert metadata["advantage_percentile"] == 60
    assert 0.0 < metadata["positive_fraction_autonomous"] < 1.0
    sidecar = recap.ReCAPSidecar(output_path)
    assert len(sidecar) == 6
    with np.load(output_path) as data:
        assert data["return_bin"].shape == (6,)
        assert data["advantage_indicator"].dtype == np.bool_


def test_value_metrics_gate_rejects_low_auroc(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"success_failure_auroc": 0.64, "gate_passed": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="below the required"):
        build_recap_sidecar.validate_value_metrics(metrics_path)
