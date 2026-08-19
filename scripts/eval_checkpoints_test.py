"""Unit tests for the checkpoint evaluation metric helpers."""

# ruff: noqa: SLF001

import math

import pytest

from scripts import eval_checkpoints


def test_resolve_metric_horizon():
    assert eval_checkpoints._resolve_metric_horizon(10, None) == 10
    assert eval_checkpoints._resolve_metric_horizon(20, 10) == 10

    with pytest.raises(ValueError, match="metric horizon must be in 1..20"):
        eval_checkpoints._resolve_metric_horizon(20, 21)


def test_finalize_metrics_includes_prefix_metrics():
    sums = dict.fromkeys(eval_checkpoints._metric_names(), 4.0)
    for name in sums:
        if name.endswith("_count"):
            sums[name] = 2.0

    metrics = eval_checkpoints._finalize_metrics(sums)

    assert metrics["action_prefix_norm_mae"] == 2.0
    assert metrics["action_prefix_norm_rmse"] == math.sqrt(2.0)
    assert metrics["joint_prefix_mae_rad"] == 2.0
    assert metrics["gripper_prefix_mae"] == 2.0
