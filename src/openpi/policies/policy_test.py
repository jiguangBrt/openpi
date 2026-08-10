import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.policies import marvinpro_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


class _FakeRtcModel:
    action_horizon = 10
    action_dim = 32

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        return jnp.zeros((1, self.action_horizon, self.action_dim))

    def sample_actions_rtc(self, rng, observation, *, old_actions, **kwargs):
        del rng, observation, kwargs
        return old_actions


def test_marvinpro_rtc_prefix_round_trip(monkeypatch):
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda function: function)
    stats = {
        "state": _normalize.NormStats(mean=np.linspace(-0.2, 0.2, 16), std=np.linspace(0.5, 1.5, 16)),
        "actions": _normalize.NormStats(mean=np.linspace(0.1, 0.3, 16), std=np.linspace(0.8, 1.8, 16)),
    }
    delta_mask = transforms.make_bool_mask(7, -1, 7, -1)
    policy = _policy.Policy(
        _FakeRtcModel(),
        transforms=[
            marvinpro_policy.MarvinProInputs(),
            transforms.DeltaActions(delta_mask),
            transforms.Normalize(stats),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(stats),
            transforms.AbsoluteActions(delta_mask),
            marvinpro_policy.MarvinProOutputs(),
        ],
    )
    state = np.linspace(-0.7, 0.7, 16, dtype=np.float32)
    old_prefix = np.stack([state + index * 0.01 for index in range(6)]).astype(np.float32)
    request = {
        "schedule": "exp",
        "d_pred": 2,
        "s": 4,
        "beta": 5.0,
        "observation": {
            "state": state,
            "images": {
                "cam_high": np.zeros((2, 2, 3), dtype=np.uint8),
                "cam_left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
                "cam_right_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            },
        },
        "old_remaining_actions_absolute": old_prefix,
    }

    result = policy.infer_rtc(request)

    expected = np.concatenate([old_prefix, np.repeat(old_prefix[-1:], 4, axis=0)])
    assert result["actions"].shape == (10, 16)
    np.testing.assert_allclose(result["actions"], expected, atol=1e-5)
    assert policy.metadata["rtc"]["protocol"] == "rtc_v1"
    assert set(result["rtc_timing"]) == {
        "preprocess_ms",
        "denoise_ms",
        "postprocess_ms",
        "total_ms",
    }

    invalid_requests = (
        {**request, "d_pred": 5},
        {**request, "s": 3},
        {**request, "schedule": "linear"},
        {**request, "old_remaining_actions_absolute": np.zeros((5, 16), dtype=np.float32)},
    )
    for invalid_request in invalid_requests:
        with pytest.raises(_policy.RtcPolicyError):
            policy.infer_rtc(invalid_request)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
