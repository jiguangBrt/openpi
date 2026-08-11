import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import get_rtc_prefix_weights
from openpi.models.pi0 import rtc_guided_velocity
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_rtc_exponential_prefix_weights():
    weights = np.asarray(get_rtc_prefix_weights(2, 4, 10))
    np.testing.assert_allclose(
        weights,
        np.array([1.0, 1.0, 0.367706, 0.076746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-5,
    )


def test_reverse_time_rtc_guidance_moves_toward_prefix():
    x_t = jnp.zeros((1, 2, 1))
    old_actions = jnp.ones_like(x_t)
    weights = jnp.ones((2,))

    def denoiser(x):
        velocity = jnp.zeros_like(x)
        return x, velocity

    guided_velocity = rtc_guided_velocity(denoiser, x_t, old_actions, weights, 0.5, 5.0)
    assert np.all(np.asarray(guided_velocity) < 0)
    next_x = x_t - 0.1 * guided_velocity
    assert np.all(np.asarray(next_x) > 0)


def test_zero_rtc_weights_leave_velocity_unchanged():
    x_t = jnp.ones((1, 2, 1))
    expected_velocity = jnp.full_like(x_t, 0.25)

    def denoiser(x):
        return x - 0.5 * expected_velocity, expected_velocity

    guided_velocity = rtc_guided_velocity(
        denoiser,
        x_t,
        jnp.zeros_like(x_t),
        jnp.zeros((2,)),
        0.5,
        5.0,
    )
    np.testing.assert_array_equal(np.asarray(guided_velocity), np.asarray(expected_velocity))


def test_rtc_guidance_endpoints_remain_finite():
    x_t = jnp.zeros((1, 2, 1))

    def denoiser(x):
        return x, jnp.zeros_like(x)

    for time in (0.0, 1.0):
        guided_velocity = rtc_guided_velocity(
            denoiser,
            x_t,
            jnp.ones_like(x_t),
            jnp.ones((2,)),
            time,
            5.0,
        )
        assert np.isfinite(np.asarray(guided_velocity)).all()
