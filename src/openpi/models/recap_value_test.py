import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import recap_value


def test_distributional_value_loss_and_expectation() -> None:
    logits = jnp.full((3, 201), -20.0)
    logits = logits.at[np.arange(3), [0, 100, 200]].set(20.0)
    targets = jnp.asarray([0, 100, 200])
    np.testing.assert_allclose(recap_value.distributional_value_loss(logits, targets), 0.0, atol=1e-5)
    np.testing.assert_allclose(recap_value.expected_value_from_logits(logits), [-1.0, -0.5, 0.0], atol=1e-5)


def test_distributional_value_head_uses_last_valid_token() -> None:
    head = recap_value.DistributionalValueHead(4, num_bins=3, rngs=nnx.Rngs(jax.random.key(0)))
    tokens = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    mask = jnp.asarray([[True, True, False, False], [True, True, True, False]])
    logits = head(tokens, mask)
    expected = head.projection(jnp.stack([tokens[0, 1], tokens[1, 2]]))
    np.testing.assert_allclose(logits, expected)
