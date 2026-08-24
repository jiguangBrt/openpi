import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0


def _observation() -> _model.Observation:
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((2, 4), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray([[1, 1], [1, 1]], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((2, 2), dtype=bool),
        tokenized_prompt_with_advantage=jnp.asarray([[2, 2], [3, 3]], dtype=jnp.int32),
        tokenized_prompt_with_advantage_mask=jnp.ones((2, 2), dtype=bool),
    )


def test_recap_condition_dropout_extremes() -> None:
    observation = _observation()
    conditioned = pi0.apply_recap_condition_dropout(jax.random.key(0), observation, dropout_prob=0.0)
    np.testing.assert_array_equal(conditioned.tokenized_prompt, [[2, 2], [3, 3]])

    unconditioned = pi0.apply_recap_condition_dropout(jax.random.key(0), observation, dropout_prob=1.0)
    np.testing.assert_array_equal(unconditioned.tokenized_prompt, [[1, 1], [1, 1]])


def test_observation_requires_recap_token_pair() -> None:
    data = {
        "image": {},
        "image_mask": {},
        "state": jnp.zeros((1, 4)),
        "tokenized_prompt_with_advantage": jnp.ones((1, 2), dtype=jnp.int32),
    }
    with pytest.raises(ValueError, match="tokenized_prompt_with_advantage"):
        _model.Observation.from_dict(data)
