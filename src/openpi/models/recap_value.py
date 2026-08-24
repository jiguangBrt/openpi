"""Distributional value model used by the pi0.5-RECAP approximation."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import recap


def distributional_value_loss(logits: jax.Array, target_bins: jax.Array) -> jax.Array:
    """Per-sample cross entropy for integer value-bin targets."""

    logits = jnp.asarray(logits, dtype=jnp.float32)
    target_bins = jnp.asarray(target_bins, dtype=jnp.int32)
    if logits.shape[:-1] != target_bins.shape:
        raise ValueError(f"logits batch shape {logits.shape[:-1]} does not match targets {target_bins.shape}")
    selected = jnp.take_along_axis(jax.nn.log_softmax(logits), target_bins[..., None], axis=-1)[..., 0]
    return -selected


def expected_value_from_logits(logits: jax.Array) -> jax.Array:
    logits = jnp.asarray(logits, dtype=jnp.float32)
    centers = jnp.linspace(-1.0, 0.0, logits.shape[-1], dtype=jnp.float32)
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * centers, axis=-1)


class DistributionalValueHead(nnx.Module):
    """Pool the last valid VLM token and classify it into return bins."""

    def __init__(self, embedding_dim: int, *, num_bins: int = recap.DEFAULT_SPEC.num_value_bins, rngs: nnx.Rngs):
        self.num_bins = num_bins
        self.projection = nnx.Linear(embedding_dim, num_bins, dtype=jnp.float32, rngs=rngs)

    def __call__(self, encoded_tokens: jax.Array, token_mask: jax.Array) -> jax.Array:
        if encoded_tokens.ndim != 3 or token_mask.shape != encoded_tokens.shape[:2]:
            raise ValueError("encoded_tokens must be [batch, tokens, width] with a matching token mask")
        last_indices = jnp.maximum(jnp.sum(token_mask, axis=-1, dtype=jnp.int32) - 1, 0)
        pooled = encoded_tokens[jnp.arange(encoded_tokens.shape[0]), last_indices]
        return self.projection(pooled.astype(jnp.float32))


class ReCAPValueModel(pi0.Pi0):
    """Pi0.5 observation backbone plus an independent 201-bin value head.

    The inherited action expert is unused. Keeping the original parameter paths
    allows H20 policy checkpoints to initialize the shared SigLIP/Gemma backbone.
    """

    def __init__(
        self,
        config: pi0_config.Pi0Config,
        *,
        num_bins: int = recap.DEFAULT_SPEC.num_value_bins,
        rngs: nnx.Rngs,
    ):
        if not config.pi05:
            raise ValueError("ReCAPValueModel requires a pi0.5 config")
        super().__init__(config, rngs)
        width = _gemma.get_config(config.paligemma_variant).width
        self.value_head = DistributionalValueHead(width, num_bins=num_bins, rngs=rngs)

    def value_logits(
        self,
        observation: _model.Observation,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> jax.Array:
        if train and rng is None:
            raise ValueError("training value logits requires an augmentation RNG")
        observation = _model.preprocess_observation(rng, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attention_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (encoded_tokens, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attention_mask,
            positions=positions,
        )
        assert encoded_tokens is not None
        return self.value_head(encoded_tokens, prefix_mask)

    def compute_value_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        target_bins: at.Int[at.Array, " b"],
        *,
        train: bool = False,
    ) -> jax.Array:
        return distributional_value_loss(self.value_logits(observation, train=train, rng=rng), target_bins)

    def predict_value(self, observation: _model.Observation) -> jax.Array:
        return expected_value_from_logits(self.value_logits(observation, train=False))


def value_freeze_filter() -> nnx.filterlib.Filter:
    """Train only a fresh main-VLM LoRA and the distributional value head."""

    main_vlm_lora = nnx.All(
        nnx_utils.PathRegex(".*llm.*lora.*"),
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    )
    trainable = nnx.Any(main_vlm_lora, nnx_utils.PathRegex(".*value_head.*"))
    return nnx.All(nnx.Param, nnx.Not(trainable))
