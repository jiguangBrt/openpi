from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class RtcPolicyError(ValueError):
    pass


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_rtc = (
                nnx_utils.module_jit(model.sample_actions_rtc) if hasattr(model, "sample_actions_rtc") else None
            )
            self._rng = rng or jax.random.key(0)
        self._rtc_supported = (
            not self._is_pytorch_model
            and getattr(self, "_sample_actions_rtc", None) is not None
            and getattr(model, "action_horizon", None) == 10
            and getattr(model, "action_dim", None) == 32
            and any(type(transform).__name__ == "MarvinProInputs" for transform in transforms)
        )
        if self._rtc_supported:
            self._metadata["rtc"] = {
                "protocol": "rtc_v1",
                "action_horizon": 10,
                "native_action_dim": 16,
                "model_action_dim": 32,
                "execution_horizon": 4,
                "max_predicted_delay": 4,
                "prefix_attention_schedule": "exp",
            }

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_rtc(self, request: dict) -> dict:
        if not self._rtc_supported:
            raise RtcPolicyError("this policy does not support rtc_v1")
        if request.get("schedule") != "exp":
            raise RtcPolicyError("rtc_v1 schedule must be 'exp'")
        predicted_delay = request.get("d_pred")
        execution_horizon = request.get("s")
        if not isinstance(predicted_delay, int) or not 1 <= predicted_delay <= 4:
            raise RtcPolicyError("rtc_v1 d_pred must be an integer in 1..4")
        if execution_horizon != 4:
            raise RtcPolicyError("rtc_v1 s must be 4")
        max_guidance_weight = request.get("beta", 5.0)
        if not isinstance(max_guidance_weight, int | float) or not 0 < max_guidance_weight <= 10:
            raise RtcPolicyError("rtc_v1 beta must be in (0, 10]")
        observation_input = request.get("observation")
        if not isinstance(observation_input, dict):
            raise RtcPolicyError("RTC observation must be a dictionary")
        old_remaining = np.asarray(request.get("old_remaining_actions_absolute"), dtype=np.float32)
        if old_remaining.shape != (6, 16) or not np.isfinite(old_remaining).all():
            raise RtcPolicyError(
                f"old_remaining_actions_absolute must have finite shape (6, 16), got {old_remaining.shape}"
            )

        rtc_started = preprocess_started = time.monotonic()
        padded_prefix = np.concatenate(
            [old_remaining, np.repeat(old_remaining[-1:, :], repeats=4, axis=0)],
            axis=0,
        )
        inputs = jax.tree.map(lambda x: x, observation_input)
        inputs["actions"] = padded_prefix
        inputs = self._input_transform(inputs)
        try:
            old_actions = inputs.pop("actions")
        except KeyError as exc:
            raise RtcPolicyError("policy input transforms removed the RTC action prefix") from exc
        old_actions = np.asarray(old_actions)
        if old_actions.shape != (10, 32) or not np.isfinite(old_actions).all():
            raise RtcPolicyError(f"transformed RTC prefix must have shape (10, 32), got {old_actions.shape}")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        old_actions_device = jnp.asarray(old_actions)[np.newaxis, ...]
        observation = _model.Observation.from_dict(inputs)
        preprocess_ms = (time.monotonic() - preprocess_started) * 1000.0

        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs.update(
            old_actions=old_actions_device,
            predicted_delay_steps=jnp.asarray(predicted_delay, dtype=jnp.int32),
            execution_horizon=jnp.asarray(execution_horizon, dtype=jnp.int32),
            max_guidance_weight=jnp.asarray(max_guidance_weight, dtype=jnp.float32),
        )
        denoise_started = time.monotonic()
        assert self._sample_actions_rtc is not None
        sampled_actions = self._sample_actions_rtc(sample_rng, observation, **sample_kwargs)
        sampled_actions = np.asarray(sampled_actions[0, ...])
        denoise_ms = (time.monotonic() - denoise_started) * 1000.0

        postprocess_started = time.monotonic()
        outputs = {
            "state": np.asarray(inputs["state"][0, ...]),
            "actions": sampled_actions,
        }
        outputs = self._output_transform(outputs)
        postprocess_ms = (time.monotonic() - postprocess_started) * 1000.0
        actions = np.asarray(outputs.get("actions"))
        if actions.shape != (10, 16) or not np.isfinite(actions).all():
            raise RtcPolicyError(f"postprocessed RTC actions must have finite shape (10, 16), got {actions.shape}")
        outputs["actions"] = actions
        outputs["policy_timing"] = {"infer_ms": denoise_ms}
        outputs["rtc_timing"] = {
            "preprocess_ms": preprocess_ms,
            "denoise_ms": denoise_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": (time.monotonic() - rtc_started) * 1000.0,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def infer_rtc(self, request: dict) -> dict:
        infer_rtc = getattr(self._policy, "infer_rtc", None)
        if infer_rtc is None:
            raise RtcPolicyError("recorded policy has no RTC inference method")
        results = infer_rtc(request)
        data = {"inputs": request, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
