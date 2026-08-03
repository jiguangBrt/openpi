"""Marvin Pro policy transforms for the native 16-dimensional joint space."""

import dataclasses

import einops
import numpy as np

from openpi import transforms

STATE_DIM = 16


def make_marvinpro_example() -> dict:
    """Create a representative input for policy-server and transform checks."""
    return {
        "state": np.zeros((STATE_DIM,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "prompt": "perform the demonstrated task",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Marvin Pro images must have 3 dimensions, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Marvin Pro images must have 3 color channels, got {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class MarvinProInputs(transforms.DataTransformFn):
    """Map native Marvin Pro observations to the common pi model input."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] != STATE_DIM or not np.isfinite(state).all():
            raise ValueError(f"Marvin Pro state must be finite with last dimension {STATE_DIM}, got {state.shape}")

        images = data["images"]
        expected = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        if set(images) != expected:
            raise ValueError(f"Marvin Pro images must contain exactly {sorted(expected)}, got {sorted(images)}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(images["cam_high"]),
                "left_wrist_0_rgb": _parse_image(images["cam_left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["cam_right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != STATE_DIM or not np.isfinite(actions).all():
                raise ValueError(
                    f"Marvin Pro actions must be finite with last dimension {STATE_DIM}, got {actions.shape}"
                )
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MarvinProOutputs(transforms.DataTransformFn):
    """Drop model padding and return native Marvin Pro absolute targets."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < STATE_DIM:
            raise ValueError(f"Model actions need at least {STATE_DIM} dimensions, got {actions.shape}")
        return {"actions": actions[..., :STATE_DIM]}
