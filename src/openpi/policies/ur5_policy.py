import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """随机输入样例,测试 policy server / 走查 transform 用。"""
    return {
        "state": np.random.rand(7),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the yellow ping-pong ball and place it in the white box.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """Convert UR observations into the common pi0 model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """Return the six arm joints and gripper action, dropping model padding."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}


def make_ur5_endpose_example() -> dict:
    """Create a representative observation for an absolute ``base -> tool0`` end-pose policy."""
    return {
        "state": np.array([0.45, -0.15, 0.30, -0.16, -3.13, 0.27, 0.0], dtype=np.float32),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the yellow ping-pong ball and place it in the white box.",
    }


@dataclasses.dataclass(frozen=True)
class UR5EndPoseInputs(UR5Inputs):
    """Accept ``[x, y, z, rx, ry, rz, gripper]`` observations in the UR base frame."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (7,):
            raise ValueError(f"UR5 end-pose state must have shape (7,), got {state.shape}")
        if not np.issubdtype(state.dtype, np.floating) or not np.isfinite(state).all():
            raise ValueError("UR5 end-pose state must contain finite floating-point values")
        return super().__call__(data)


@dataclasses.dataclass(frozen=True)
class UR5EndPoseOutputs(transforms.DataTransformFn):
    """Return absolute ``base -> tool0`` pose plus the dataset-native gripper target."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < 7:
            raise ValueError(f"UR5 end-pose actions must have at least 7 dimensions, got {actions.shape}")
        return {"actions": actions[..., :7]}
