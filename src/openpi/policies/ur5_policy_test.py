import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import ur5_policy


def test_endpose_policy_preserves_absolute_actions():
    data = ur5_policy.make_ur5_endpose_example()
    actions = np.array(
        [
            [0.45, -0.15, 0.30, -0.16, -3.13, 0.27, 0.0],
            [0.46, -0.14, 0.31, -0.17, -3.12, 0.28, 1.0],
        ],
        dtype=np.float32,
    )
    data["actions"] = actions.copy()

    inputs = ur5_policy.UR5EndPoseInputs(model_type=_model.ModelType.PI05)(data)
    outputs = ur5_policy.UR5EndPoseOutputs()({"actions": np.pad(inputs["actions"], ((0, 0), (0, 25)))})

    assert np.array_equal(inputs["state"], data["state"])
    assert np.array_equal(inputs["actions"], actions)
    assert np.array_equal(outputs["actions"], actions)
    assert inputs["image_mask"]["base_0_rgb"]
    assert inputs["image_mask"]["left_wrist_0_rgb"]
    assert not inputs["image_mask"]["right_wrist_0_rgb"]


def test_endpose_policy_rejects_non_pose_state():
    data = ur5_policy.make_ur5_endpose_example()
    data["state"] = np.zeros(6, dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        ur5_policy.UR5EndPoseInputs(model_type=_model.ModelType.PI05)(data)
