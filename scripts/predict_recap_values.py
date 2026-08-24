"""Export frame-aligned pi0.5-RECAP value expectations to NPZ."""

import argparse
import json
import math
import pathlib

import flax.nnx as nnx
import jax
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import recap_value
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _total_frames(dataset_root: pathlib.Path) -> int:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    return int(info["total_frames"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-params", type=pathlib.Path, required=True)
    parser.add_argument("--base-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", default="stack_red_cones_recap")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    total_frames = _total_frames(args.dataset_root)
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=20,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    data_factory = _config.LeRobotMarvinProDataConfig(
        repo_id=args.repo_id,
        assets=_config.AssetsConfig(
            assets_dir=str(args.base_checkpoint / "assets"),
            asset_id="stack_red_cones",
        ),
        base_config=_config.DataConfig(
            repo_root=str(args.dataset_root),
            recap_include_frame_keys=True,
            prompt_from_task=True,
        ),
        use_delta_joint_actions=True,
    )
    predict_config = _config.TrainConfig(
        name="pi05_marvinpro_recap_value_predict",
        exp_name="predict",
        model=model_config,
        data=data_factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = _data_loader.create_data_loader(
        predict_config,
        shuffle=False,
        num_batches=math.ceil(total_frames / args.batch_size),
        drop_last=False,
    )

    model = recap_value.ReCAPValueModel(model_config, rngs=nnx.Rngs(jax.random.key(0)))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(_model.restore_params(args.value_params))
    model = nnx.merge(graphdef, state)
    model.eval()
    predict = nnx_utils.module_jit(model.predict_value)

    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for observation, _ in loader:
        if observation.episode_index is None or observation.frame_index is None:
            raise ValueError("prediction batch lost episode/frame alignment keys")
        episode_indices.append(np.asarray(observation.episode_index))
        frame_indices.append(np.asarray(observation.frame_index))
        model_observation = observation.replace(episode_index=None, frame_index=None)
        values.append(np.asarray(predict(model_observation)))

    episode_index = np.concatenate(episode_indices)
    frame_index = np.concatenate(frame_indices)
    value = np.concatenate(values)
    if len(value) != total_frames:
        raise ValueError(f"predicted {len(value)} frames, expected {total_frames}")
    keys = set(zip(episode_index.tolist(), frame_index.tolist(), strict=True))
    if len(keys) != total_frames:
        raise ValueError("value export contains duplicate episode/frame keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, episode_index=episode_index, frame_index=frame_index, value=value)
    print(f"saved {len(value)} frame-aligned predictions to {args.output}")


if __name__ == "__main__":
    main()
