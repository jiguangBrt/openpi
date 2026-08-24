"""Train the independent 201-bin pi0.5-RECAP value model.

This entry point intentionally starts every value iteration from the same H20
Iteration-0 checkpoint. It reloads the non-LoRA backbone while initializing a
fresh main-VLM LoRA and value head.
"""

import argparse
import dataclasses
import functools
import json
import math
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from tqdm_loggable.auto import tqdm

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import recap_value
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import recap
from openpi.training import weight_loaders


def _load_total_frames(dataset_root: pathlib.Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return int(info["total_frames"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read total_frames from {info_path}: {exc}") from exc


def _load_model(config: pi0_config.Pi0Config, checkpoint: pathlib.Path, rng: jax.Array) -> recap_value.ReCAPValueModel:
    model = recap_value.ReCAPValueModel(config, rngs=nnx.Rngs(rng))
    graphdef, state = nnx.split(model)
    loaded = weight_loaders.ReCAPValueWeightLoader(str(checkpoint / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def _make_train_config(
    args: argparse.Namespace,
    model_config: pi0_config.Pi0Config,
    checkpoint: pathlib.Path,
    episode_indices: tuple[int, ...],
    *,
    name: str,
) -> _config.TrainConfig:
    data_factory = _config.LeRobotMarvinProDataConfig(
        repo_id=args.repo_id,
        assets=_config.AssetsConfig(
            assets_dir=str(checkpoint / "assets"),
            asset_id="stack_red_cones",
        ),
        base_config=_config.DataConfig(
            repo_root=str(args.dataset_root),
            episodes=episode_indices,
            recap_value_targets_path=str(args.targets),
            recap_include_frame_keys=True,
            prompt_from_task=True,
        ),
        use_delta_joint_actions=True,
    )
    return _config.TrainConfig(
        name=name,
        exp_name="value",
        model=model_config,
        data=data_factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def _train_step(
    model_def: nnx.GraphDef,
    params: nnx.State,
    opt_state: optax.OptState,
    rng: jax.Array,
    observation: _model.Observation,
    *,
    tx: optax.GradientTransformation,
    trainable_filter: nnx.filterlib.Filter,
) -> tuple[nnx.State, optax.OptState, jax.Array]:
    model = nnx.merge(model_def, params)
    model.train()
    if observation.value_target_bin is None:
        raise ValueError("value training batch has no value_target_bin")
    targets = observation.value_target_bin
    model_observation = dataclasses.replace(observation, value_target_bin=None)

    def loss_fn(value_model: recap_value.ReCAPValueModel) -> jax.Array:
        return jnp.mean(value_model.compute_value_loss(rng, model_observation, targets, train=True))

    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
    trainable_params = params.filter(trainable_filter)
    updates, opt_state = tx.update(grads, opt_state, trainable_params)
    nnx.update(model, optax.apply_updates(trainable_params, updates))
    return nnx.state(model), opt_state, loss


def _predict_validation(
    model_def: nnx.GraphDef,
    params: nnx.State,
    loader: _data_loader.DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = nnx.merge(model_def, params)
    model.eval()
    predict = nnx_utils.module_jit(model.predict_value)
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for observation, _ in loader:
        if observation.episode_index is None or observation.frame_index is None:
            raise ValueError("validation batch lost episode/frame alignment keys")
        episode_indices.append(np.asarray(observation.episode_index))
        frame_indices.append(np.asarray(observation.frame_index))
        model_observation = observation.replace(
            value_target_bin=None,
            episode_index=None,
            frame_index=None,
        )
        values.append(np.asarray(predict(model_observation)))
    return (
        np.concatenate(episode_indices),
        np.concatenate(frame_indices),
        np.concatenate(values),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--outcomes", type=pathlib.Path, required=True)
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", default="stack_red_cones_recap")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--peak-lr", type=float, default=2.5e-5)
    parser.add_argument("--final-lr", type=float, default=2.5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if not (args.base_checkpoint / "params").is_dir():
        raise FileNotFoundError(f"missing H20 Iteration-0 params: {args.base_checkpoint / 'params'}")
    if not args.targets.is_file():
        raise FileNotFoundError(f"missing return-bin sidecar: {args.targets}")

    outcomes = recap.load_episode_outcomes(args.outcomes)
    if any(outcome.source is recap.EpisodeSource.EVALUATION for outcome in outcomes):
        raise ValueError("value training outcomes must not contain held-out policy evaluation episodes")
    total_frames = _load_total_frames(args.dataset_root)
    manifest_frames = sum(outcome.num_frames for outcome in outcomes)
    if total_frames != manifest_frames:
        raise ValueError(f"dataset has {total_frames} frames but outcome manifest describes {manifest_frames}")
    train_outcomes, validation_outcomes = recap.split_episode_outcomes(outcomes, seed=args.seed)
    if not train_outcomes or not validation_outcomes:
        raise ValueError("value training requires non-empty episode-level train and validation splits")
    train_frames = sum(outcome.num_frames for outcome in train_outcomes)
    validation_frames = sum(outcome.num_frames for outcome in validation_outcomes)
    steps = max(1, math.ceil(train_frames * args.epochs / args.batch_size))
    warmup_steps = max(1, round(steps * 0.1))
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=20,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    train_config = _make_train_config(
        args,
        model_config,
        args.base_checkpoint,
        tuple(outcome.episode_index for outcome in train_outcomes),
        name="pi05_marvinpro_recap_value_train",
    )
    loader = _data_loader.create_data_loader(train_config, shuffle=True, num_batches=steps)
    validation_config = _make_train_config(
        args,
        model_config,
        args.base_checkpoint,
        tuple(outcome.episode_index for outcome in validation_outcomes),
        name="pi05_marvinpro_recap_value_validation",
    )
    validation_loader = _data_loader.create_data_loader(
        validation_config,
        shuffle=False,
        num_batches=math.ceil(validation_frames / args.batch_size),
        drop_last=False,
    )

    rng = jax.random.key(args.seed)
    rng, model_rng = jax.random.split(rng)
    model = _load_model(model_config, args.base_checkpoint, model_rng)
    model_def, params = nnx.split(model)
    trainable_filter = nnx.All(nnx.Param, nnx.Not(recap_value.value_freeze_filter()))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=args.final_lr,
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = tx.init(params.filter(trainable_filter))
    step_fn = jax.jit(
        functools.partial(
            _train_step,
            model_def,
            tx=tx,
            trainable_filter=trainable_filter,
        )
    )

    for observation, _ in tqdm(loader, total=steps, desc="RECAP value"):
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss = step_fn(params, opt_state, step_rng, observation)
    jax.block_until_ready(loss)

    episode_index, frame_index, predicted_value = _predict_validation(model_def, params, validation_loader)
    metrics = recap.evaluate_value_predictions(
        validation_outcomes,
        episode_index,
        frame_index,
        predicted_value,
    )
    metrics.update(
        {
            "train_episodes": len(train_outcomes),
            "validation_episodes": len(validation_outcomes),
            "train_frames": train_frames,
            "validation_frames": validation_frames,
            "steps": steps,
            "final_train_loss": float(loss),
            "train_episode_ids": [outcome.episode_id for outcome in train_outcomes],
            "validation_episode_ids": [outcome.episode_id for outcome in validation_outcomes],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(args.output, {"params": params.to_pure_dict()})
    metrics_path = args.output.parent / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved value model to {args.output}; metrics={metrics_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
