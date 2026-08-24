import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _IndexedDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"episode_index": np.asarray(4), "frame_index": np.asarray(index.__index__())}


def test_recap_labels_dataset_aligns_episode_and_frame(tmp_path):
    path = tmp_path / "recap.npz"
    np.savez_compressed(
        path,
        episode_index=np.asarray([4, 4]),
        frame_index=np.asarray([0, 1]),
        advantage_indicator=np.asarray([False, True]),
    )
    dataset = _data_loader.ReCAPLabelsDataset(_IndexedDataset(), str(path))
    assert not dataset[0]["advantage_indicator"]
    assert dataset[1]["advantage_indicator"]


def test_recap_labels_dataset_rejects_missing_frame(tmp_path):
    path = tmp_path / "recap.npz"
    np.savez_compressed(
        path,
        episode_index=np.asarray([4]),
        frame_index=np.asarray([0]),
        advantage_indicator=np.asarray([False]),
    )
    dataset = _data_loader.ReCAPLabelsDataset(_IndexedDataset(), str(path))
    with pytest.raises(KeyError, match="no label"):
        dataset[1]


def test_recap_value_targets_dataset_aligns_episode_and_frame(tmp_path):
    path = tmp_path / "value_targets.npz"
    np.savez_compressed(
        path,
        episode_index=np.asarray([4, 4]),
        frame_index=np.asarray([0, 1]),
        return_bin=np.asarray([0, 200]),
    )
    dataset = _data_loader.ReCAPValueTargetsDataset(_IndexedDataset(), str(path))
    assert dataset[0]["value_target_bin"] == 0
    assert dataset[1]["value_target_bin"] == 200
