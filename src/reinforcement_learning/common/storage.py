import tempfile
from functools import partial

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    RandomSampler,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    RandomSampler,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import EnvBase, ExplorationType


class ReplayBufferDataset(torch.utils.data.Dataset):
    def __init__(self, replay_buffer, batch_size, n_batches=10):
        """
        Args:
            replay_buffer: Your replay buffer instance.
            batch_size: Number of samples per batch.
            n_batches: Maximum number of batches to sample per epoch.
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __len__(self):
        # Return the number of batches to sample.
        # Use n_batches to control the number of optimization steps per epoch.
        length = min(len(self.replay_buffer), self.n_batches * self.batch_size)
        return int(np.ceil(length / self.batch_size))

    def __getitem__(self, idx):
        return self.replay_buffer.sample(self.batch_size)


def process_batch(batch: TensorDictBase, group_keys: list[str]) -> TensorDictBase:
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    for group in group_keys:
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch


def append_replay_buffer(
    batch: TensorDictBase,
    group_keys: list[str],
    replay_buffers: dict[str, TensorDictReplayBuffer],
    actor_explores: dict[str, torch.nn.ModuleDict] | None = None,
):
    current_frames = batch.numel()
    batch = process_batch(batch, group_keys)
    for group in group_keys:
        group_batch = batch.exclude(
            *[
                key
                for _group in group_keys
                if _group != group
                for key in [_group, ("next", _group)]
            ]
        )  # Exclude data from other groups
        group_batch = group_batch.reshape(
            -1
        )  # This just affects the leading dimensions in batch_size of the tensordict
        replay_buffers[group].extend(group_batch)

    if actor_explores is not None and hasattr(actor_explores[group][-1], "step"):
        actor_explores[group][-1].step(current_frames)


def make_replay_buffers(
    env,
    device: torch.device | str | None = "cpu",
    train_batch_size: int = 64,
    memory_size: int = 100000,
    prb: bool = False,
    alpha: float = 0.7,
    beta: float = 0.5,
    prefetch: int = 0,
) -> dict[str, TensorDictReplayBuffer]:
    """
    Create a replay buffer using a lazy memmap storage.
    """
    replay_buffers = {}
    for group, _agents in env.group_map.items():
        if prb:
            buffer_class = partial(
                TensorDictPrioritizedReplayBuffer, alpha=alpha, beta=beta
            )
        else:
            buffer_class = partial(TensorDictReplayBuffer, sampler=RandomSampler())

        tmpdir = tempfile.TemporaryDirectory()
        buffer_scratch_dir = tmpdir.name

        buffer_storage = LazyMemmapStorage(
            max_size=memory_size, scratch_dir=buffer_scratch_dir, device="cpu"
        )

        # Create a replay buffer using a lazy memmap storage.
        replay_buffer = buffer_class(
            storage=buffer_storage,
            batch_size=train_batch_size,  # sample mini-batches of 64 transitions
            prefetch=prefetch,
        )
        # Cast stored data to the desired device when appending.
        replay_buffer.append_transform(lambda x: x.to(device))

        replay_buffers[group] = replay_buffer

    return replay_buffer


def make_collector(
    env: EnvBase,
    exploration_policies,
    total_frames=1_000_000,
    frames_per_batch=1,  # 2 causes gradient stacking bug
    max_frames_per_traj=None,
    init_random_frames=0,
    device: torch.device | str | None = "cpu",
    storing_device: torch.device | str | None = "cpu",
    env_device: torch.device | str | None = "cpu",
    policy_device: torch.device | str | None = "cpu",
    exploration_type=ExplorationType.RANDOM,
):
    agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

    collector = SyncDataCollector(
        env,
        policy=agents_exploration_policy,
        device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_random_frames,
        max_frames_per_traj=max_frames_per_traj,
        reset_at_each_iter=False,
        split_trajs=False,
        exploration_type=exploration_type,
        no_cuda_sync=False,
        storing_device=storing_device,
        env_device=env_device,
        policy_device=policy_device,
    )

    return collector
