import logging

import torch
from tensordict import TensorDict

from federated_learning.client import Client
from federated_learning.device import Device
from federated_learning.server import Server

from .base_fl_env import BaseFederatedEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaselineEnv(BaseFederatedEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _server_agg(self, device: Server, action: torch.Tensor, gradient: bool = False):
        pass

    def _client_agg(
        self,
        device: Client,
        action: torch.Tensor,
        global_update: dict[str, torch.Tensor],
    ):
        device.W = device.load_model(global_update)

    def _device_aggregate(self, device: Device, action: TensorDict):
        if isinstance(device, Server):
            self.global_model = device.get_model(
                remove_keys=[
                    "num_batches_tracked",
                    # "running_mean",
                    # "running_var",
                    "feature_extractor",
                ],
            )
            self.global_update = device.get_model_update()

        elif isinstance(device, Client):
            action = action.to(device.model.device)
            device.save_old_model()
            device.load_model(self.global_model)
            device.sync_W_to_active()

    def _step(self, tensordict: TensorDict) -> TensorDict:
        step_td = TensorDict()

        self.layer_keys = (
            [k for k, p in self.servers[0].model.named_parameters() if p.requires_grad]
            if self.layer_keys is None
            else self.layer_keys
        )

        for group_name, group in self.devices.items():
            actions = (
                tensordict[group_name]["action"]
                if group_name in self.group_map
                else torch.ones(len(group))
            )
            rewards = []
            failed = False

            # Collect state before (model, loss)
            for dev_idx, device in enumerate(group):
                action = actions[dev_idx]

                # check if any action is nan
                if action.isnan().any():
                    logger.warning(f"Action is NaN for {group_name} {dev_idx}")
                    print(tensordict[group_name]["observation"])

                result_before_agg = device.validate()
                # Aggregate the models
                self._device_aggregate(device, action)

                result_after_agg = device.validate()

                failed, result_after_agg = self._check_nan(
                    device, group_name, dev_idx, result_after_agg
                )

                # Train the models
                if hasattr(device, "fit"):
                    pass
                else:
                    self.clients[0].load_model(self.global_model)
                    self.clients[0].save_old_model()
                    self.clients[0].trainer.fit_loop.epoch_progress.reset()
                    self.clients[0].model.fl_round += 1
                    self.clients[0].trainer.fit(
                        self.clients[0].model,
                        datamodule=device.data_module,
                    )
                    self.clients[0].sync_active_to_W()
                    device.load_model(
                        self.clients[0].get_model(
                            remove_keys=[
                                "num_batches_tracked",
                                # "running_mean",
                                # "running_var",
                                "feature_extractor",
                            ]
                        )
                    )

                result_after_train = device.validate()

                reward = self._get_reward(
                    result_before_agg,
                    result_after_agg,
                    result_after_train,
                    failed=failed,
                )

                if reward.isnan():
                    print(
                        f"!!! NaN reward for {group_name} {dev_idx}. Resetting model to old one."
                    )

                rewards.append(reward)

            if group_name in self.group_map:
                step_td[group_name] = TensorDict(
                    {"reward": torch.tensor(rewards)},
                    batch_size=len(self.group_map[group_name]),
                )

            logger.info(
                f"Step {self.current_step} - {group_name} rewards: {['%.2f' % r for r in rewards]}"
            )

        # Get new state
        self.state = self._get_state()

        step_td.update(self.state)

        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        if done:
            logger.info(f"Step {self.current_step} - Done")

        step_td.update(self.get_done_td(done))

        return step_td
