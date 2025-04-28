import logging

import lightning as L
import torch
from tensordict import TensorDict

from federated_learning.client import Client
from federated_learning.device import Device
from federated_learning.server import Server

from .base_fl_env import BaseFederatedEnv

# Set the logging level to ERROR to suppress lower-level logs
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
L.__version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GlobalLocalFederatedEnv(BaseFederatedEnv):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Reset the server and clients.
        return super()._reset(tensordict, **kwargs)

    def _server_agg(self, device: Server, action):
        if self.use_gradients:
            device.aggregate(
                target=device.W,
                sources=[client.dW for client in self.clients],
                source_weights=action,
                is_gradient_update=True,
                remove_keys=[
                    "num_batches_tracked",
                    # "running_mean",
                    # "running_var",
                    "feature_extractor",
                ],
            )
        else:
            device.aggregate(
                target=device.W,
                sources=[client.W for client in self.clients],
                source_weights=action,
                is_gradient_update=False,
                remove_keys=[
                    "num_batches_tracked",
                    # "running_mean",
                    # "running_var",
                    "feature_extractor",
                ],
            )

    def _client_agg(self, device: Device, action, global_update):
        # device.W = device.add_model_update(
        #     global_update, target=device.W, weight=action
        # )
        # device.W = device.add_model_update(
        #     device.dW, target=device.W, weight=(1 - action)
        # )

        action = 0.5 * (action + 1.0)

        if self.use_gradients:
            device.aggregate(
                target=device.W,
                sources=[self.global_update, device.dW],
                source_weights=torch.cat([action, (1 - action)]),
                is_gradient_update=True,
                remove_keys=[
                    "num_batches_tracked",
                    "running_mean",
                    "running_var",
                    "feature_extractor",
                ],
            )
        else:
            device.aggregate(
                target=device.W,
                sources=[self.global_model, device.W],
                source_weights=torch.cat([action, (1 - action)]),
                is_gradient_update=False,
                remove_keys=[
                    "num_batches_tracked",
                    "running_mean",
                    "running_var",
                    "feature_extractor",
                ],
            )

    def _device_aggregate(self, device: Device, action: TensorDict):
        if isinstance(device, Server):
            device.sync_W_to_active()
            self.global_model_old = device.save_old_model()
            self._server_agg(device, action)
            self.global_model = device.sync_W_to_active()
            self.global_update = device.get_model_update()

        elif isinstance(device, Client):
            action = action.to(device.model.device)
            if self.use_gradients:
                device.load_old_model()
                # device.load_model(self.global_model_old)
            self._client_agg(device, action, self.global_update)
            device.sync_W_to_active()  # This might also cause problems

    def _check_nan(self, device, group_name, dev_idx, result_after_agg):
        failed = False
        if torch.tensor(result_after_agg[0]["val/loss"]).isnan():
            logger.warning(f"Loss is NaN for {group_name} {dev_idx}")

            print(
                f"!!! NaN loss for {group_name} {dev_idx}. Resetting model to old one."
            )

            # device.load_model(self.global_model)

            device.load_old_model()
            action = torch.tensor(1.0)
            device.W = device.add_model_update(
                device.dW, target=device.W, weight=action
            )
            device.sync_W_to_active()

            result_after_agg = device.validate()
            failed = True

        return failed, result_after_agg

    def _get_reward(
        self,
        result_before_agg: dict,
        result_after_agg: dict,
        result_after_train: dict,
        failed: bool = False,
    ):
        reward = torch.tensor(-1 * result_after_agg[0]["val/loss"])
        reward = torch.tensor(
            (result_before_agg[0]["val/loss"] - result_after_agg[0]["val/loss"])
            / (result_before_agg[0]["val/loss"] + 1e-5)
        ).clamp(min=0, max=None)

        if failed:
            reward = torch.tensor(-1.0)

        reward = reward.clamp(min=-1.0, max=1.0)

        return reward
