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


class FedSGDEnv(BaseFederatedEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_action_key = "batches"  # losses batches

    def _server_agg(self, device: Server, action: torch.Tensor, gradient: bool = False):
        # L-weighted average of the gradients from the clients
        action_range = self.observation_mapper["server"][self.observation_action_key]
        action = self.state["server"]["observation"][0][
            action_range[0] : action_range[1]
        ]

        action = torch.functional.F.softmax(action, dim=0)

        if gradient:
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

    def _client_agg(
        self,
        device: Client,
        action: torch.Tensor,
        global_update: dict[str, torch.Tensor],
    ):
        action = (
            torch.tensor(1.0)
            if self.current_step < self.max_global_steps
            else torch.tensor(0.0)
        )
        device.W = device.add_model_update(
            global_update,
            target=device.W,
            weight=action,
            remove_keys=[
                "num_batches_tracked",
                # "running_mean",
                # "running_var",
                "feature_extractor",
            ],
        )
        device.W = device.add_model_update(
            device.dW,
            target=device.W,
            weight=(1 - action),
            remove_keys=[
                "num_batches_tracked",
                # "running_mean",
                # "running_var",
                "feature_extractor",
            ],
        )

    def _device_aggregate(self, device: Device, action: TensorDict):
        if isinstance(device, Server):
            device.sync_W_to_active()
            self.global_update_old = device.save_old_model()
            self._server_agg(device, action)
            self.global_model = device.sync_W_to_active()
            self.global_update = device.get_model_update()

        elif isinstance(device, Client):
            action = action.to(device.model.device)
            device.load_old_model()
            # device.load_model(self.global_update_old)
            self._client_agg(device, action, self.global_update)
            device.sync_W_to_active()  # This might also cause problems
