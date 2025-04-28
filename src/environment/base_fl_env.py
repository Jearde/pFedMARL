import logging

import lightning as L
import torch
from tensordict import TensorDict
from torchrl.envs import ObservationNorm, RewardScaling

from federated_learning.client import Client
from federated_learning.device import Device
from federated_learning.server import Server

from .base_env import BaseFLEnv

# Set the logging level to ERROR to suppress lower-level logs
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
L.__version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseFederatedEnv(BaseFLEnv):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.action_shape["client"] = (1,)
        self.action_shape["server"] = (self.num_clients,)
        self.layer_keys = None

        self._init()

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Reset the server and clients.
        return super()._reset(tensordict, **kwargs)

    def _server_agg(self, device, action):
        device.aggregate(
            target=device.W,
            sources=[client.dW for client in self.clients],
            source_weights=action,
            is_gradient_update=True,
        )

    def _client_agg(self, device: Device, action, global_update):
        device.W = device.add_model_update(
            global_update, target=device.W, weight=action
        )
        device.W = device.add_model_update(
            device.dW, target=device.W, weight=(1 - action)
        )

    def _device_aggregate(self, device: Device, action: TensorDict):
        if isinstance(device, Server):
            device.sync_W_to_active()
            self.global_update_old = device.save_old_model()
            self._server_agg(device, action)
            self.global_model = device.sync_W_to_active()
            self.global_update = device.get_model_update()

            if (
                self.global_update["model.encoder.encoder.0.weight"].sum() == 0
                and self.current_step > 0
            ):
                raise ValueError("Global update is zero")

        elif isinstance(device, Client):
            action = action.to(device.model.device)
            # device.load_old_model()
            device.load_model(self.global_update_old)
            self._client_agg(device, action, self.global_update)
            device.sync_W_to_active()  # This might also cause problems

    def _get_reward(
        self,
        result_before_agg: dict,
        result_after_agg: dict,
        result_after_train: dict,
        failed: bool = False,
    ):
        reward = torch.tensor(-1 * result_after_agg[0]["val/loss"])
        reward = torch.tensor(
            result_before_agg[0]["val/loss"] - result_after_agg[0]["val/loss"]
        )

        return reward

    def _check_nan(self, device, group_name, dev_idx, result_after_agg):
        failed = False
        if torch.tensor(result_after_agg[0]["val/loss"]).isnan():
            logger.warning(f"Loss is NaN for {group_name} {dev_idx}")
            torch.nn.utils.get_total_norm(self.global_update.values())
            torch.nn.utils.get_total_norm(device.dW.values())
            # device.load_model(self.global_model)
            device.load_old_model()
            # result_after_agg = device.validate()
            print(
                f"!!! NaN loss for {group_name} {dev_idx}. Resetting model to old one."
            )
            result_after_agg = device.validate()
            failed = True
        return failed, result_after_agg

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
                    device.fit()  # This causes the gradient stacking error

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

    def _get_device_observation(self, device: Device, normalize=False) -> TensorDict:
        client_state = TensorDict()

        client_losses = torch.stack(
            [client.validation_loss_after_training for client in self.clients]
        ).mean()

        if isinstance(device, Client):
            # client_state["device_type"] = 0
            # Cosine similarity between client and server model updates
            # L2 norm between client and server model updates
            client_state["similarities"] = [
                server.get_similarity_and_norm(
                    source_dict=device.dW,
                    target_dict=server.dW,
                    layer_keys=self.layer_keys,
                )[0].mean()
                for server in self.servers
            ]
            client_state["norms"] = [
                server.get_similarity_and_norm(
                    source_dict=device.W,
                    target_dict=server.W,
                    layer_keys=self.layer_keys,
                )[1].mean()
                for server in self.servers
            ]
            client_state["loss_local"] = device.validate()[0]["val/loss"]
            client_state["loss_global"] = torch.tensor(
                [server.validate()[0]["val/loss"] for server in self.servers]
            ).mean()

        elif isinstance(device, Server):
            # client_state["device_type"] = 1
            # Validation loss of clients
            client_state["losses"] = [
                client.validation_loss_after_training for client in self.clients
            ]
            if normalize:
                # Softmax of the validation losses
                client_state["losses"] = torch.functional.F.softmax(
                    client_state["losses"], dim=0
                )
            # Cosine similarity between client and server model updates
            client_state["similarities"] = [
                device.get_similarity_and_norm(
                    source_dict=client.dW,
                    target_dict=device.dW,
                    layer_keys=self.layer_keys,
                )[0].mean()
                for client in self.clients
            ]
            # L2 norm of the client update.
            pass
            # L2 norm between client and server models
            client_state["norms"] = [
                device.get_similarity_and_norm(
                    source_dict=client.W,
                    target_dict=device.W,
                    layer_keys=self.layer_keys,
                )[1].mean()
                for client in self.clients
            ]

            client_state["batches"] = [
                len(client.data_module.train_dataloader()) for client in self.clients
            ]

        if normalize:
            client_losses = (
                client_losses if client_losses != 0 else result[0]["val/loss"]
            )
            client_state["loss"] = client_state["loss"] / client_losses

        # Flatten client state into single vector
        client_state_flat = torch.cat(
            [tensor.flatten() for tensor in client_state.values()]
        )

        if client_state_flat.isnan().any():
            logger.warning(f"Client state is NaN for {device.__class__.__name__}")

        if self.observation_mapper[device.__class__.__name__.lower()] is None:
            # Assign start and end indices for each observation key
            self.observation_mapper[device.__class__.__name__.lower()] = {}
            start_idx = 0
            for key, value in client_state.items():
                self.observation_mapper[device.__class__.__name__.lower()][key] = (
                    start_idx,
                    start_idx + value.numel(),
                )
                start_idx += value.numel()

        return client_state_flat

    @classmethod
    def make_env(cls, norm_obs_reward: bool = False, **kwargs):
        env = super().make_env(**kwargs)

        if norm_obs_reward:
            td = env.rollout(max_steps=2)

            loc = torch.stack(
                [
                    td.get((group, "observation")).mean()
                    for group in env.group_map.keys()
                ]
            ).mean()
            scale = torch.stack(
                [td.get((group, "observation")).std() for group in env.group_map.keys()]
            ).mean()

            observation_norm = ObservationNorm(
                loc=loc,
                scale=scale,
                in_keys=[(group, "observation") for group in env.group_map.keys()],
                standard_normal=True,
            )

            reward_scaling = RewardScaling(
                loc=-100.0,
                scale=10.0,
                standard_normal=True,
                in_keys=[(group, "reward") for group in env.group_map.keys()],
                out_keys=[(group, "reward") for group in env.group_map.keys()],
            )

            env.append_transform(reward_scaling)
            env.append_transform(observation_norm)

        # env = TransformedEnv(
        #     env,
        #     RewardSum(
        #         in_keys=env.reward_keys,
        #         reset_keys=["_reset"] * len(env.group_map.keys()),
        #     ),
        # )

        return env
