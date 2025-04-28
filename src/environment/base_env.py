import logging
from abc import abstractmethod
from pathlib import Path

import lightning as L
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded, UnboundedContinuous
from torchrl.envs import (
    EnvBase,  # base class for custom environments
    check_env_specs,
)
from torchrl.envs.transforms import InitTracker

from config.config_model import Config, parse_config
from federated_learning.client import Client
from federated_learning.data_creator import DataCreator
from federated_learning.device import Device
from federated_learning.server import Server

# Set the logging level to ERROR to suppress lower-level logs
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
L.__version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_composite_from_td(td):
    # recursively builds a CompositeSpec where each leaf is an Unbounded spec.
    if isinstance(td, torch.Tensor):
        return UnboundedContinuous(shape=td.shape, dtype=td.dtype)
    return Composite(
        {
            key: make_composite_from_td(val)
            if isinstance(val, TensorDictBase)
            else UnboundedContinuous(shape=val.shape, dtype=val.dtype)
            for key, val in td.items()
        },
        shape=td.batch_size,
    )


class BaseFLEnv(EnvBase):
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: Config | None = None,
        device: str = "cpu",
        batch_size: int = None,
        max_clients: int | None = None,
        max_servers: int = 1,
        max_global_steps: int = 0,
        fast: bool = True,
    ):
        super().__init__(device=device, batch_size=batch_size)
        if config is None and config_path is None:
            config = parse_config(config_path)
        elif config is None and config_path is None:
            raise ValueError("Either config or config_path must be provided.")

        self.max_global_steps = max_global_steps
        self.use_gradients = config.fl.use_gradients

        self.servers = [
            Server(
                config=config,
                parent_id=None,
                tags={
                    "type": "server",
                    "num_clients": config.fl.num_clients,
                    "dataset": "all",
                },
                fast=fast,
            )
            for _ in range(max_servers)
        ]

        ddp_settings = {
            "world_size": self.servers[0].trainer.world_size,
            "local_rank": self.servers[0].trainer.local_rank,
            "global_rank": self.servers[0].trainer.global_rank,
        }

        self.data_creator = DataCreator(
            self.servers[0].config, ddp_settings=ddp_settings
        )
        num_clients = self.data_creator.divide_dataset(
            num_subsets=config.fl.num_clients,
            key=config.fl.client_key,
            uneven_distribution=config.fl.uneven_distribution,
            multiple_datasets=config.fl.multiple_datasets,
            cluster_skew=config.fl.cluster_skew,
        )

        for server in self.servers:
            server.init_data(self.data_creator.get_overall_data_module())
            server.init_model(label_names=self.data_creator.label_names)
            server.data_module.setup("fit")

        self.clients = [
            Client(
                client_id=i,
                data_module=self.data_creator.get_subset(i),
                config=config,
                label_names=self.data_creator.label_names,
                epoch=config.trainer.max_epochs,
                extra_callbacks=[],
                tags={
                    "type": "client",
                    "client_id": i,
                },
                fast=fast,
            )
            for i in range(num_clients)
        ][:max_clients]

        # Initialize the server and clients data loaders
        for client in self.clients:
            client.data_module.setup("fit")

        self.max_steps = config.fl.max_rounds
        self.current_step = 0

        # Optionally, you can define action and state specs here.
        # For example, if each client gets an impact factor, then:
        self.num_servers = len(self.servers)
        self.num_clients = len(self.clients)

        self.devices = {
            "server": self.servers,
            "client": self.clients,
        }

        self.action_shape = {}

    def _init(self):
        self._init_group_map()
        self.state = self._get_state()
        self._init_specs()
        self._reset()

    def _init_group_map(self, server_only: bool = False):
        if server_only:
            self.group_map = {
                "server": [f"server_{i}" for i in range(self.num_servers)],
                # "client": [f"client_{i}" for i in range(self.num_clients)],
            }
        else:
            self.group_map = {
                key: [f"{key}_{i}" for i in range(len(self.devices[key]))]
                for key in self.devices.keys()
            }

        self.observation_mapper = {key: None for key in self.group_map.keys()}

    def _init_specs(self):
        # Define action specs
        self.action_spec = Composite(
            {
                group_name: Composite(
                    {
                        "action": Bounded(
                            low=torch.zeros(
                                len(self.group_map[group_name]),
                                *self.action_shape[group_name],
                            ),
                            high=torch.ones(
                                len(self.group_map[group_name]),
                                *self.action_shape[group_name],
                            ),
                        )
                    },
                    shape=(len(self.group_map[group_name]),),
                )
                for group_name in self.group_map.keys()
            }
        )

        # Define observation specs
        self.observation_spec = Composite(
            {
                group_name: Composite(
                    {
                        "observation": make_composite_from_td(
                            self.state[group_name]["observation"]
                        )
                    },
                    shape=(len(self.group_map[group_name]),),
                )
                for group_name in self.group_map.keys()
            }
        )

        # For simplicity, set state_spec equal to observation_spec.
        self.state_spec = self.observation_spec.clone()

        # Define reward specs (each agent gets a scalar reward)
        reward_shape = (1,)
        self.reward_spec = Composite(
            {
                group_name: Composite(
                    {
                        "reward": Unbounded(
                            shape=torch.Size(
                                [len(self.group_map[group_name]), *reward_shape]
                            )
                        )
                    },
                    shape=(len(self.group_map[group_name]),),
                )
                for group_name in self.group_map.keys()
            }
        )

        # Define a shared done flag (root level)
        self.done_spec = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size([1]),
                    dtype=torch.bool,
                ),
                "terminated": Categorical(
                    n=2,
                    shape=torch.Size([1]),
                    dtype=torch.bool,
                ),
            },
            # shape=(1,),
        )

    def get_done_td(self, done) -> TensorDict:
        done = torch.tensor(done, dtype=torch.bool)
        done_td = TensorDict(
            {key: done for key in self.done_keys},
            batch_size=torch.Size([]),
        )
        return done_td

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        # Reset the server and clients.
        self.servers[0].reset_parameters()
        global_model = self.servers[0].get_model()
        for server in self.servers:
            server.load_model(global_model, reset=True)
        for client in self.clients:
            client.load_model(global_model, reset=True)
        self.current_step = 0

        # Compute and return the initial state.
        self.state = self._get_state()

        td = self.state

        td.update(self.get_done_td(False))
        return td

    def _get_state(self) -> TensorDict:
        """Compute the state vector from the client models."""

        self.layer_keys = [
            k for k, p in self.servers[0].model.named_parameters() if p.requires_grad
        ]
        self.num_layers = len(self.layer_keys)

        state = TensorDict()

        for group_name, group in self.group_map.items():
            group = self.devices[group_name]
            device_states = []
            for idx, device in enumerate(group):
                device_states.append(self._get_device_observation(device))

            if isinstance(device_states[0], TensorDict):
                batched_device_states = TensorDict(
                    {
                        key: torch.stack(
                            [device_state[key] for device_state in device_states]
                        )
                        for key in device_states[0].keys()
                    },
                    batch_size=(len(device_states),),
                )
            else:
                batched_device_states = torch.stack(device_states)

            state[group_name] = TensorDict(
                {
                    "observation": batched_device_states,
                },
                batch_size=len(self.group_map[group_name]),
            )

        return state

    def test(
        self, full_test: bool = False, verbose: bool = True, prefix=""
    ) -> tuple[dict, dict]:
        results_anomaly = []
        results_test = {device_type: [] for device_type in self.devices.keys()}

        for device_type in self.devices.keys():
            for device in self.devices[device_type]:
                results_test[device_type].append(
                    device.trainer.test(
                        model=device.model,
                        datamodule=device.data_module,
                        verbose=verbose,
                    )
                )
                if device_type == "client" and full_test:
                    results_anomaly.append(
                        device.test_model(
                            model=device.model,
                            datamodule=device.data_module,
                        )
                    )

        results: dict[str, list] = {}
        # Add to results with the group-name_key-name
        for group, group_results in results_test.items():
            for device_result in group_results:
                for device_result in device_result:
                    for key, value in device_result.items():
                        key = f"{prefix}{group}_{key.split('/')[-1]}"
                        if key not in results.keys():
                            results[key] = []
                        results[key].append(value)

        # Compute mean and std for each key
        keys = list(results.keys())
        for key in keys:
            values = torch.tensor(results[key])
            if len(values) == 1:
                results[f"{key}_mean"] = values[0]
            else:
                results[f"{key}_mean"] = values.mean(dim=0)
                results[f"{key}_std"] = values.std(dim=0)
            del results[key]

        results_fine: dict[str, list] = {}
        for srv_idx, server in enumerate(self.servers):
            for clt_idx, client in enumerate(self.clients):
                # Server on client datasets
                server_result = server.trainer.test(
                    model=server.model,
                    datamodule=client.data_module,
                    verbose=verbose,
                )[0]
                for key, value in server_result.items():
                    key = f"{prefix}server{srv_idx}_on_client{clt_idx}_{key.split('/')[-1]}"
                    results_fine[key] = value

                # Clients on server datasets
                client_result = client.trainer.test(
                    model=client.model,
                    datamodule=server.data_module,
                    verbose=verbose,
                )[0]
                for key, value in client_result.items():
                    key = f"{prefix}client{clt_idx}_on_server{srv_idx}_{key.split('/')[-1]}"
                    results_fine[key] = value

                # Clients on server datasets
                own_result = client.trainer.test(
                    model=client.model,
                    datamodule=client.data_module,
                    verbose=verbose,
                )[0]
                for key, value in own_result.items():
                    key = f"{prefix}client{clt_idx}_{key.split('/')[-1]}"
                    results_fine[key] = value

                own_result = server.trainer.test(
                    model=server.model,
                    datamodule=server.data_module,
                    verbose=verbose,
                )[0]
                for key, value in own_result.items():
                    key = f"{prefix}server{srv_idx}_{key.split('/')[-1]}"
                    results_fine[key] = value

        return results, results_fine

    def get_observation_dict(self, group_data, group, prefix=""):
        # TODO: Better to return original, because this can be normalized
        return_dict = {}

        if group in group_data and "action" in group_data[group]:
            # get actions
            group_data[group]["action"]
            obs_key = "action"
            obs_dict = {
                f"{prefix}{obs_key}_{group}_{idx}": group_data[group]["action"][
                    :, idx
                ].mean()
                for idx in range(group_data[group]["action"].shape[1])
            }
            return_dict.update(obs_dict)

        group_data = group_data["next"] if "next" in group_data else group_data
        group_data = group_data[group] if group in group_data else group_data

        if not hasattr(self, "observation_mapper"):
            return return_dict

        for obs_key in self.observation_mapper[group].keys():
            loss_idcs = self.observation_mapper[group][obs_key]
            obs_key = "loss" if obs_key == "losses" else obs_key
            if group_data["observation"].dim() == 2:
                obs_dict = {
                    f"{prefix}{obs_key}_{group}_{idx}": val.mean()
                    for idx, val in enumerate(
                        group_data["observation"][:, loss_idcs[0] : loss_idcs[1]]
                    )
                }
            else:
                obs_dict = {
                    f"{prefix}{obs_key}_{group}_{idx}": val.mean()
                    for idx, val in enumerate(
                        group_data["observation"][
                            :, :, loss_idcs[0] : loss_idcs[1]
                        ].mean(axis=0)
                    )
                }

            return_dict.update(obs_dict)

        return return_dict

    def plot_class_distribution(self):
        return self.data_creator.plot_class_distribution()

    @abstractmethod
    def _get_device_observation(self, device: Device, normalize=False) -> TensorDict:
        raise NotImplementedError

    def _set_seed(self, seed: int):
        # Set seed for reproducibility.
        for server in self.servers:
            server.init_seeds(seed)
        for client in self.clients:
            client.init_seeds(seed)

    def check_specs(self):
        print("----------------- action_spec -----------------")
        print(self.full_action_spec)
        print("----------------- reward_spec -----------------")
        print(self.full_reward_spec)
        print("----------------- done_spec -----------------")
        print(self.full_done_spec)
        print("----------------- observation_spec -----------------")
        print(self.observation_spec)
        print("----------------- state_spec -----------------")
        print(self.full_state_spec)
        print("----------------- group_map -----------------")
        print(self.group_map)
        print("----------------- rollout_data -----------------")
        n_rollout_steps = 2
        rollout_data = self.rollout(max_steps=n_rollout_steps)
        print(
            f"Shape of the {n_rollout_steps} rollout steps TensorDict:",
            rollout_data.batch_size,
        )
        print(rollout_data)
        print("----------------- Env Check -----------------")
        check_env_specs(self)
        print("----------------- Keys -----------------")
        print("action_keys:", self.action_keys)
        print("reward_keys:", self.reward_keys)
        print("done_keys:", self.done_keys)

    @classmethod
    def make_env(cls, enable_checks: bool = False, **kwargs):
        env = cls(**kwargs)

        # Apply a transform that initializes the tracker for the environment
        init_tracker = InitTracker()
        env.append_transform(init_tracker)

        if enable_checks:
            env.check_env_specs()

        return env
