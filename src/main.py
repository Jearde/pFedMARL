# %% Include
import logging
from pathlib import Path

from torchrl.envs import VmasEnv
from torchrl.envs.transforms import InitTracker, RewardSum

from config.config_model import parse_config, write_example_config
from environment.baseline import BaselineEnv
from environment.fed_sgd import FedSGDEnv
from environment.global_personal import GlobalLocalFederatedEnv
from reinforcement_learning.td3.td3 import TD3Module
from utils.lightning_utils import init_callbacks_plugins, init_seeds, init_trainer
from utils.logger import init_logger

# from typing import Any, Dict
# import hydra
# from hydra import compose, initialize
# from hydra.core.hydra_config import HydraConfig
# from omegaconf import DictConfig, OmegaConf
# from rich.pretty import pprint

logger = logging.getLogger(__name__)

# %%
config_path = Path(".configs") / "example_config.yaml"
write_example_config(config_path)
config = parse_config(config_path)

# %%
type_to_run = config.rl.environment_name
init_seeds(config.seed)

if type_to_run == "maddpg_global_local":
    env = GlobalLocalFederatedEnv.make_env(
        config_path=config_path,
        config=config,
        **config.rl.environment_config.model_dump(),
    )
elif type_to_run == "maddpg_fed_sgd":
    env = FedSGDEnv.make_env(
        config=config,
        **config.rl.environment_config.model_dump(),
    )

elif type_to_run == "baseline":
    env = BaselineEnv.make_env(
        config=config,
        **config.rl.environment_config.model_dump(),
    )
elif type_to_run == "debug":
    env = VmasEnv(
        scenario="simple_tag",
        num_envs=config.rl.algorithm_config.batch_size,
        continuous_actions=True,
        max_steps=100,
        device="cpu",
        seed=42,
        # Scenario specific
        num_good_agents=7,
        num_adversaries=7,
        num_landmarks=2,
    )
    env.append_transform(InitTracker())
    env.append_transform(
        RewardSum(
            in_keys=env.reward_keys,
            reset_keys=["_reset"] * len(env.group_map.keys()),
        )
    )
else:
    raise ValueError(
        f"Environment {type_to_run} is not supported. Supported environments: maddpg_global_local, maddpg_fed_sgd, debug"
    )


# %%
rl_module = TD3Module(env=env, **config.rl.algorithm_config.model_dump())

# %%
logger_list = init_logger(
    **config.rl.logger_config.model_dump(),
)
callbacks, plugins = init_callbacks_plugins(
    **config.rl.callback_config.model_dump(),
)
trainer = init_trainer(
    callbacks, plugins, logger_list, **config.rl.trainer_config.model_dump()
)

trainer.fit(rl_module)

# with initialize(version_base=None, config_path=str(config_path)):
#     cfg = compose(config_name="config.yaml")
#     print(OmegaConf.to_yaml(cfg))


# # %%
# def hydra_to_pydantic(config: DictConfig) -> Config:
#     """Converts Hydra config to Pydantic config."""
#     # use to_container to resolve
#     config_dict: Dict[str, Any] = OmegaConf.to_object(config)  # type: ignore[assignment]
#     return Config(**config_dict)


# # %%
# @hydra.main(version_base=None, config_path=Path(".configs"), config_name="config")
# def run(config: DictConfig) -> None:
#     logger.info("Type of config is: %s", type(config))
#     logger.info("Merged Yaml:\n%s", OmegaConf.to_yaml(config))
#     logger.info(HydraConfig.get().job.name)

#     config_pydantic = hydra_to_pydantic(config)
#     pprint(config_pydantic)


# if __name__ == "__main__":
#     run()
