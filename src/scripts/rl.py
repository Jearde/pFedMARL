import logging
from pathlib import Path

from torchrl.envs import VmasEnv
from torchrl.envs.transforms import InitTracker, RewardSum

from config.config_model import Config, parse_config
from environment.baseline import BaselineEnv
from environment.fed_sgd import FedSGDEnv
from reinforcement_learning.td3.td3 import TD3Module
from utils.lightning_utils import init_callbacks_plugins, init_seeds, init_trainer
from utils.logger import init_logger

logger = logging.getLogger("lightning.pytorch")
logging.getLogger("lightning").setLevel(logging.ERROR)


def rl_train(
    type_to_run: str = "debug",
    config_path: Path | str | None = None,
    config: Config | None = None,
    extra_callbacks: list | None = None,
):
    if config is None and config_path is not None:
        config = parse_config(config_path)
    elif config is None and config_path is None:
        raise ValueError("Either config or config_path must be provided.")

    init_seeds(config.seed)

    if type_to_run == "baseline":
        env = BaselineEnv.make_env(
            config=config,
            **config.rl.environment_config.model_dump(),
        )
    elif type_to_run == "fedavg":
        env = FedSGDEnv.make_env(
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

    rl_module = TD3Module(env=env, **config.rl.algorithm_config.model_dump())

    logger_list, model_checkpoint_path = init_logger(
        **config.rl.logger_config.model_dump(),
        config_path=config.config_path,
    )
    callbacks, plugins = init_callbacks_plugins(
        **config.rl.callback_config.model_dump(),
        model_checkpoint_path=model_checkpoint_path,
        extra_callbacks=extra_callbacks,
    )
    trainer = init_trainer(
        callbacks, plugins, logger_list, **config.rl.trainer_config.model_dump()
    )

    trainer.fit(rl_module)

    return env, rl_module
