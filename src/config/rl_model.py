from pathlib import Path

from pydantic import BaseModel

from config.logger_model import LOG_FOLDER, LoggerModel
from config.trainer_model import CallbackModel, TrainerModel


class EnvironmentModel(BaseModel):
    enable_checks: bool = False
    max_servers: int = 1
    max_clients: int | None = None
    batch_size: int | None = None
    device: str = "cpu"
    fast: bool = True
    max_global_steps: int | None = 0
    norm_obs_reward: bool = False


class AgentModel(BaseModel):
    # Policy Network
    share_parameters_policy: bool = False
    centralized_policy: bool = False
    depth_policy: int = 2
    num_cells_policy: int | list[int] = 256
    annealing_num_steps: int = 90
    noise_type: str = "AdditiveGaussian"
    init_value: float = 0.25
    end_value: float = 0.1
    # Value Network
    share_parameters_critic: bool = True
    centralised_critic: bool = True
    depth_critic: int = 2
    num_cells_critic: int | list[int] = 256


class DDPGLossModel(BaseModel):
    gamma: float = 0.99
    lmbda: float | None = None
    polyak_tau: float = 0.99
    actor_lr: float = 4e-4
    critic_lr: float = 3e-3
    actor_weight_decay: float = 0.0
    critic_weight_decay: float = 1e-2


class TD3LossModel(BaseModel):
    gamma: float = 0.82
    lmbda: float | None = 0.67
    polyak_tau: float = 0.99
    actor_lr: float = 1e-3
    critic_lr: float = 1e-4
    actor_weight_decay: float = 0.0
    critic_weight_decay: float = 0.0

    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"


class CollectorModel(BaseModel):
    total_frames: int = -1
    frames_per_batch: int = 1
    max_frames_per_traj: int | None = (
        None  # Reset the environment after this many frames
    )
    init_random_frames: int = 0
    device: str | None = "cpu"
    storing_device: str | None = "cpu"
    env_device: str | None = "cpu"
    policy_device: str | None = "cpu"


class BufferModel(BaseModel):
    device: str | None = "cpu"
    train_batch_size: int = 8
    memory_size: int = 100000
    prb: bool = True
    alpha: float = 0.7
    beta: float = 0.5
    prefetch: int = 0


class MADDPGModel(BaseModel):
    batch_size: int = 8
    train_freq: float = 1
    collections_per_call: int = 1
    initial_collect_steps: int = 1
    max_grad_norm: float = 10.0
    test_freq: int = 200
    policy_update_delay: int = 2
    perform_validation: bool = False
    agent_config: AgentModel = AgentModel()
    loss_config: DDPGLossModel | TD3LossModel = DDPGLossModel()
    collector_config: CollectorModel = CollectorModel()
    buffer_config: BufferModel = BufferModel()


class ReinforcementLearningModel(BaseModel):
    environment_name: str = (
        "maddpg_global_local"  # maddpg_global_local maddpg_fed_sgd debug
    )
    environment_config: EnvironmentModel = EnvironmentModel()
    algorithm_name: str = "td3"  # "maddpg"
    algorithm_config: MADDPGModel = MADDPGModel()
    logger_config: LoggerModel = LoggerModel(
        logger_name="pFedMARL",
        tensorboard_dir=Path(f"{LOG_FOLDER}/tensorboard"),
    )
    trainer_config: TrainerModel = TrainerModel(
        max_epochs=200,
        log_every_n_steps=1,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=1,
    )
    callback_config: CallbackModel = CallbackModel(save_model=False)
