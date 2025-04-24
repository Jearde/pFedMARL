import torch
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase
from torchrl.modules import (
    AdditiveGaussianModule,
    MultiAgentMLP,
    OrnsteinUhlenbeckProcessModule,
    ProbabilisticActor,
    SafeModule,
    TanhDelta,
    ValueOperator,
)
from torchrl.modules.distributions import TanhDelta
from torchrl.objectives import DDPGLoss, SoftUpdate
from torchrl.objectives.td3 import TD3Loss
from torchrl.objectives.utils import SoftUpdate, ValueEstimators


def make_maddpg_actors(
    env: EnvBase,
    share_parameters_policy: bool = True,
    centralized: bool = False,
    depth: int = 2,
    num_cells: int | list[int] = 256,
    activation_class: torch.nn.Module = torch.nn.ReLU,  # torch.nn.Tanh,
    distribution_class: type = TanhDelta,
    default_interaction_type=InteractionType.RANDOM,
) -> dict[str, ProbabilisticActor]:
    """
    Create the MADDPG agent consisting of the actor, critic, and an exploration policy.
    The actor maps observations to actions, while the critic estimates the state-action value.
    """
    policies = {}
    for group, _agents in env.group_map.items():
        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[group, "observation"].shape[
                    -1
                ],  # n_obs_per_agent
                n_agent_outputs=env.full_action_spec[group, "action"].shape[
                    -1
                ],  # n_actions_per_agents
                n_agents=len(_agents),  # Number of agents in the group
                centralised=centralized,  # the policies are decentralised (i.e., each agent will act from its local observation)
                share_params=share_parameters_policy,
                depth=depth,
                num_cells=num_cells,
                activation_class=activation_class,
            ),
            torch.nn.Softmax(dim=-1) if group == "server" else torch.nn.Tanh(),
        )

        # policy_module = TensorDictModule(
        policy_module = SafeModule(
            policy_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "param")],
        )  # We just name the input and output that the network will read and write to the input tensordict

        policy = ProbabilisticActor(
            module=policy_module,
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            distribution_class=distribution_class,
            distribution_kwargs={
                "low": env.full_action_spec[group, "action"].space.low,
                "high": env.full_action_spec[group, "action"].space.high,
            },
            default_interaction_type=default_interaction_type,
            return_log_prob=False,
        )
        policies[group] = policy

    return policies


def make_maddpg_explorer(
    policies: torch.nn.ModuleDict,  # dict[str, ProbabilisticActor],
    annealing_num_steps: int = 200,  # 1_000,
    device: torch.device | str | None = "cpu",
    noise_type: str = "AdditiveGaussian",
    init_value: float = 0.5,
    end_value: float = 0.001,
    mean: float = 0,
    std: float = 0.01,
):
    """
    Create a synchronous data collector for MADDPG.
    """
    # Wrap actor with an exploration noise module.
    exploration_policies = {}
    for group, agents in policies.items():
        if noise_type == "AdditiveGaussian":
            exploration_policy = AdditiveGaussianModule(
                spec=agents.spec,
                annealing_num_steps=annealing_num_steps,  # Number of frames after which sigma is sigma_end
                action_key=(group, "action"),
                sigma_init=init_value,
                sigma_end=end_value,
                mean=mean,
                std=std,
            )
        elif noise_type == "OrnsteinUhlenbeck":
            exploration_policy = OrnsteinUhlenbeckProcessModule(
                spec=agents.spec,
                annealing_num_steps=annealing_num_steps,  # Number of frames after which sigma is sigma_end
                action_key=(group, "action"),
                eps_init=init_value,
                eps_end=end_value,
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        exploration_policy = TensorDictSequential(agents, exploration_policy)

        if device == torch.device("cpu"):
            exploration_policy.share_memory()

        exploration_policies[group] = exploration_policy

    return exploration_policies


def make_maddpg_critics(
    env: EnvBase,
    centralised: bool = True,
    share_parameters_critic: bool = True,
    depth: int = 2,
    num_cells: int | list[int] = 256,
    activation_class: torch.nn.Module = torch.nn.ReLU,
):
    critics = {}

    for group, agents in env.group_map.items():
        # This module applies the lambda function: reading the action and observation entries for the group
        # and concatenating them in a new ``(group, "obs_action")`` entry
        cat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=[(group, "observation"), (group, "action")],
            out_keys=[(group, "obs_action")],
        )

        critic_module = ValueOperator(
            module=MultiAgentMLP(
                n_agent_inputs=env.observation_spec[group, "observation"].shape[-1]
                + env.full_action_spec[group, "action"].shape[-1],
                n_agent_outputs=1,  # 1 value per agent
                n_agents=len(agents),
                centralised=centralised,
                share_params=share_parameters_critic,
                depth=depth,
                num_cells=num_cells,
                activation_class=activation_class,
            ),
            in_keys=[(group, "obs_action")],  # Read ``(group, "obs_action")``
            out_keys=[
                (group, "state_action_value")
            ],  # Write ``(group, "state_action_value")``
        )

        critics[group] = TensorDictSequential(cat_module, critic_module)

    return critics


def make_maddpg_agents(
    env,
    share_parameters_policy: bool = False,
    centralized_policy: bool = False,
    depth_policy: int = 2,
    num_cells_policy: int | list[int] = 256,
    activation_class_policy: torch.nn.Module = torch.nn.ReLU,  # torch.nn.Tanh,
    annealing_num_steps=20,
    noise_type: str = "AdditiveGaussian",
    init_value: float = 0.5,
    end_value: float = 0.001,
    share_parameters_critic: bool = True,
    centralised_critic: bool = True,
    depth_critic: int = 2,
    num_cells_critic: int | list[int] = 256,
    activation_class_critic: torch.nn.Module = torch.nn.Tanh,
):
    policies = torch.nn.ModuleDict(
        make_maddpg_actors(
            env,
            share_parameters_policy=share_parameters_policy,
            centralized=centralized_policy,
            depth=depth_policy,
            num_cells=num_cells_policy,
            activation_class=activation_class_policy,
        )
    )

    critics = torch.nn.ModuleDict(
        make_maddpg_critics(
            env,
            share_parameters_critic=share_parameters_critic,
            centralised=centralised_critic,
            depth=depth_critic,
            num_cells=num_cells_critic,
            activation_class=activation_class_critic,
        )
    )

    explorers = torch.nn.ModuleDict(
        make_maddpg_explorer(
            policies,
            annealing_num_steps=annealing_num_steps,
            noise_type=noise_type,
            init_value=init_value,
            end_value=end_value,
        )
    )

    return policies, explorers, critics


def make_maddpg_loss_modules(
    env: EnvBase,
    policies,
    critics,
    gamma: float = 0.99,
    lmbda: float | None = None,
    polyak_tau: float = 0.005,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    actor_weight_decay: float = 0,
    critic_weight_decay: float = 1e-2,
):
    loss_modules = {}
    for group, _agents in env.group_map.items():
        loss_module = DDPGLoss(
            actor_network=policies[group],  # Use the non-explorative policies
            value_network=critics[group],
            delay_actor=False,
            delay_value=True,  # Whether to use a target network for the value
            loss_function="l2",
            separate_losses=False,
            reduction=None,
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        if lmbda is not None and lmbda >= 0.0:
            loss_module.make_value_estimator(
                ValueEstimators.TDLambda, gamma=gamma, lmbda=lmbda
            )
        else:
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

        loss_modules[group] = loss_module

    target_updaters = {
        group: SoftUpdate(loss, tau=polyak_tau) for group, loss in loss_modules.items()
    }

    optimizers = {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(),
                lr=actor_lr,
                weight_decay=actor_weight_decay,
            ),
            "loss_value": torch.optim.Adam(
                loss.value_network_params.flatten_keys().values(),
                lr=critic_lr,
                weight_decay=critic_weight_decay,
            ),
        }
        for group, loss in loss_modules.items()
    }

    return torch.nn.ModuleDict(loss_modules), target_updaters, optimizers


def make_td3_loss_modules(
    env: EnvBase,
    policies,
    critics,
    num_qvalue_nets: int = 2,
    loss_function: str = "smooth_l1",
    gamma: float = 0.99,
    lmbda: float | None = None,
    polyak_tau: float = 0.005,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    actor_weight_decay: float = 0,
    critic_weight_decay: float = 1e-2,
):
    loss_modules = {}
    for group, _agents in env.group_map.items():
        loss_module = TD3Loss(
            actor_network=policies[group],  # Use the non-explorative policies
            qvalue_network=critics[group],
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            action_spec=env.full_action_spec[group, "action"],
        )
        loss_module.set_keys(
            action=(group, "action"),
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        if lmbda is not None and lmbda >= 0.0:
            loss_module.make_value_estimator(
                ValueEstimators.TDLambda, gamma=gamma, lmbda=lmbda
            )
        else:
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

        loss_modules[group] = loss_module

    target_updaters = {
        group: SoftUpdate(loss, tau=polyak_tau) for group, loss in loss_modules.items()
    }

    optimizers = {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(),
                lr=actor_lr,
                weight_decay=actor_weight_decay,
            ),
            "loss_qvalue": torch.optim.Adam(
                loss.qvalue_network_params.flatten_keys().values(),
                lr=critic_lr,
                weight_decay=critic_weight_decay,
            ),
        }
        for group, loss in loss_modules.items()
    }

    return torch.nn.ModuleDict(loss_modules), target_updaters, optimizers


if __name__ == "__main__":
    pass
