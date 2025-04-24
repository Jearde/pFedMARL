# https://pytorch.org/rl/stable/tutorials/coding_ppo.html
# https://pytorch.org/rl/stable/tutorials/multiagent_competitive_ddpg.html
# https://pytorch.org/rl/stable/tutorials/coding_ddpg.html
# https://github.com/XuehaiPan/torchrl/blob/main/examples/ddpg/ddpg.py
# https://github.com/XuehaiPan/torchrl/blob/main/examples/td3/td3.py

# Hooks https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks


from lightning.pytorch.utilities import grad_norm

from ..common.agent import make_maddpg_agents, make_td3_loss_modules
from ..common.rl_module import DRLModule
from ..common.storage import make_collector, make_replay_buffers


class TD3Module(DRLModule):
    def __init__(
        self,
        *args,
        agent_config: dict = {},
        loss_config: dict = {},
        collector_config: dict = {},
        buffer_config: dict = {},
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["env"])

        # Initialize actor and critic networks
        self.actors, self.actor_explores, self.critics = make_maddpg_agents(
            self.env, **agent_config
        )

        # Initialize the loss modules, target updaters, and optimizers
        self.loss_modules, self.target_updaters, self.rl_optimizers = (
            make_td3_loss_modules(self.env, self.actors, self.critics, **loss_config)
        )

        # Initialize the environment collector
        self.collector = make_collector(
            self.env, self.actor_explores, **collector_config
        )

        # Initialize the replay buffers
        self.replay_buffers = {
            group_name: make_replay_buffers(self.env, **buffer_config)
            for group_name in self.env.group_map.keys()
        }

        self.loss_names = ["loss_actor", "loss_qvalue"]

    def _step(self, group_data, group):
        losses = {}

        self.loss_modules = self.loss_modules.to(self.device)

        loss_vals = self.loss_modules[group](group_data)

        for loss_name in self.loss_names:
            loss = loss_vals[loss_name]
            optimizer = self.rl_optimizers[group][loss_name]

            if (
                self.current_epoch % self.policy_update_delay != 0
                and "actor" in loss_name
            ):
                continue

            optimizer.zero_grad()
            self.manual_backward(loss, retain_graph="actor" in loss_name)
            self.clip_gradients(optimizer, self.max_grad_norm)
            optimizer.step()

            # Log the loss
            if loss_name not in losses:
                losses[loss_name + "_" + group] = 0.0
                losses["rewards/" + group] = 0.0
            losses[loss_name + "_" + group] += loss

            network = (
                self.loss_modules[group].actor_network_params
                if "actor" in loss_name
                else self.loss_modules[group].qvalue_network_params
            )
            norm = sum(grad_norm(network, norm_type=2).values())

            losses["grad_norm_" + loss_name + "_" + group] = norm

        if self.current_epoch % self.policy_update_delay == 0:
            # Update target networks
            self.target_updaters[group].step()

        loss = sum([loss_vals[key].mean() for key in self.loss_names])

        return losses, loss
