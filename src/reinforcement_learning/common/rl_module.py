import copy
import logging

import lightning as L
import optuna
import torch
import tqdm
from lightning.pytorch.utilities import grad_norm
from torchrl.data.replay_buffers import TensorDictReplayBuffer

from environment.base_env import BaseFLEnv
from utils.logger import flatten_dict

from .storage import ReplayBufferDataset, append_replay_buffer, make_collector

logger = logging.getLogger("lightning.pytorch")
logging.getLogger("lightning").setLevel(logging.ERROR)


class DRLModule(L.LightningModule):
    def __init__(
        self,
        env: BaseFLEnv,
        batch_size: int = 64,
        train_freq: float = 1,
        collections_per_call: int = 1,
        initial_collect_steps: int = 1,
        max_grad_norm: float = 10.0,
        test_freq: int = 200,
        policy_update_delay: int = 0,
        buffer_config: dict = {},
        perform_validation: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])

        self.env = env
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.train_freq = train_freq
        self.collections_per_call = collections_per_call
        self.initial_collect_steps = initial_collect_steps
        self.test_freq = test_freq
        self.policy_update_delay = policy_update_delay
        self.prb = buffer_config.get("prb", False)

        self.perform_validation = perform_validation
        self.train_group_map = copy.deepcopy(self.env.group_map)

        self.automatic_optimization = False

        self.actors = None
        self.actor_explores = None
        self.critics = None
        self.loss_modules = None
        self.target_updaters = None
        self.rl_optimizers = None
        self.collector = None
        self.replay_buffers: dict[str, TensorDictReplayBuffer] = {}

    def on_fit_start(self):
        if hasattr(self.env, "plot_class_distribution"):
            fig = self.env.plot_class_distribution()
            self.loggers[0].experiment.add_figure(
                "class_distribution", fig, self.global_step
            )

        self.collector = make_collector(
            self.env,
            self.actor_explores,
            env_device="cpu",
            storing_device="cpu",
            policy_device=self.device,
        )
        first_group = next(iter(self.env.group_map.keys()))

        self.env.reset()

        pbar = tqdm.tqdm(
            total=self.initial_collect_steps, desc="Collecting initial data"
        )
        while len(self.replay_buffers[first_group]) < self.initial_collect_steps:
            batch = next(iter(self.collector))
            append_replay_buffer(
                batch,
                self.train_group_map.keys(),
                self.replay_buffers,
                self.actor_explores,
            )
            group = list(self.env.group_map.keys())[0]  # "server"
            pbar.set_postfix(reward=batch["next"][group]["reward"].mean().item())
            pbar.update(1)

        self.env.reset()

    def on_train_epoch_start(self):
        self.collector.update_policy_weights_()

        self.log_test(verbose=False)

        # Run the collector every `train_freq` batches.
        if self.global_step % self.train_freq == 0:
            # Iterate over your collector to get new data
            for i, batch in tqdm.tqdm(
                enumerate(self.collector),
                desc="Collecting data",
                total=self.collections_per_call,
                leave=False,
            ):
                # Assuming data is a torch tensor on the appropriate device.
                append_replay_buffer(
                    batch,
                    self.train_group_map.keys(),
                    self.replay_buffers,
                    self.actor_explores,
                )

                for group in self.env.group_map.keys():
                    if batch["next"][group]["reward"].isnan().any():
                        logger.error("Nan in reward for group %s", group)
                        if group == "client":
                            raise optuna.exceptions.TrialPruned()
                            # self.trainer.should_stop = True
                        else:
                            batch["next"][group]["reward"] = -100 * torch.ones_like(
                                batch["next"][group]["reward"]
                            )
                            pass

                if i + 1 >= self.collections_per_call:
                    break

    def forward(self, state):
        return self.actors(state)

    def log_hparams(self, results, hp_metric_name="mean_auc_recon"):
        hparam_dict = flatten_dict(self.hparams)
        metric_dict = flatten_dict(results, prefix="hp/")

        self.logger.log_hyperparams(
            hparam_dict,
            metrics=metric_dict,
        )
        self.loggers[0].experiment.add_scalar("hp_metric", results[hp_metric_name])

    def log_results_dict(self, dict: dict[str, torch.Tensor], prefix=""):
        for key, value in dict.items():
            key = prefix + key
            try:
                self.log(key, value)
            except Exception:
                self.loggers[0].experiment.add_scalar(key, value)

    def log_environment(self, group_data, group):
        if not hasattr(self.env, "get_observation_dict"):
            return None
        observation = self.env.get_observation_dict(group_data, group, prefix="train/")
        self.log_dict(observation, prog_bar=False)
        pass

    def log_test(self, **kwargs):
        if not hasattr(self.env, "test"):
            return None

        results, results_fine = self.env.test(**kwargs)

        self.log_results_dict(results, prefix="test/")
        self.log_results_dict(results_fine, prefix="fine/")

        return results

    def _step(self, group_data, group):
        losses = {}

        loss_vals = self.loss_modules[group](group_data)

        for loss_name in ["loss_actor", "loss_value"]:
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
                self.loss_modules[group].actor_network.module
                if "actor" in loss_name
                else self.loss_modules[group].value_network.module
            )
            norm = sum(grad_norm(network, norm_type=2).values())

            losses["grad_norm_" + loss_name + "_" + group] = norm

        # Update target networks
        self.target_updaters[group].step()

        loss = sum([loss_vals[key].mean() for key in ["loss_actor", "loss_value"]])

        return losses, loss

    def training_step(self, batch, batch_idx):
        losses = {}
        rewards = []
        for group in self.train_group_map.keys():
            self.loss_modules[group] = self.loss_modules[group].to(self.device)

            group_data = batch[group]

            step_losses, loss = self._step(group_data, group)
            losses.update(step_losses)

            # Update priorities in the replay buffer
            if self.prb:
                self.replay_buffers[group].update_tensordict_priority(batch)

            losses["replay_buffer_size"] = len(self.replay_buffers[group])
            losses["rewards/" + group] += group_data["next"][group]["reward"].mean()
            for idx, client_loss in enumerate(
                group_data["next"][group]["reward"].mean(0).mean(-1)
            ):
                self.log(f"rewards/{group}_{idx}", client_loss, prog_bar=False)
            rewards.append(group_data["next"][group]["reward"].mean())

            self.log_environment(group_data, group)

        self.log("loss", loss, prog_bar=True)
        self.log("reward", sum(rewards) / len(rewards), prog_bar=True)
        self.log_dict(losses, prog_bar=False)

        return None

    def on_train_epoch_end(self):
        if self.perform_validation:
            self.log_test(verbose=False)

    def configure_optimizers(self):
        return None

    def train_dataloader(self):
        return {
            name: torch.utils.data.DataLoader(
                ReplayBufferDataset(replay_buffer, self.batch_size),
                batch_size=None,
                collate_fn=lambda x: x,
                num_workers=0,
            )
            for name, replay_buffer in self.replay_buffers.items()
        }
