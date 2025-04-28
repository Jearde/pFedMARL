# Federated Learning Client
import logging
import random

import lightning as L
import numpy as np
import torch
import torchinfo

from config.config_model import Config
from utils.fl_utils import aggregate, copy_, init_weights, subtract_
from utils.lightning_utils import init_callbacks_plugins, init_trainer
from utils.logger import flatten_dict, init_logger

logger = logging.getLogger("lightning.pytorch")


class Device:
    def __init__(
        self,
        config: Config,
        parent_id: str | None = None,
        client_id: str | int | None = None,
        tags: dict | None = None,
        fast: bool = False,
        verbose: bool = False,
        enable_model_summary: bool = False,
    ):
        self.client_id = client_id
        self.config = config
        self.fast = fast
        self.not_verbose = not verbose
        self.enable_model_summary = enable_model_summary

        self.init_seeds()

        if not self.fast:
            if self.not_verbose:
                self.config.trainer.enable_progress_bar = False
            else:
                self.config.trainer.enable_progress_bar = True

            self.init_logger(parent_id=parent_id, tags=tags)
        elif not self.not_verbose:
            self.config.trainer.enable_progress_bar = True
        else:
            self.not_verbose = True
            self.config.trainer.enable_checkpointing = False
            self.config.trainer.enable_progress_bar = False
            self.config.trainer.log_every_n_steps = 0
            self.logger_list = [
                L.pytorch.loggers.logger.DummyLogger(),
            ]
            self.model_checkpoint_path = None
            self.callbacks = []
            self.plugins = []
            logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    def init_variables(self):
        self.validation_loss_before_training = torch.tensor([0.0])
        self.validation_loss_after_training = torch.tensor([0.0])
        self.validation_losses_after_training = torch.tensor([0.0])

        self.W = self.get_model()
        self.W_old = self.get_model()
        self.dW = {key: torch.zeros_like(value) for key, value in dict(self.W).items()}

        self.model_l2_norm = {
            k: torch.zeros(1)
            for k, v in self.model.state_dict().items()
            if v.dtype == torch.float
        }

    def init_trainable(
        self,
        epoch: int | None = None,
        extra_callbacks: list | None = None,
        data_module: L.LightningDataModule | None = None,
        label_names: list[str] | None = None,
    ):
        self.config.trainer.max_epochs = epoch or self.config.trainer.max_epochs

        if not self.fast:
            self.init_callbacks_plugins(extra_callbacks=extra_callbacks)

        self.init_trainer()

        self.init_data(data_module)

        self.init_model(label_names=label_names)

    def init_seeds(self, seed: int | None = None):
        seed = seed or self.config.seed

        # Setting the seed
        L.seed_everything(seed, workers=True)
        torch.set_float32_matmul_precision("high")  # 'highest', 'high', 'medium', 'low'
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms = True
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def init_logger(
        self,
    ):
        self.logger_list, self.model_checkpoint_path = init_logger(
            **self.config.logger.model_dump(),
            client_id=self.client_id,
        )

        return self.logger_list, self.model_checkpoint_path

    def init_callbacks_plugins(self, extra_callbacks: list | None = None):
        logger.info("Initializing Trainer callbacks and plugins")
        self.callbacks, self.plugins = init_callbacks_plugins(
            # **self.config.rl.callback_config.model_dump(),
            model_checkpoint_path=self.model_checkpoint_path,
            extra_callbacks=extra_callbacks,
        )

        return self.callbacks, self.plugins

    def init_trainer(self, enable_model_summary: bool = False):
        # Initialize a trainer
        logger.info("Initializing Trainer")

        self.trainer = init_trainer(
            self.callbacks,
            self.plugins,
            self.logger_list,
            **self.config.trainer.model_dump(),
            enable_model_summary=enable_model_summary,
        )

        return self.trainer

    def init_data(self, data_module: L.LightningDataModule | None = None):
        # Get the dataset
        logger.info("Initializing Data Set")
        self.data_module = data_module
        self.data_module.prepare_data()

        return self.data_module

    def get_feature_dim(self):
        logger.info("Preparing Test Data")
        self.data_module.setup(stage="test")

        input_size = self.config.network.network_class.get_feature_dim(
            next(iter(list(self.data_module.test_dataloader().values())[0]))[0][0],
            self.config.feature.model_dump(),
        )
        logger.info(f"Feature size: {input_size}")

        return input_size

    def get_input_dim(self):
        logger.info("Preparing Test Data")
        self.data_module.setup(stage="test")

        input_size = next(iter(list(self.data_module.test_dataloader().values())[0]))[
            0
        ][0].shape
        logger.info(f"Input size: {input_size}")

        return input_size

    def init_model(self, label_names: list[str] | None = None):
        input_size = self.get_input_dim()
        feature_size = self.get_feature_dim()

        logger.info("Initializing Model")
        # TODO: No class should be passed, as it leads to logging issues
        self.model = self.config.network.network_class(
            input_dim=feature_size,
            epochs=self.config.trainer.max_epochs,
            **self.config.network.network_params.model_dump(),
            single_dataset=False if len(self.data_module.test_datasets) > 1 else True,
            feature_extractor_config=self.config.feature.model_dump(),
            label_names=label_names,
        )

        if self.client_id is None and self.enable_model_summary:
            if self.config.logger.tensorboard_use:
                self.logger_list[0].log_graph(
                    self.model,
                    torch.zeros(
                        (self.config.network.network_params.batch_size, *input_size)
                    ),
                )

            self.get_summary(self.model.feature_extractor)

            x, y = self.model.feature_forward(
                next(iter(list(self.data_module.test_dataloader().values())[0]))[0].to(
                    self.model.device
                )
            )

            self.get_summary(self.model, input_size=x[0].shape)

        self.init_variables()

    def reset_parameters(self):
        self.model.apply(init_weights)
        self.sync_active_to_W()

    def get_model(
        self,
        remove_keys: list = [
            "num_batches_tracked",
            # "running_mean", # Do not comment in
            # "running_var",
            "feature_extractor",
        ],
    ) -> dict[str, torch.Tensor]:
        # Get model weights and states
        # W = dict(self.model.state_dict())
        W = {
            k: v.detach().cpu().data.clone() for k, v in self.model.state_dict().items()
        }

        # Update required gradients
        for key, value in self.model.named_parameters():
            W[key].requires_grad = value.requires_grad

        # Remove states that should not be averaged
        W = {k: v for k, v in W.items() if not any(map(k.__contains__, remove_keys))}

        return W

    def load_model(self, model_dict: dict[str, torch.Tensor], reset=False):
        result = self.trainer.validate(
            self.model,
            datamodule=self.data_module,
            verbose=False if self.not_verbose else True,
        )
        self.validation_loss_before_loading = result[0]["val/loss"]

        model_dict = {k: v.detach().cpu().data.clone() for k, v in model_dict.items()}

        for key, value in self.model.named_parameters():
            model_dict[key].requires_grad = value.requires_grad

        missing_keys, unexpected_keys = self.model.load_state_dict(
            model_dict, strict=False
        )
        self.sync_active_to_W()

        result = self.trainer.validate(
            self.model,
            datamodule=self.data_module,
            verbose=False if self.not_verbose else True,
        )
        self.validation_loss_after_loading = result[0]["val/loss"]

        if reset:
            self.init_variables()

    def sync_W_to_active(self, W: dict[str, torch.Tensor] | None = None):
        if W is not None:
            self.W = W
        self.load_model(self.W)
        return self.W

    def sync_active_to_W(self):
        self.W = self.get_model()
        return self.W

    def save_old_model(self):
        self.sync_active_to_W()
        copy_(target=self.W_old, source=self.W)
        return self.W_old

    def load_old_model(self):
        """Reset weights to state before training cycle."""
        copy_(target=self.W, source=self.W_old)
        self.load_model(self.W_old)
        return self.W

    def get_model_update(self) -> dict[str, torch.Tensor]:
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return self.dW

    def add_model_update(
        self,
        update: dict[str, torch.Tensor],
        weight: float = 1.0,
        target: dict[str, torch.Tensor] | None = None,
        remove_keys: list = [
            "num_batches_tracked",
            # "running_mean",
            # "running_var",
            "feature_extractor",
        ],
    ) -> dict[str, torch.Tensor]:
        target = target or self.W

        update = {
            k: v for k, v in update.items() if not any(map(k.__contains__, remove_keys))
        }

        for key, value in update.items():
            target[key].data += weight * value.data.clone()

        return target

    def aggregate(
        self,
        target: dict,
        sources: list[dict],
        source_weights: torch.Tensor,
        normalize_weights: bool = True,
        is_gradient_update: bool = True,
        remove_keys=[
            "num_batches_tracked",
            # "running_mean",
            # "running_var",
            "feature_extractor",
        ],
    ) -> dict:
        aggregate(
            target=target,
            sources=sources,
            weights=source_weights,
            is_gradient_update=is_gradient_update,
            normalize_weights=normalize_weights,
            remove_keys=remove_keys,
        )

        return target

    def get_summary(
        self,
        model,
        input_size: tuple | None = None,
        input_data: torch.Tensor | None = None,
    ):
        torchinfo.summary(
            model,
            input_size=tuple(
                [self.config.network.network_params.batch_size] + list(input_size)
            )
            if input_size is not None
            else None,
            input_data=next(iter(list(self.data_module.test_dataloader().values())[0]))[
                0
            ].to(self.model.device)
            if input_size is None
            else input_data,
            depth=10,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ),
            verbose=1,
        )

    def log_hparams(self, results: dict):
        hparam_dict = flatten_dict(self.config.model_dump())
        metric_dict = flatten_dict(results, prefix="hparam/")

        self.logger_list[0].log_hyperparams(
            hparam_dict,
            metrics=metric_dict,
        )

    def merge_dict_list(self, dict_list: list[list[dict]]):
        results = {}
        for client_result in dict_list:
            for result in client_result:
                for key, value in result.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
        # Mean of all results
        for key, value in results.items():
            results[key] = np.mean(value)

        return results

    def validate(self):
        return self.trainer.validate(
            datamodule=self.data_module,
            model=self.model,
            verbose=False,
        )
