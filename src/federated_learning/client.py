# Federated Learning Client
import logging
from pathlib import Path

import lightning as L
import torch

from federated_learning.device import Device

logger = logging.getLogger("lightning.pytorch")


class Client(Device):
    def __init__(
        self,
        client_id: str | int,
        data_module: L.LightningDataModule,
        epoch: int | None = None,
        extra_callbacks: list | None = None,
        label_names: list[str] | None = None,
        reset_trainer_on_fit: bool = True,
        **kwargs,
    ):
        super().__init__(client_id=client_id, **kwargs)
        self.epoch = epoch
        self.init_trainable(
            epoch=epoch,
            extra_callbacks=extra_callbacks,
            data_module=data_module,
            label_names=label_names,
        )
        self.reset_trainer_on_fit = reset_trainer_on_fit

        self.data_module.setup("fit")
        self.data_module.setup("test")

    def fit(self, checkpoint: Path | None = None, fl_mode: bool = True):
        if fl_mode:
            # Get loss before training
            result = self.trainer.validate(
                self.model,
                datamodule=self.data_module,
                verbose=False if self.not_verbose else True,
            )
            self.validation_loss_before_training = torch.tensor([result[0]["val/loss"]])

            # Save the model before training
            self.sync_W_to_active()
            self.save_old_model()

        if self.reset_trainer_on_fit:
            self.trainer.fit_loop.epoch_progress.reset()
            self.model.fl_round += 1

        # Train the model
        self.trainer.fit(
            self.model,
            datamodule=self.data_module,
            ckpt_path=checkpoint,
        )

        if fl_mode:
            self.sync_active_to_W()
            self.get_model_update()

            # Get loss after training
            result = self.trainer.validate(
                self.model,
                datamodule=self.data_module,
                verbose=False if self.not_verbose else True,
            )
            self.validation_loss_after_training = torch.tensor([result[0]["val/loss"]])
            self.validation_losses_after_training = self.model.validation_losses

        if self.trainer.checkpoint_callback:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
        else:
            best_model_path = None

        return best_model_path

    def test(self):
        logger.info("Start testing")
        self.trainer.test(
            model=self.model,
            datamodule=self.data_module,
            # ckpt_path="last",  # "best", "last", "path/to/checkpoint"
        )

        if self.trainer.checkpoint_callback:
            best_model_path = self.trainer.checkpoint_callback.best_model_path
        else:
            best_model_path = None

        return best_model_path, self.get_run_id()
