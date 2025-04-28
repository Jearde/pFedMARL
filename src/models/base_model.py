import copy
import logging

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.feature.identity_feature import Identity

logger = logging.getLogger("lightning.pytorch")

# Hooks https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks


class BaseModel(L.LightningModule):
    def __init__(
        self,
        input_dim: tuple,
        module_params: dict,
        epochs: int,
        lr: float = 0.001,
        batch_size: int = 32,
        norm_pix_loss: bool = False,
        log_projector: bool = False,
        classifiers: dict = None,
        feature_extractor_config: dict = None,
        label_names: list[str] | None = None,
        reduce_test_samples: bool = False,
        use_preprocessing: bool = False,
        plot_reconstruction: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Autoencoder Network

        # Attributes for training
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.classifiers = classifiers
        self.fl_round = 0

        self.lr = lr
        self.norm_pix_loss = norm_pix_loss
        self.log_projector = log_projector

        # Just forwards the input
        self.module_params = module_params
        self.feature_extractor_config = feature_extractor_config
        self.use_preprocessing = use_preprocessing
        self.plot_reconstruction = plot_reconstruction

        self.dataloader_idx = 0
        self.latent_labels = {}

        if label_names is not None:
            for label_name in label_names:
                self.latent_labels[label_name] = np.array([])

        self.reduce_test_samples = reduce_test_samples

        self.model = None
        self.feature_extractor = Identity()

    def feature_forward(self, x, y=None, shuffle=True):
        x, y = self.feature_extractor(x, y, shuffle=shuffle)
        return x, y

    @classmethod
    def get_feature_extractor(cls, feature_extractor_config: dict | None) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def get_feature_dim(cls, x, feature_extractor_config: dict | None):
        raise NotImplementedError

    def forward(self, x):
        x_hat, z = self.model(x)

        return x, z, [x_hat]

    def loss_recon_forward(self, x, x_hat, mask=None):
        if mask is not None:
            return F.mse_loss(x_hat, x.view(x_hat.shape), reduction="none") * mask
        return F.mse_loss(x_hat, x.view(x_hat.shape), reduction="none")

    def loss_forward(self, x, x_hat, labels, preds):
        loss_recon = self.loss_recon_forward(x, x_hat)
        loss = loss_recon.mean()

        return loss, [loss_recon]

    def on_fit_start(self):
        super().on_fit_start()
        self.model_before_fit = copy.deepcopy(self.state_dict())

    def on_fit_end(self):
        super().on_fit_end()
        self.model_after_fit = copy.deepcopy(self.state_dict())

        self.mode_update = {}
        for key in self.model_before_fit.keys():
            self.mode_update[key] = (
                self.model_after_fit[key].cpu() - self.model_before_fit[key].cpu()
            )

        pass

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

    def log_loss(
        self, loss, loss_separate, acc_separate=None, log_type="train", recon_idx=None
    ):
        self.log(
            "fl/fl_round",
            self.fl_round,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            "fl/overall_epoch",
            (self.fl_round * self.epochs) + self.current_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            f"{log_type}/loss",
            loss,
            on_step=True if log_type == "train" else False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=True if log_type == "test" else False,
        )

        if recon_idx is not None:
            self.log(
                f"{log_type}/loss_recon",
                loss_separate[recon_idx].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False if log_type == "train" else True,
                logger=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )
            head_idx = recon_idx + 1
        else:
            head_idx = 0

        for i, loss_sep in enumerate(loss_separate[head_idx:]):
            self.log(
                f"{log_type}/loss_{self.classifiers[i]['label_name']}",
                loss_sep.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )
            if acc_separate is not None:
                self.log(
                    f"{log_type}/acc_{self.classifiers[i]['label_name']}",
                    acc_separate[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if log_type == "train" else True,
                    logger=True,
                    batch_size=self.batch_size,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        x, y = batch

        x_vec, y_vec = self.feature_forward(x, y, shuffle=True)

        x_vec, z, preds = self.forward(x_vec)
        x_hat = preds[0]

        loss, loss_separate = self.loss_forward(x_vec, x_hat, y_vec, preds[1:])
        loss = loss.mean()

        self.log_loss(
            loss,
            loss_separate,
            acc_separate=None,
            log_type="train",
            recon_idx=0,
        )

        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

    def on_validation_start(self):
        super().on_validation_start()
        self.validation_losses = torch.tensor([])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_vec, y_vec = self.feature_forward(x, y, shuffle=False)

        x, z, preds = self.forward(x_vec)
        x_hat = preds[0]

        loss, loss_separate = self.loss_forward(x_vec, x_hat, y_vec, preds[1:])
        loss = loss.mean()

        self.log_loss(
            loss,
            loss_separate,
            log_type="val",
            recon_idx=0,
        )

        self.validation_losses = torch.cat(
            [
                self.validation_losses,
                loss_separate[0]
                .mean(axis=tuple(np.arange(loss_separate[0].ndim)[1:]))
                .cpu()
                .detach(),
            ]
        )

        return loss

    def on_validation_end(self):
        super().on_validation_end()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != self.dataloader_idx:
            self.dataloader_idx = dataloader_idx

        x, y = batch

        x_vec, y_vec = self.feature_forward(x, y, shuffle=False)

        x_vec, z, preds = self.forward(x_vec)
        x_hat = preds[0]

        loss, loss_separate = self.loss_forward(x_vec, x_hat, y_vec, preds[1:])
        loss = loss.mean()

        self.log_loss(
            loss,
            loss_separate,
            log_type="test",
            recon_idx=0,
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true = batch

        x_vec, y_vec = self.feature_forward(x, y_true, shuffle=False)
        x, z, preds = self.forward(x_vec)

        return (x_vec, z, None, preds), y_vec

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true = batch

        x_vec, y_vec = self.feature_forward(x, y_true, shuffle=False)
        x_vec, z, preds = self.forward(x_vec)

        new_shape = x.size(0), -1, *list(x_vec.shape[1:])
        x_vec = x_vec.reshape(new_shape)
        z = z.view(x.size(0), -1, z.size(-1)).cpu()  # .numpy()
        preds[0] = preds[0].view(new_shape).cpu()
        preds[1:] = [
            pred.view(x.size(0), -1, pred.size(-1)).cpu() for pred in preds[1:]
        ]
        y_vec = [y.view(x.size(0), -1).cpu() for y in y_vec]

        return (x_vec, z, None, preds), y_vec

    def configure_optimizers(self):
        raise NotImplementedError
