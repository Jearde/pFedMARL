import random

import lightning as L
import numpy as np
import torch


def init_seeds(seed: int | None = None):
    # Setting the seed
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")  # 'highest', 'high', 'medium', 'low'
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_callbacks_plugins(
    extra_callbacks: list | None = None,
    extra_plugins: list | None = None,
    async_logging: bool = False,
    verbose: bool = False,
    client_id: str | None = None,
    model_checkpoint_path: str | None = None,
    save_model: bool = True,
):
    callbacks = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    if save_model:
        callbacks.append(
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=model_checkpoint_path,
                filename="epoch={epoch}-step={global_step}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                every_n_epochs=1,  # Save every n epochs (0 = disabled save top k)
                save_last=True,
                verbose=False,
            )
        )
    if verbose:
        callbacks.append(L.pytorch.callbacks.RichProgressBar(refresh_rate=1))
    if client_id is None and verbose:
        callbacks.append(L.pytorch.callbacks.RichModelSummary(max_depth=-1))

    callbacks.extend(extra_callbacks or [])

    if async_logging:
        plugins = [
            L.pytorch.plugins.io.AsyncCheckpointIO(),  # Log MLflow models without blocking. Can cause errors, when the model is not fully saved before needed.
        ]
    else:
        plugins = []

    if plugins is not None:
        plugins.extend(extra_plugins or [])
    else:
        plugins = plugins

    return callbacks, plugins


def init_trainer(
    callbacks: list, plugins: list, logger_list: list, devices: int, **kwargs
):
    extra_plugins = kwargs.pop("plugins", None)
    plugins.extend(extra_plugins or [])
    trainer = L.Trainer(
        **kwargs,
        callbacks=callbacks,
        plugins=plugins,
        logger=logger_list,
        strategy="ddp_find_unused_parameters_true"
        if devices > 1 or devices == -1
        else "auto",
    )

    return trainer
