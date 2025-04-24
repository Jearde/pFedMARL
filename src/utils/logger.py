import logging
import os
from enum import Enum
from pathlib import Path

import dagshub
import lightning as L
import mlflow
import numpy as np
import torch
import torchaudio
import torchinfo
import torchmetrics
import torchvision

logger = logging.getLogger("lightning.pytorch")


def print_stats():
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Lightning Version: {L.__version__}")
    logger.info(f"Torchvision Version: {torchvision.__version__}")
    logger.info(f"Torchaudio Version: {torchaudio.__version__}")
    logger.info(f"PyTorch Metrics Version: {torchmetrics.__version__}")
    logger.info(f"MLFlow Version: {mlflow.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")

def init_logger(
    tensorboard_dir: Path,
    logger_name: str,
    client_id: str | int | None = None,
):
    logger.info(f"Init Logger for {client_id if client_id is not None else 'Server'}")

    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=tensorboard_dir,
        name=logger_name,
        default_hp_metric=False,
    )

    logger.info(f"Tensorboard: {tb_logger.log_dir}")
    logger.info(
        f"Open with: tensorboard --logdir {str(Path(tb_logger.log_dir).parent)} --port 6007 --host 0.0.0.0"
    )

    logger_list = [
        logger
        for logger in [tb_logger]
        if logger is not None
    ]

    return logger_list