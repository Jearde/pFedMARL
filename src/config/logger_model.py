from pathlib import Path

from pydantic import BaseModel

from config.data_loader_model import LOG_FOLDER


class LoggerModel(BaseModel):
    logger_name: str = "pFedMARL"

    tensorboard_dir: Path = LOG_FOLDER / Path("tensorboard")
