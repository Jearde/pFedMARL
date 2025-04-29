from enum import Enum
from pathlib import Path
from typing import Union

import lightning as L
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from models.ae.ae_model import AEModel
from models.ae.ae_pydentic import AEConfigModel


# Used
class ModelType(str, Enum):
    AEModel = "AEModel"


# Change this for the default model configs
MODEL = ModelType.AEModel


class NetworkModel(BaseModel):
    network_class: ModelType = MODEL
    network_params: Union[None, AEConfigModel] = (
        Field(AEConfigModel()) if MODEL == ModelType.AEModel else None
    )
    checkpoint_dir: Union[Path, None] = None

    @field_validator("network_params", mode="before")
    @classmethod
    def validate_network_params(cls, v, info: ValidationInfo):
        network_class = info.data.get("network_class")
        if network_class == AEModel:
            return AEConfigModel(**v)
        else:
            raise ValueError(f"Unsupported network_class: {network_class}")

    @field_validator("network_class", mode="after")
    @classmethod
    def transform(cls, raw: str) -> L.LightningModule:
        return get_class_from_str(raw)


def get_class_from_str(class_name: str):
    return globals()[class_name]
