from pathlib import Path
from typing import Union

from pydantic import BaseModel


class AEParams(BaseModel):
    block_size: int = 128
    fc_units: int = 256  # 128
    latent_dim: int = 64  # 8


class AEConfigModel(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32  # 128
    module_params: AEParams = AEParams()
    backbone_checkpoint: Union[Path, None] = None
