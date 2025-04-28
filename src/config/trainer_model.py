from typing import Union

from pydantic import BaseModel


class TrainerModel(BaseModel):
    max_epochs: int = 1
    accelerator: str = "auto"
    devices: int = 1  # -1 for using all available GPUs with DDP
    num_nodes: int = 1
    precision: int = 32
    log_every_n_steps: int = 2
    deterministic: bool = False
    benchmark: Union[bool, None] = None
    enable_checkpointing: bool = True
    fast_dev_run: bool = False
    enable_progress_bar: bool = True
    profiler: Union[str, None] = None  # e.g. "simple"
    reload_dataloaders_every_n_epochs: int = 0


class CallbackModel(BaseModel):
    async_logging: bool = False
    verbose: bool = False
    save_model: bool = False
