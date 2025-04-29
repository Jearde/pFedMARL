from pathlib import Path
from typing import Union

from pydantic import BaseModel


def get_mount_folder(folder: Path) -> Path:
    # check, if folder is present
    if not folder.exists() and Path("/mnt/nfs").exists():
        folder = Path("/mnt/nfs")

    return folder


DATA_FOLDER = get_mount_folder(Path("/mnt/data")) / Path("data")
LOG_FOLDER = get_mount_folder(Path(__file__).parents[2]) / Path(".logs")


class DataLoaderModel(BaseModel):
    num_workers: Union[int, None] = 0  # 0, null, -1
    prefetch_factor: Union[int, None] = None
    prepare_data_per_node: bool = False
