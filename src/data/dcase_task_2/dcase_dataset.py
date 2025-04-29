from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from data.dcase_task_2 import loader_common as com
from data.dcase_task_2.dcase_pydentic import MachineType
from data.wav_dataset.torch_wav_dataset import WAVDataset


class Mean(nn.Module):
    def forward(self, input):
        return torch.mean(input, dim=0)


class DCASEAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        dataset_name: str,
        section_keyword,
        machine_type: str,
        section_ids: list[str],
        data_type: str,
        train: bool = True,
        **kwargs,
    ):
        self.root_dir = Path(root) if isinstance(root, str) else root
        self.machine_type = machine_type
        self.train = train
        self.dataset_name = dataset_name
        self.data_type = data_type

        self.machine_index = list(MachineType).index(self.machine_type)

        data_path = self.root_dir / dataset_name.lower() / f"{data_type}_data"
        self.target_dir = data_path / "raw" / machine_type
        self.dir_name = "train" if train else "test"
        self.mode = data_type == "dev"

        self.transforms = None

        # get section names from wave file names
        section_names = [
            f"{section_keyword}_{section_id}" for section_id in section_ids
        ]
        unique_section_names = np.unique(section_names)
        n_sections = len(unique_section_names)

        file_list = []
        self.meta = pd.DataFrame(
            columns=["y_true", "condition", "basename", "target", "machine"]
        )

        for section_idx, section_name in enumerate(unique_section_names):
            files, y_true, condition = com.file_list_generator(
                target_dir=str(self.target_dir),
                section_name=section_name,
                unique_section_names=unique_section_names,
                dir_name=self.dir_name,
                mode=self.mode,
                train=train,
            )
            file_list.extend([str(file) for file in files])

            # Add meta information
            self.meta = pd.concat(
                [
                    self.meta,
                    pd.DataFrame(
                        {
                            "y_true": [int(y) for y in y_true],
                            "condition": [c.tolist() for c in condition],
                            "basename": [Path(file).name for file in files],
                            "target": [1 if "target" in file else 0 for file in files],
                            "machine": [self.machine_index] * len(files),
                            "file": [str(file) for file in files],
                        }
                    ),
                ]
            )

        self.wav_dataset = WAVDataset(
            wav_files=file_list,
            transforms=self.transforms,
            **kwargs,
        )

        if len(self.wav_dataset) > len(file_list):
            # Extend meta to match the length of the dataset
            self.meta = self.meta.iloc[
                np.repeat(np.arange(len(self.meta)), self.wav_dataset.ratio_lists)
            ]

    @classmethod
    def download(
        cls,
        root: Union[str, Path],
        dataset_name: str,
        machine_type: str,
        data_type: str,
        train: bool = True,
        **kwargs,
    ):
        root = Path(root) if isinstance(root, str) else root
        data_path = root / dataset_name.lower() / f"{data_type}_data"
        target_dir = data_path / "raw" / machine_type
        dir_name = "train" if train else "test"

        com.download_raw_data(
            target_dir=target_dir,
            dir_name=dir_name,
            machine_type=machine_type,
            data_type=data_type,
            dataset=dataset_name,
            root=root,
        )

    def get_sample_data(self, idx):
        feature, label, meta = self.__getitem__(idx)

        basename = meta["basename"]

        return feature, feature, label, basename

    def __len__(self):
        assert len(self.wav_dataset) == len(self.meta)

        return len(self.wav_dataset)

    def __getitem__(self, idx):
        return self.wav_dataset[idx], torch.tensor(
            [
                self.meta.iloc[idx]["y_true"],
                self.meta.iloc[idx]["machine"],
                self.meta.iloc[idx]["target"],
                # self.meta.iloc[idx]['condition']
                idx,
            ],
            dtype=torch.int64,
        )
