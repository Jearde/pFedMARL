import logging
from pathlib import Path
from typing import Union

import lightning as L
import torch
from torch.utils.data import DataLoader

from data.dali_loader.dali_wav_loader import DaliAudioPipeline
from src.data.dali_loader.dali_numpy_external_loader import DaliNumpyExternalPipeline

logger = logging.getLogger(__name__)


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        meta_dir: str,
        batch_size: int = 32,
        separate_train_sets: bool = False,
        num_workers: int = 0,
        prefetch_factor: Union[int, None] = None,
        download_data: bool = False,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        target_sr: int = 16000,
        target_audio_length: float = 10,
        mono: bool = True,
        audio_slice_length: float = None,
        prepare_data_per_node: bool = False,
        use_dali: bool = True,
        files_name: str = "file",
        label_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()

        # Get cpu count
        self.num_workers = (
            num_workers
            if num_workers is not None and num_workers != -1
            else torch.multiprocessing.cpu_count() - 1
        )
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        self.persistent_workers = True if self.num_workers > 0 else False
        self.batch_size = batch_size
        self.separate_train_sets = separate_train_sets
        self.download_data = download_data

        self.use_dali = use_dali
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.data_dir = Path(data_dir)
        self.meta_dir = Path(meta_dir)

        self.files_name = files_name
        self.label_names = label_names
        self.datasets = []

        # Settings for AudiosetDataset
        self.datasets.append(
            {
                "data_dir": self.data_dir,
                "meta_dir": self.meta_dir,
                "perform_checks": True,
            }
        )

        # Settings for WavDataset
        self.dataset_settings = {
            "target_sr": target_sr,
            "target_length": target_audio_length,
            "mono": mono,
            "random_slice": False,
            "audio_slice_length": audio_slice_length,
            "audio_slice_overlap": None,
            "check_silence": False,
        }

        if self.use_dali:
            # self.data_loader_class = DaliAudioPipeline
            self.data_loader_settings = {
                "batch_size": self.batch_size,
                "num_threads": self.num_workers,
                "prefetch_factor": self.prefetch_factor,
                "local_rank": self.local_rank,
                "global_rank": self.global_rank,
                "world_size": self.world_size,
                "target_sr": target_sr,
                "target_length": target_audio_length,
                "mono": mono,
                "random_crop_size": audio_slice_length,
            }
        else:
            # self.data_loader_class = DataLoader
            self.data_loader_settings = {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": True,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
            }

        pass

    def _create_dataloader(self, datasets, train=True, test=False):
        if isinstance(datasets, list):
            dataset = datasets[0]
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
        else:
            dataset = datasets

        if self.label_names is None:
            assert ValueError("label_names not set")

        if dataset.meta["file"].iloc[0].endswith(".wav"):
            pipeline_class = DaliAudioPipeline
        elif dataset.meta["file"].iloc[0].endswith(".npy"):
            pipeline_class = DaliNumpyExternalPipeline
        else:
            logger.error(f"Unknown file format: {datasets[0].meta['file'][0]}")
            return None

        if test:
            if self.use_dali:
                return {
                    f"Dataset {idx}": pipeline_class(
                        files=dataset.meta[self.files_name].tolist(),
                        labels=[
                            dataset.meta[label_name].tolist()
                            for label_name in self.label_names
                        ],
                        shuffle=False,
                        **self.data_loader_settings,
                    )
                    for idx, dataset in enumerate(datasets)
                }
            else:
                return {
                    f"Dataset {idx}": DataLoader(
                        dataset,
                        shuffle=False,
                        **self.data_loader_settings,
                    )
                    for idx, dataset in enumerate(datasets)
                }

        if self.separate_train_sets:
            if self.use_dali:
                return [
                    # f"{self.datasets[idx]['machine_type']}_{','.join(self.datasets[idx]['section_ids'])}":
                    pipeline_class(
                        files=dataset.dataset.meta.iloc[dataset.indices][
                            self.files_name
                        ].tolist(),
                        labels=[
                            dataset.dataset.meta.iloc[dataset.indices][
                                label_name
                            ].tolist()
                            for label_name in self.label_names
                        ],
                        shuffle=train,
                        **self.data_loader_settings,
                    )
                    for idx, dataset in enumerate(datasets)
                ]
            else:
                return [
                    DataLoader(
                        dataset,
                        shuffle=train,
                        **self.data_loader_settings,
                    )
                    for dataset in self.datasets
                ]
        else:
            dataset_concat = torch.utils.data.ConcatDataset(datasets)

            if self.use_dali:
                files = []
                labels = [[] for _ in self.label_names]
                for dataset in dataset_concat.datasets:
                    files.extend(
                        dataset.dataset.meta.iloc[dataset.indices][
                            self.files_name
                        ].tolist()
                    )
                    for idx, label_name in enumerate(self.label_names):
                        labels[idx].extend(
                            dataset.dataset.meta.iloc[dataset.indices][
                                label_name
                            ].tolist()
                        )

                return pipeline_class(
                    files=files,
                    labels=labels,
                    shuffle=train,
                    **self.data_loader_settings,
                )
            else:
                return DataLoader(
                    dataset_concat,
                    shuffle=train,
                    **self.data_loader_settings,
                )

    def prepare_data(self):
        assert NotImplementedError("Not implemented yet")

    def setup(self, stage: str):
        assert NotImplementedError("Not implemented yet")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.predict_loader

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
