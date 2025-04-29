import logging
from typing import Union

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from data.dali_loader.dali_wav_loader import DaliAudioPipeline

logger = logging.getLogger(__name__)


class SuperDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasets: list[torch.utils.data.Dataset],
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
        train_split: bool = False,
        prepare_functions: list[callable] = [],
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

        self.files_name = files_name
        self.label_names = label_names
        self.datasets = datasets
        self.train_split = train_split

        self.prepare_functions = prepare_functions

        if self.use_dali:
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
            self.data_loader_settings = {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": True,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
            }

        self.prepared = False

        pass

    def _get_dataset_meta_attribute(self, dataset, attribute):
        # Subdataset
        if hasattr(dataset, "dataset"):
            return dataset.dataset.meta.loc[dataset.indices][attribute].tolist()
        else:
            return dataset.meta[attribute].tolist()

    def _get_loader(
        self, pipeline_class, dataset: torch.utils.data.Dataset, shuffle=True
    ):
        files = []
        labels = [[] for _ in self.label_names]
        # Concatenated dataset
        if hasattr(dataset, "datasets"):
            for dataset in dataset.datasets:
                files.extend(self._get_dataset_meta_attribute(dataset, self.files_name))
                for idx, label_name in enumerate(self.label_names):
                    labels[idx].extend(
                        self._get_dataset_meta_attribute(dataset, label_name)
                    )
        else:
            files = self._get_dataset_meta_attribute(dataset, self.files_name)
            labels = [
                self._get_dataset_meta_attribute(dataset, label_name)
                for label_name in self.label_names
            ]

        return pipeline_class(
            files=files,
            labels=labels,
            shuffle=shuffle,
            **self.data_loader_settings,
        )

    def _create_dataloader(self, datasets, train=True, test=False):
        if isinstance(datasets, list):
            if hasattr(datasets[0], "dataset"):
                sample_file_path = datasets[0].dataset.meta[self.files_name].iloc[0]
            else:
                sample_file_path = datasets[0].meta[self.files_name].iloc[0]
        else:
            sample_file_path = datasets.meta[self.files_name].iloc[0]

        if self.label_names is None:
            assert ValueError("label_names not set")

        if sample_file_path.endswith(".wav"):
            pipeline_class = DaliAudioPipeline
        else:
            logger.error(f"Unknown file format: {sample_file_path}")
            return None

        if test:
            if self.use_dali:
                return {
                    f"Dataset {idx}": self._get_loader(
                        pipeline_class, dataset, shuffle=False
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
                    self._get_loader(pipeline_class, dataset, shuffle=train)
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
                return self._get_loader(pipeline_class, dataset_concat, shuffle=train)
            else:
                return DataLoader(
                    dataset_concat,
                    shuffle=train,
                    **self.data_loader_settings,
                )

    def prepare_data(self):
        if self.prepared:
            # logger.warning("prepare_data already called. Skipping prepare_data step.")
            return

        for idx, dataset in enumerate(self.datasets):
            # TODO: Implement data preparation
            # Check if dataset has prepare_data method
            if hasattr(dataset, "prepare_data"):
                dataset.prepare_data()

        for prepare_function in self.prepare_functions:
            prepare_function()

        self.prepared = True

    def setup(self, stage: str):
        if stage == "fit":
            self.train_datasets = []
            self.valid_datasets = []

            for idx, dataset in enumerate(self.datasets):
                if self.train_split:
                    size_train = int(len(dataset) * 0.9)
                    size_valid = len(dataset) - size_train
                    dataset_train, dataset_val = random_split(
                        dataset, [size_train, size_valid]
                    )
                elif "split" in dataset.meta:
                    # Get idcs for train and validation set
                    train_indices = dataset.meta.index[
                        dataset.meta["split"] == "train"
                    ].tolist()
                    dataset_train = torch.utils.data.Subset(dataset, train_indices)

                    valid_indices = dataset.meta.index[
                        dataset.meta["split"] == "validation"
                    ].tolist()
                    dataset_val = torch.utils.data.Subset(dataset, valid_indices)
                else:
                    assert ValueError("No split information found")

                self.train_datasets.append(dataset_train)
                self.valid_datasets.append(dataset_val)

            self.train_loader = self._create_dataloader(
                self.train_datasets, train=True, test=False
            )
            self.val_loader = self._create_dataloader(
                self.valid_datasets, train=False, test=False
            )

        elif stage == "test":
            self.test_datasets = []
            for idx, dataset in enumerate(self.datasets):
                if "split" in dataset.meta:
                    if "dataset" in dataset.meta:
                        for dataset_name in dataset.meta["dataset"].unique():
                            test_indices = dataset.meta.index[
                                (dataset.meta["split"] == "test")
                                & (dataset.meta["dataset"] == dataset_name)
                            ].tolist()
                            self.test_datasets.append(
                                torch.utils.data.Subset(dataset, test_indices)
                            )
                    else:
                        test_indices = dataset.meta.index[
                            dataset.meta["split"] == "test"
                        ].tolist()

                        self.test_datasets.append(
                            torch.utils.data.Subset(dataset, test_indices)
                        )
                else:
                    # TODO: What to do if no split information is available
                    logger.warning("No split information found. Using full dataset")
                    self.test_datasets.append(dataset)

            self.test_loader = self._create_dataloader(
                self.test_datasets, train=False, test=True
            )
        elif stage == "predict":
            assert NotImplementedError("Not implemented yet")
        else:
            assert ValueError(f"Unknown stage: {stage}")

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
