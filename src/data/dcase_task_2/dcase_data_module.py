import logging
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms  # type: ignore

from data.dali_loader.dali_wav_loader import DaliAudioPipeline
from data.dcase_task_2.dcase_dataset import DCASEAudioDataset
from data.dcase_task_2.dcase_pydentic import MachineModel
from data.dcase_task_2.loader_common import get_machine_type_dict

logger = logging.getLogger(__name__)

train_sets = [
    "bearing",
    "fan",
    "gearbox",
    "slider",
    "valve",
    "ToyCar",
    "ToyTrain",
]

eval_sets = [
    "bandsaw",
    "grinder",
    "shaker",
    "ToyDrone",
    "ToyNscale",
    "ToyTank",
    "Vacuum",
    "3DPrinter",
    "AirCompressor",
    "Scanner",
    "ToyCircuit",
    "HoveringDrone",
    "HairDryer",
    "ToothBrush",
    "RoboticArm",
    "BrushlessMotor",
]


class DCASE202XT2DataModule(L.LightningDataModule):
    def __init__(
        self,
        year: str,
        machines: list[MachineModel],
        root_dir: str,
        task: str = "T2",
        train_only: bool = False,
        data_set_type: Optional[str] = None,
        machine_index: Optional[Union[int, list[int]]] = None,
        section_index: Optional[Union[int, list[int]]] = None,
        num_workers: Optional[int] = None,
        batch_size: int = 32,
        separate_train_sets: bool = False,
        use_dali: bool = True,
        local_rank=0,
        global_rank=0,
        world_size=1,
        target_sr: int = 16000,
        target_audio_length: float = 10,
        mono: bool = True,
        audio_slice_length: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        # Get cpu count
        self.num_workers = (
            num_workers
            if num_workers is not None and num_workers != -1
            else torch.multiprocessing.cpu_count() - 1
        )
        self.prefetch_factor = 2 if self.num_workers > 0 else None
        self.persistent_workers = True if self.num_workers > 0 else False
        self.batch_size = batch_size

        self.data_dir = Path(root_dir)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.separate_train_sets = separate_train_sets

        current_dir = Path(__file__).resolve().parents[1]

        self.dataset_name = f"DCASE{year}{task}"

        self.datasets = []

        if isinstance(machine_index, int):
            machines = [machines[machine_index]]
        elif isinstance(machine_index, list):
            machines = [machines[idx] for idx in machine_index]

        for machine in machines:
            machine_type = machine["name"]
            data_set_type = None

            if data_set_type is None and machine_type in train_sets:
                data_set_type = "dev"
            elif data_set_type is None and machine_type in eval_sets:
                data_set_type = "eval"

            if data_set_type == "eval":
                data_path = f"{self.data_dir}/eval_data/"
                data_type = "eval"
            elif data_set_type == "dev":
                data_path = f"{self.data_dir}/dev_data/"
                data_type = "dev"

            section_ids = machine["sections"]

            if isinstance(section_index, int):
                section_ids = [section_ids[section_index]]
            elif isinstance(section_index, list):
                section_ids = [section_ids[idx] for idx in section_index]

            for section_id in section_ids:
                # Legacy datasets
                if data_set_type is None:
                    machine_type_dict_tmp = get_machine_type_dict(
                        self.dataset_name, mode="dev"
                    )["machine_type"]
                    data_type = (
                        "dev"
                        if section_id
                        in [
                            int(item)
                            for item in machine_type_dict_tmp[machine_type]["dev"]
                        ]
                        else "eval"
                    )
                    data_path = f"{self.data_dir}/{data_type}_data/"

                data_path = current_dir / data_path

                machine_type_dict = get_machine_type_dict(
                    self.dataset_name, mode=data_type
                )["machine_type"]
                section_id_list = machine_type_dict[machine_type][data_type]
                num_classes = len(section_id_list)
                logger.info("Sections: %d" % (num_classes))
                id_list = [int(machine_id) for machine_id in section_id_list]
                section_keyword = get_machine_type_dict(
                    self.dataset_name, mode=data_type
                )["section_keyword"]

                section_id_list = ["{:02}".format(section_id)]

                self.datasets.append(
                    {
                        "root": self.data_dir,
                        "dataset_name": self.dataset_name,
                        "section_keyword": section_keyword,
                        "machine_type": machine_type.name,
                        # "train": data_type == "dev",
                        "section_ids": section_id_list,
                        "data_type": data_type,
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

        self.use_dali = use_dali
        if self.use_dali:
            # self.data_loader_class = DaliAudioPipeline
            self.data_loader_settings = {
                "batch_size": self.batch_size,
                "num_threads": self.num_workers,
                "prefetch_factor": self.prefetch_factor,
                "local_rank": local_rank,
                "global_rank": global_rank,
                "world_size": world_size,
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

    def _create_dataloader(self, datasets, train=True, test=False, predict=False):
        if test:
            if self.use_dali:
                return {
                    f"{self.datasets[idx]['machine_type']}_{','.join(self.datasets[idx]['section_ids'])}": DaliAudioPipeline(
                        files=dataset.meta["file"].tolist(),
                        labels=[
                            dataset.meta["y_true"].tolist(),
                            dataset.meta["machine"].tolist(),
                            dataset.meta["target"].tolist(),
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

        if predict:
            if self.use_dali:
                self.predict_files = []
                labels = [[], [], []]
                for dataset in self.predict_dataset.datasets:
                    self.predict_files.extend(dataset.meta["file"].tolist())
                    labels[0].extend(dataset.meta["y_true"].tolist())
                    labels[1].extend(dataset.meta["machine"].tolist())
                    labels[2].extend(dataset.meta["target"].tolist())

                return DaliAudioPipeline(
                    files=self.predict_files,
                    labels=labels,
                    shuffle=train,
                    **self.data_loader_settings,
                )
            else:
                return DataLoader(
                    self.predict_dataset,
                    batch_size=self.batch_size,
                    num_workers=2,
                    shuffle=False,
                )

        if self.use_dali:
            if self.separate_train_sets:
                return [
                    DaliAudioPipeline(
                        files=dataset.dataset.meta.iloc[dataset.indices][
                            "file"
                        ].tolist(),
                        labels=[
                            dataset.dataset.meta.iloc[dataset.indices][
                                "y_true"
                            ].tolist(),
                            dataset.dataset.meta.iloc[dataset.indices][
                                "machine"
                            ].tolist(),
                            dataset.dataset.meta.iloc[dataset.indices][
                                "target"
                            ].tolist(),
                        ],
                        shuffle=train,
                        **self.data_loader_settings,
                    )
                    for idx, dataset in enumerate(datasets)
                ]
            else:
                dataset_concat = torch.utils.data.ConcatDataset(datasets)

                files = []
                labels = [[], [], []]
                for dataset in dataset_concat.datasets:
                    files.extend(
                        dataset.dataset.meta.iloc[dataset.indices]["file"].tolist()
                    )
                    labels[0].extend(
                        dataset.dataset.meta.iloc[dataset.indices]["y_true"].tolist()
                    )
                    labels[1].extend(
                        dataset.dataset.meta.iloc[dataset.indices]["machine"].tolist()
                    )
                    labels[2].extend(
                        dataset.dataset.meta.iloc[dataset.indices]["target"].tolist()
                    )

                return DaliAudioPipeline(
                    files=files,
                    labels=labels,
                    shuffle=train,
                    **self.data_loader_settings,
                )
        else:
            if self.separate_train_sets:
                return [
                    DataLoader(
                        dataset,
                        shuffle=train,
                        **self.data_loader_settings,
                    )
                    for idx, dataset in enumerate(datasets)
                ]
            else:
                dataset_concat = torch.utils.data.ConcatDataset(datasets)
                return DataLoader(
                    dataset_concat,
                    shuffle=train,
                    **self.data_loader_settings,
                )

    def prepare_data(self):
        for idx, dataset in enumerate(self.datasets):
            DCASEAudioDataset.download(
                **dataset,
                train=True,
            )

            DCASEAudioDataset.download(
                **dataset,
                train=False,
            )
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_datasets = []
            self.valid_datasets = []

            for idx, dataset in enumerate(self.datasets):
                dataset_full = DCASEAudioDataset(
                    **dataset,
                    train=True,
                )

                size_train = int(len(dataset_full) * 0.9)
                size_valid = len(dataset_full) - size_train
                dataset_train, dataset_val = random_split(
                    dataset_full, [size_train, size_valid]
                )
                self.train_datasets.append(dataset_train)
                self.valid_datasets.append(dataset_val)

            self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)
            self.valid_dataset = torch.utils.data.ConcatDataset(self.valid_datasets)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_datasets = []
            for idx, dataset in enumerate(self.datasets):
                self.test_datasets.append(
                    DCASEAudioDataset(
                        **dataset,
                        train=False,
                    )
                )
            self.test_dataset = torch.utils.data.ConcatDataset(self.test_datasets)

        if stage == "predict":
            self.predict_datasets = []
            for idx, dataset in enumerate(self.datasets):
                self.predict_datasets.append(
                    DCASEAudioDataset(
                        **dataset,
                        train=False,
                    )
                )
            self.predict_dataset = torch.utils.data.ConcatDataset(self.predict_datasets)

    def train_dataloader(self):
        return self._create_dataloader(self.train_datasets, train=True, test=False)

    def val_dataloader(self):
        return self._create_dataloader(self.valid_datasets, train=False, test=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_datasets, train=False, test=True)

    def predict_dataloader(self):
        return self._create_dataloader(self.predict_dataset, train=False, predict=True)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
