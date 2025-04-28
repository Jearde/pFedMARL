from enum import Enum
from pathlib import Path

import lightning as L
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from config.data_loader_model import DATA_FOLDER, DataLoaderModel
from data.dcase_task_2.dcase_multi_dataset import DCASEMultiDataset
from data.dcase_task_2.dcase_pydentic import DCASEModel, MachineModel, MachineType


class DatasetType(str, Enum):
    DCASEModel = "DCASEMultiDataset"


class TrainType(str, Enum):
    all = "all"  # Will be split into train, validation, test
    train = "train"
    validation = "validation"
    fit = "fit"  # Will be split into train, validation
    test = "test"
    prediction = "prediction"
    keep = "keep"


DATASET = DatasetType.DCASEModel


class DatasetModel(BaseModel):
    dataset_class: DatasetType = DATASET
    dataset_params: DCASEModel = Field(
        DCASEModel() if DATASET == DatasetType.DCASEModel else None
    )
    split_type: TrainType = TrainType.all

    @field_validator("dataset_params", mode="before")
    @classmethod
    def validate_dataset_params(cls, v, info: ValidationInfo):
        dataset_class = info.data.get("dataset_class")
        if isinstance(dataset_class, DatasetType):
            dataset_class = get_class_from_str(dataset_class)
        if dataset_class == DCASEMultiDataset:
            return DCASEModel(**v) if isinstance(v, dict) else v
        else:
            raise ValueError(f"Unsupported dataset: {dataset_class}")

    # @field_validator("dataset_class", mode="after")
    # @classmethod
    # def transform(cls, raw: str) -> L.LightningDataModule:
    #     return get_class_from_str(raw)

    def dataset_class_object(self) -> L.LightningDataModule:
        return get_class_from_str(self.dataset_class)


def get_class_from_str(class_name: str):
    return globals()[class_name]


class MergeLabelsModel(BaseModel):
    columns: list[str] = ["class"]
    logits: bool = False


class SuperDatasetModel(BaseModel):
    label_names: list[str] = ["machine", "target", "anomaly"]
    merge_labels: dict[str, MergeLabelsModel] = {
        "class": MergeLabelsModel(columns=["machine"]),
        "anomaly": MergeLabelsModel(columns=["y_true"]),
    }
    create_dataset: bool = False


class PreProcessingModel(BaseModel):
    use_preprocessing: bool = False
    save_dir: Path = DATA_FOLDER / Path("PreProcessed")


class DatasetModels(BaseModel):
    datasets: list[DatasetModel] = [DatasetModel()]
    datasets: list[DatasetModel] = [
        # DatasetModel(type=TrainType.fit),
        DatasetModel(
            dataset_params=DCASEModel(
                machines=[
                    MachineModel(name=MachineType.ToyCar, data_type="dev"),
                    MachineModel(name=MachineType.ToyTrain, data_type="dev"),
                    MachineModel(name=MachineType.bearing, data_type="dev"),
                    MachineModel(name=MachineType.fan, data_type="dev"),
                    MachineModel(name=MachineType.gearbox, data_type="dev"),
                    MachineModel(name=MachineType.slider, data_type="dev"),
                    MachineModel(name=MachineType.valve, data_type="dev"),
                    # # Evaluation
                    # MachineModel(name=MachineType.ToyDrone, data_type="eval"),
                    # MachineModel(name=MachineType.ToyNscale, data_type="eval"),
                    # MachineModel(name=MachineType.ToyTank, data_type="eval"),
                    # MachineModel(name=MachineType.Vacuum, data_type="eval"),
                    # MachineModel(name=MachineType.bandsaw, data_type="eval"),
                    # MachineModel(name=MachineType.grinder, data_type="eval"),
                    # MachineModel(name=MachineType.shaker, data_type="eval"),
                    # MachineModel(
                    #     name=MachineType.Printer3D, year=2024, data_type="eval"
                    # ),
                    # MachineModel(
                    #     name=MachineType.AirCompressor, year=2024, data_type="eval"
                    # ),
                    # MachineModel(name=MachineType.Scanner, year=2024, data_type="eval"),
                    # MachineModel(
                    #     name=MachineType.ToyCircuit, year=2024, data_type="eval"
                    # ),
                    # MachineModel(
                    #     name=MachineType.HoveringDrone, year=2024, data_type="eval"
                    # ),
                    # MachineModel(
                    #     name=MachineType.HairDryer, year=2024, data_type="eval"
                    # ),
                ],
                train=True,
            ),
            split_type=TrainType.fit,
        ),
        DatasetModel(
            dataset_params=DCASEModel(
                machines=[
                    MachineModel(name=MachineType.ToyCar, data_type="dev"),
                    MachineModel(name=MachineType.ToyTrain, data_type="dev"),
                    MachineModel(name=MachineType.bearing, data_type="dev"),
                    MachineModel(name=MachineType.fan, data_type="dev"),
                    MachineModel(name=MachineType.gearbox, data_type="dev"),
                    MachineModel(name=MachineType.slider, data_type="dev"),
                    MachineModel(name=MachineType.valve, data_type="dev"),
                    # Evaluation
                    # MachineModel(name=MachineType.ToyDrone, data_type="eval"),
                    # MachineModel(name=MachineType.ToyNscale, data_type="eval"),
                    # MachineModel(name=MachineType.ToyTank, data_type="eval"),
                    # MachineModel(name=MachineType.Vacuum, data_type="eval"),
                    # MachineModel(name=MachineType.bandsaw, data_type="eval"),
                    # MachineModel(name=MachineType.grinder, data_type="eval"),
                    # MachineModel(name=MachineType.shaker, data_type="eval"),
                    # MachineModel(
                    #     name=MachineType.Printer3D, year=2024, data_type="eval"
                    # ),
                    # MachineModel(
                    #     name=MachineType.AirCompressor, year=2024, data_type="eval"
                    # ),
                    # MachineModel(name=MachineType.Scanner, year=2024, data_type="eval"),
                    # MachineModel(
                    #     name=MachineType.ToyCircuit, year=2024, data_type="eval"
                    # ),
                    # MachineModel(
                    #     name=MachineType.HoveringDrone, year=2024, data_type="eval"
                    # ),
                    # MachineModel(
                    #     name=MachineType.HairDryer, year=2024, data_type="eval"
                    # ),
                ],
                train=False,
            ),
            split_type=TrainType.test,
        ),
        # DatasetModel(
        #     dataset_class=DatasetType.SONYCModel,
        #     dataset_params=SONYCModel(type=None),
        #     split_type=TrainType.keep,
        # ),
    ]
    data_loader_settings: DataLoaderModel = DataLoaderModel()

    super_dataset: SuperDatasetModel = SuperDatasetModel()
