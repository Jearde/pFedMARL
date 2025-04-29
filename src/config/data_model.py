from enum import Enum

import lightning as L
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from config.data_loader_model import DataLoaderModel
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
                ],
                train=False,
            ),
            split_type=TrainType.test,
        ),
    ]
    data_loader_settings: DataLoaderModel = DataLoaderModel()

    super_dataset: SuperDatasetModel = SuperDatasetModel()
