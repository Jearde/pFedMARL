from pathlib import Path

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

from config.data_model import DatasetModels
from config.feature_model import FeatureModel
from config.fl_model import FederatedLearningModel
from config.logger_model import LoggerModel
from config.network_model import NetworkModel
from config.rl_model import ReinforcementLearningModel
from config.trainer_model import CallbackModel, TrainerModel


class Config(BaseModel):
    seed: int = 42
    config_path: Path | None = None
    data: DatasetModels = DatasetModels()
    feature: FeatureModel = FeatureModel()
    network: NetworkModel = NetworkModel()
    logger: LoggerModel = LoggerModel()
    callback: CallbackModel = CallbackModel()
    trainer: TrainerModel = TrainerModel()
    fl: FederatedLearningModel = FederatedLearningModel()
    rl: ReinforcementLearningModel = ReinforcementLearningModel()

    def to_yaml(self, **kwargs):
        return to_yaml_str(self, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_str):
        return parse_yaml_raw_as(cls, yaml_str)

    def __repr__(self):
        return self.to_yaml()

    def save(self, path: Path):
        with open(path, "w") as f:
            f.write(self.to_yaml())


def write_example_config(output_path: Path):
    config = Config()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(config.to_yaml())


def parse_config(config_path: Path) -> Config:
    config = parse_yaml_raw_as(Config, config_path.read_text())
    config.config_path = config_path
    return config
