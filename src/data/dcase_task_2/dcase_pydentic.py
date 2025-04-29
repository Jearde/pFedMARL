from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from config.data_loader_model import DATA_FOLDER


class MachineType(str, Enum):
    ToyCar = "ToyCar"
    ToyTrain = "ToyTrain"
    bearing = "bearing"
    fan = "fan"
    gearbox = "gearbox"
    slider = "slider"
    valve = "valve"

    ToyDrone = "ToyDrone"
    ToyNscale = "ToyNscale"
    ToyTank = "ToyTank"
    Vacuum = "Vacuum"
    bandsaw = "bandsaw"
    grinder = "grinder"
    shaker = "shaker"

    Printer3D = "3DPrinter"
    AirCompressor = "AirCompressor"
    Scanner = "Scanner"
    ToyCircuit = "ToyCircuit"
    HoveringDrone = "HoveringDrone"
    HairDryer = "HairDryer"
    ToothBrush = "ToothBrush"
    RoboticArm = "RoboticArm"
    BrushlessMotor = "BrushlessMotor"


class MachineModel(BaseModel):
    name: MachineType
    sections: list[int] = [0]
    year: int = 2023
    data_type: str = "dev"  # dev, eval


class DCASEModel(BaseModel):
    task: str = "T2"
    machines: list[MachineModel] = [
        # Development
        MachineModel(name=MachineType.ToyCar),
        MachineModel(name=MachineType.ToyTrain),
        MachineModel(name=MachineType.bearing),
        MachineModel(name=MachineType.fan),
        MachineModel(name=MachineType.gearbox),  # Good one for testing in development
        MachineModel(name=MachineType.slider),  # Good one for testing in development
        MachineModel(name=MachineType.valve),
        # 2023 Evaluation
        MachineModel(name=MachineType.ToyDrone, data_type="eval"),
        MachineModel(name=MachineType.ToyNscale, data_type="eval"),
        MachineModel(name=MachineType.ToyTank, data_type="eval"),
        MachineModel(name=MachineType.Vacuum, data_type="eval"),
        MachineModel(
            name=MachineType.bandsaw, data_type="eval"
        ),  # Good one for testing in development
        MachineModel(
            name=MachineType.grinder, data_type="eval"
        ),  # Good one for testing in development
        MachineModel(name=MachineType.shaker, data_type="eval"),
        # 2024 Evaluation
        MachineModel(name=MachineType.Printer3D, year=2024, data_type="eval"),
        MachineModel(name=MachineType.AirCompressor, year=2024, data_type="eval"),
        MachineModel(name=MachineType.Scanner, year=2024, data_type="eval"),
        MachineModel(name=MachineType.ToyCircuit, year=2024, data_type="eval"),
        MachineModel(name=MachineType.HoveringDrone, year=2024, data_type="eval"),
        MachineModel(name=MachineType.HairDryer, year=2024, data_type="eval"),
        # MachineModel(
        #     name=MachineType.BrushlessMotor, year=2024, data_type="eval"
        # ),  # Causes Errors
    ]
    root_dir: Path = DATA_FOLDER / Path("DCASE2024")
    auto_download: bool = True
    train: bool = True
