from pathlib import Path

import torch

from data.dcase_task_2.dcase_dataset import DCASEAudioDataset
from data.dcase_task_2.dcase_pydentic import MachineModel


class DCASEMultiDataset(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        root_dir: str | Path,
        machines: list[dict | MachineModel],
        task: str = "T2",
        section_keyword: str = "section",
        train=True,
        auto_download=True,
        **kwargs,
    ):
        self.data_dir = Path(root_dir)

        datasets = []

        for machine in machines:
            machine_model = (
                MachineModel(**machine) if isinstance(machine, dict) else machine
            )

            dataset_name = f"DCASE{machine_model.year}{task}"

            # Sections from int to str with leading zeros
            section_ids = [f"{section:02d}" for section in machine_model.sections]

            if auto_download:
                DCASEAudioDataset.download(
                    root=self.data_dir,
                    dataset_name=dataset_name,
                    machine_type=machine_model.name.value,
                    data_type=machine_model.data_type,
                    train=train,
                )

            datasets.append(
                DCASEAudioDataset(
                    root=self.data_dir,
                    dataset_name=dataset_name,
                    section_keyword=section_keyword,
                    machine_type=machine_model.name.value,
                    section_ids=section_ids,
                    data_type=machine_model.data_type,
                    train=train,
                )
            )

        super().__init__(datasets)

        pass
