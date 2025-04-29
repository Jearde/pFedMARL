import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch

from config.config_model import Config
from config.data_model import TrainType
from data.super_dataset.sub_data_module import SuperDataModule
from data.super_dataset.super_dataset import SuperDataset

logger = logging.getLogger("lightning.pytorch")


class DataCreator:
    def __init__(
        self,
        config: Config,
        ddp_settings: dict | None = None,
    ):
        self.config = config

        self.ddp_settings = (
            ddp_settings
            if ddp_settings is not None
            else {
                "world_size": 1,
                "local_rank": 0,
                "global_rank": 0,
            }
        )

        logger.info("Initializing Data Set")

        self.datasets: list[
            torch.utils.data.Dataset | torch.utils.data.ConcatDataset
        ] = []

        for dataset_config in self.config.data.datasets:
            dataset = dataset_config.dataset_class_object()(
                **dataset_config.dataset_params.model_dump(),
            )

            n_datasets = len(self.datasets)

            if hasattr(dataset, "meta"):
                self.datasets.append(dataset)
            elif hasattr(dataset, "datasets"):
                self.datasets.extend(dataset.datasets)
            else:
                raise ValueError("Invalid dataset")

            for ds in self.datasets[n_datasets:]:
                ds.meta["split"] = (
                    dataset_config.split_type.value
                    if dataset_config.split_type is not (None or TrainType.keep)
                    else ds.meta["split"]
                )
                # Unifie validation names
                ds.meta["split"] = ds.meta["split"].apply(
                    lambda x: TrainType.validation.value if "val" in x else x
                )

        self.dataset_names: dict[str, list[int]] = {}
        for idx, dataset in enumerate(self.datasets):
            if dataset.__class__.__name__ not in self.dataset_names.keys():
                self.dataset_names[dataset.__class__.__name__] = [idx]
            else:
                self.dataset_names[dataset.__class__.__name__].append(idx)

        self.dataset_metadata: dict[str, pd.DataFrame] = {}
        for dataset_name in self.dataset_names:
            self.dataset_metadata[dataset_name] = pd.concat(
                [self.datasets[idx].meta for idx in self.dataset_names[dataset_name]],
                ignore_index=True,
            )

        self.super_dataset = SuperDataset(
            meta_data_list=self.dataset_metadata,
            **self.config.data.super_dataset.model_dump(),
        )

        self.label_names = self.super_dataset.label_names

    def create_data_module(self, dataset: torch.utils.data.Dataset):
        return SuperDataModule(
            datasets=[dataset],
            label_names=self.super_dataset.label_names,
            batch_size=self.config.network.network_params.batch_size,
            **self.config.feature.audio_params.model_dump(),
            **self.config.data.data_loader_settings.model_dump(),
            **self.ddp_settings,
            prepare_functions=[
                ds.prepare_data for ds in self.datasets if hasattr(ds, "prepare_data")
            ],
        )

    def divide_dataset(
        self,
        num_subsets: int | None = 1,
        key: str | None = None,
        uneven_distribution: float = 0.5,
        multiple_datasets: bool = False,
        cluster_skew: bool = False,
    ):
        assert (
            ValueError("Either num_subsets or key should be provided")
            if (num_subsets is None and key is None)
            else True
        )
        assert (
            ValueError("Both num_subsets and key should not be provided")
            if (num_subsets is not None and key is not None)
            else True
        )

        if num_subsets is not None:
            self.super_dataset.divide_dataset(
                num_subsets,
                key=key,
                multiple_datasets=multiple_datasets,
                uneven_distribution=uneven_distribution,
                cluster_skew=cluster_skew,
            )
        elif key is not None:
            num_subsets = self.super_dataset.divide_dataset_by_key(key=key)

        return num_subsets

    def get_subset(self, subset_idx: int):
        return self.create_data_module(
            self.super_dataset.create_subset_dataset(subset_idx)
        )

    def get_overall_data_module(self):
        return self.create_data_module(self.super_dataset)

    def plot_class_distribution(
        self,
        client_key: str = "assignment",
        label_key: str = "machine",
        figsize=(10, 6),
        scale_factor: float = 1000,
    ):
        df_count = (
            self.super_dataset.meta.groupby([client_key, label_key])
            .size()
            .reset_index(name="count")
        )

        scale_factor = scale_factor / df_count["count"].max()

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            df_count["assignment"],
            df_count["machine"],
            s=df_count["count"] * scale_factor,
            c="blue",
            alpha=0.7,
            label="Count",
        )
        ax.set_xlabel("Client")
        ax.set_ylabel(label_key.capitalize())
        ax.set_title("Class Distribution")
        ax.grid(True)
        ax.set_xticks(df_count["assignment"].unique())
        ax.set_yticks(df_count["machine"].unique())
        ax.set_xticklabels(df_count["assignment"].unique().astype(int))
        ax.set_yticklabels(df_count["machine"].unique())
        # ax.legend(
        #     markerscale=0.5,
        # )
        ax.set_aspect("equal")
        fig.tight_layout()
        # plt.show()

        return fig
