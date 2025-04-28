import logging
from operator import add

import numpy as np
import pandas as pd
import torch

from config.data_model import TrainType
from data.wav_dataset.torch_wav_dataset import WAVDataset

if "lightning.pytorch" in logging.Logger.manager.loggerDict:
    logger = logging.getLogger("lightning.pytorch")
else:
    logger = logging.getLogger(__name__)


def update_list(x, index):
    x[index] = x[index] + 1.0
    return x


def class_to_vector(x, value, label, conversion_dict):
    x = [0.0] * len(conversion_dict)
    if label in conversion_dict:
        if value < 0:
            pass
        x[conversion_dict[label]] += (
            value  # TODO what should I do with lists here, like with "condition" label
        )
    elif isinstance(value, list):
        for i in range(len(value)):
            x[conversion_dict[label + "_" + str(i)]] += value[i]
    else:
        x[conversion_dict[label + "_" + str(float(value))]] += 1.0
    return x


class MetaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        split_ratio: list[float] = [0.85, 0.1, 0.05],
        create_dataset: bool = False,
        train_split: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.meta = meta
        self.create_dataset = create_dataset
        self.split_ratio = split_ratio

        # Set category for training (train 85 %/ val 10% / test 5%) as int 0, 1, 2 and new column "split"
        if train_split:
            assert sum(split_ratio) == 1, "Split ratio must sum to 1"
            self.split_meta(split_ratio)

        if self.create_dataset:
            file_list = self.meta["file"].tolist()
            self.wav_dataset = WAVDataset(
                wav_files=file_list,
                **kwargs,
            )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.create_dataset:
            return self.wav_dataset[idx], [
                torch.tensor(self.meta.iloc[idx][name]) for name in self.label_names
            ]
        else:
            return [
                torch.tensor(self.meta.iloc[idx][name]) for name in self.label_names
            ]

    def split_meta(self, split_ratio: list[float]):
        # Split fit into train and validation
        self.meta.loc[
            self.meta.loc[self.meta["split"] == TrainType.fit.value]
            .sample(frac=split_ratio[1])
            .index,
            "split",
        ] = TrainType.validation.value
        self.meta.loc[self.meta["split"] == TrainType.fit.value, "split"] = (
            TrainType.train.value
        )

        # Split all into train, test and validation
        self.meta.loc[
            self.meta.loc[self.meta["split"] == TrainType.all.value]
            .sample(frac=split_ratio[1])
            .index,
            "split",
        ] = TrainType.validation.value
        self.meta.loc[
            self.meta.loc[self.meta["split"] == TrainType.all.value]
            .sample(frac=split_ratio[2])
            .index,
            "split",
        ] = TrainType.test.value
        self.meta.loc[self.meta["split"] == TrainType.all.value, "split"] = (
            TrainType.train.value
        )


class SuperDataset(MetaDataset):
    def __init__(
        self,
        meta_data_list: dict[str, pd.DataFrame],
        label_names: list[str],
        merge_labels: dict[
            str, dict[str, list[str] | dict[str, dict[str, list[str]]] | bool]
        ] = None,
        **kwargs,
    ):
        self.meta_data_list = meta_data_list
        self.label_names = label_names

        # Create merged meta data
        self.meta = pd.DataFrame()

        for dataset, meta_data in meta_data_list.items():
            meta_data["dataset"] = dataset
            self.meta = pd.concat([self.meta, meta_data], ignore_index=True)

        if merge_labels is not None:
            for label_name, merge_label_names in merge_labels.items():
                logits = merge_label_names["logits"]
                merge_label_names = merge_label_names["columns"]
                if not logits:
                    for dataset in meta_data_list.keys():
                        meta_data_idx = (
                            self.meta.where(self.meta["dataset"] == dataset)["dataset"]
                            .dropna()
                            .index
                        )
                        self.meta.loc[meta_data_idx, label_name] = self.meta.loc[
                            meta_data_idx, merge_label_names
                        ].apply(
                            lambda row: dataset
                            + "_"
                            + "_".join(row.values.astype(str)),
                            axis=1,
                        )
                    # Change unique labels to int
                    self.meta[label_name] = pd.Categorical(self.meta[label_name])
                    self.meta[label_name] = self.meta[label_name].cat.codes.astype(
                        float
                    )

                    logger.info(
                        f"Merge labels: {merge_label_names} to '{label_name}': {self.meta[label_name].unique()}"
                    )
                else:
                    # Create new list with number of unique labels across all datasets
                    conversion_dict = {}
                    for dataset_name, label_names in merge_label_names.items():
                        for label in label_names:
                            column = self.meta.loc[
                                self.meta["dataset"] == dataset_name, label
                            ]
                            if isinstance(column.iloc[0], list) or isinstance(
                                column.iloc[0], np.ndarray
                            ):
                                num_classes = len(column.iloc[0])
                            else:
                                num_classes = column.nunique()

                            if (
                                label not in conversion_dict
                                and num_classes <= 2
                                and not isinstance(column.iloc[0], list)
                            ):
                                conversion_dict[label] = len(conversion_dict)
                            elif isinstance(column.iloc[0], list):
                                for i in range(num_classes):
                                    conversion_dict[label + "_" + str(i)] = len(
                                        conversion_dict
                                    )
                            else:
                                for i in column.unique():
                                    conversion_dict[label + "_" + str(float(i))] = len(
                                        conversion_dict
                                    )
                    self.meta.loc[:, label_name] = np.zeros(
                        (len(self.meta), len(conversion_dict))
                    ).tolist()
                    # Set 1.0 where the label is present
                    for dataset_name in merge_label_names.keys():
                        for label in merge_label_names[dataset_name]:
                            # Convert all labels to single one hot
                            old_logits_vectors = self.meta.loc[
                                self.meta["dataset"] == dataset_name, label_name
                            ]

                            new_logits_vectors = self.meta.loc[
                                self.meta["dataset"] == dataset_name, label
                            ].apply(
                                lambda x: class_to_vector(0, x, label, conversion_dict)
                            )

                            # Element wise addition of lists
                            self.meta.loc[
                                self.meta["dataset"] == dataset_name, label_name
                            ] = old_logits_vectors.combine(
                                new_logits_vectors, lambda x, y: list(map(add, x, y))
                            )

        super().__init__(meta=self.meta, **kwargs)

    def get_single_dataset(self, idx):
        return self.meta_data_list[idx]

    def _make_uneven(self, matrix, uneven_distribution):
        # Adjust dataset_probability to be unevenly distributed across the splits
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] *= np.clip(
                    1 + np.random.uniform(-uneven_distribution, uneven_distribution),
                    a_min=0,
                    a_max=None,
                )
        return matrix

    def _normalize_matrix(self, matrix, axis=0):
        # Normalize the dataset_probability so the columns (dataset distributed over splits) sum to 1
        matrix = matrix / np.sum(matrix, axis=axis)
        return np.nan_to_num(matrix)

    def divide_dataset_by_key(
        self,
        key: str,
    ):
        for split_indices, key_value in enumerate(self.meta[key].unique()):
            self.meta.loc[self.meta[key] == key_value, "assignment"] = split_indices

        return len(self.meta["assignment"].unique())

    def divide_dataset(
        self,
        num_splits: int,
        key: str | None = None,
        multiple_datasets: bool = False,
        uneven_distribution: bool | float = False,
        cluster_skew: bool = True,
    ):
        """Divide the dataset into num_splits number of splits."""

        dataset_names = self.meta["dataset"].unique()

        if multiple_datasets:
            # Assign a probability of how much of each dataset should be in each split

            if not uneven_distribution:
                dataset_probability = np.ones((num_splits, len(dataset_names)))
            elif isinstance(uneven_distribution, bool) and uneven_distribution:
                # The sum of each row should be 1
                dataset_probability = np.random.dirichlet(
                    np.ones(num_splits), size=len(dataset_names)
                )

                # Transpose the matrix so that the rows are the splits
                dataset_probability = dataset_probability.T
            elif isinstance(uneven_distribution, float):
                dataset_probability = np.ones((num_splits, len(dataset_names)))

                dataset_probability = self._make_uneven(
                    dataset_probability, uneven_distribution
                )
                dataset_probability = self._normalize_matrix(
                    dataset_probability, axis=0
                )
        elif key is not None:
            # Assign a probability of how much of each dataset should be in each split
            num_classes = len(self.meta[key].unique())
            dataset_probability = np.ones((num_splits, num_classes))

            if cluster_skew:
                labels_per_cluster = 2
                dataset_probability = np.zeros((num_splits, num_classes))
                cluster_group_1 = num_splits - num_classes // labels_per_cluster
                dataset_probability[0:cluster_group_1][:, 0:labels_per_cluster] = 1

                for i in range(num_classes // labels_per_cluster):
                    dataset_probability[
                        i + cluster_group_1,
                        2 * i + labels_per_cluster : 2 * i
                        + labels_per_cluster
                        + labels_per_cluster,
                    ] = 1

            dataset_probability = self._make_uneven(
                dataset_probability, uneven_distribution
            )

            dataset_probability = self._normalize_matrix(dataset_probability, axis=0)
        else:
            # Assign each split to a single dataset. Splits are distributed evenly across the datasets size

            # Get amount of data for each dataset
            dataset_sizes = [
                len(self.meta.loc[self.meta["dataset"] == ds]) for ds in dataset_names
            ]

            # Assign which split should be assigned to which dataset based on the dataset sizes
            dataset_probability = np.zeros((num_splits, len(dataset_names)))

            # How many splits per column in the dataset_probability matrix should have a 1
            dataset_assignments = np.random.multinomial(
                num_splits, pvals=dataset_sizes / np.sum(dataset_sizes)
            )

            for i in range(len(dataset_assignments)):
                for j in range(dataset_assignments[i]):
                    dataset_probability[j][i] = 1

            if uneven_distribution:
                if isinstance(uneven_distribution, bool):
                    raise ValueError(
                        "Need to specify the uneven distribution factor e.g. 0.1"
                    )

                dataset_probability = self._make_uneven(
                    dataset_probability, uneven_distribution
                )

            dataset_probability = self._normalize_matrix(dataset_probability, axis=0)

        # Shuffle
        if not cluster_skew:
            np.random.shuffle(dataset_probability)

        # Assign the split index to the meta data
        self.assign_subset(dataset_probability, key=key)

    def assign_subset(self, dataset_probability: np.ndarray, key: str | None = None):
        """Assign the dataset to a subset based on the dataset_probability matrix."""
        key = key if key is not None else "dataset"

        num_splits = dataset_probability.shape[0]
        dataset_names = self.meta[key].unique()

        self.meta.loc[:, "assignment"] = -1
        for ds_index, ds in enumerate(dataset_names):
            dataset_indices = self.meta.loc[self.meta[key] == ds].index

            split_probabilities = dataset_probability[:, ds_index]

            # Random split the dataset_indices into num_splits based on the split_probabilities
            split_indices = np.random.choice(
                num_splits, size=len(dataset_indices), p=split_probabilities
            )

            # self.meta["assignment"].loc[self.meta["dataset"] == ds] = split_indices
            self.meta.loc[self.meta[key] == ds, "assignment"] = split_indices

    def create_subset_dataset(self, idx: int, train_split: bool = True):
        """Create a subset dataset based on the assignment column."""
        subset_meta = self.meta.loc[self.meta["assignment"] == idx]

        return MetaDataset(
            meta=subset_meta,
            create_dataset=self.create_dataset,
            split_ratio=self.split_ratio,
            train_split=train_split,
        )

    def get_subset(
        self,
        datasets: list[str],
        class_prob: dict[str, float] | list[float] | None = None,
        meta_filter: dict[str, list[str]] | None = None,
        exclusive: bool = False,
    ):
        """Get a subset of the dataset based on the restrictions.

        Args:
            datasets: list of dataset names that should be included in the subset
            class_prob: dict of class probabilities for each dataset
            meta_filter: dict of filters for each dataset. The key is the column name and the value is the list of values to keep.
            exclusive: if True, only samples are only allowed to be in one subset. If this method is called multiple times with exclusive=True, the subsets will be exclusive to each other.
        """
        raise NotImplementedError
