import random
from typing import Iterable, List

import numpy as np
import nvidia.dali.fn as fn
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class ExternalInputIterator(object):
    def __init__(self, files, num_shards, shard_id, shuffle):
        self.files = list(enumerate(files))
        self.shuffle = shuffle
        self.data_set_len = len(self.files)

        self.files = self.files[
            self.data_set_len * shard_id // num_shards : self.data_set_len
            * (shard_id + 1)
            // num_shards
        ]

        self.n = len(self.files)
        random.seed(42)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            random.shuffle(self.files)
        return self

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            # raise StopIteration

        # batch = []
        # labels = []
        # for _ in range(self.batch_size):
        #     jpeg_filename, label = self.files[self.i].split(" ")
        #     f = open(self.images_dir + jpeg_filename, "rb")
        #     batch.append(np.frombuffer(f.read(), dtype=np.uint8))
        #     labels.append(np.array([label], dtype=np.uint8))
        #     self.i = (self.i + 1) % self.n

        label, file = self.files[self.i]
        data = np.load(file)
        # data = torch.from_numpy(np.load(file)).unsqueeze(0)
        label = torch.tensor([label])
        self.i += 1
        return (data, label)


class PyTorchIterator(DALIGenericIterator):
    def __init__(self, labels: list[list[int]], device, *kargs, **kvargs):
        """Overloading of DALIGenericIterator for Multi-Label

        This is for supporting multiple labels per file. The index of the file is used to get the value in the list.

        Args:
            labels (List[List[int]]): _description_
        """
        super().__init__(*kargs, **kvargs)
        self.labels = [torch.tensor(label).squeeze(-1).to(device) for label in labels]

    def __next__(self):
        out = super().__next__()

        # Use only the output of the first pipeline (not suitable for multiple pipelines in parallel)
        out = out[0]

        labels = [
            self.labels[i][out["label"]].squeeze(1) for i in range(len(self.labels))
        ]

        # Get the label based on the index
        return out["feature"], labels


@pipeline_def
def external_source_pipeline(
    external_source: Iterable,
    device: str = "cpu",
):
    """Load audio files and return audio and label

    Args:
        files (List[str]): List of file paths
        target_sr (int, optional): Target sample rate. Defaults to 16000.
        target_length (int, optional): Target length in seconds. Defaults to 10.
        shuffle (bool, optional): Shuffle the dataset. Defaults to False.
        device_id (int, optional): Device ID of the GPU. Also known as local rank in DDP. Defaults to -1.
        shard_id (int, optional): Shard ID. Also known as global rank in DDP. Defaults to 0.
        num_shards (int, optional): Number of shards. Also known as world size in DDP. Defaults to 1.
        py_num_workers(int, optional): Number of Python workers for data loading via fn.external_source(). Defaults to 1.
    """
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.html

    data, label = fn.external_source(
        source=external_source,
        num_outputs=2,
        # name="Reader",
        device="cpu",
        batch=False,
        parallel=False,
        cycle="quiet",  # ensures the source cycles indefinitely
    )
    # data = fn.cast(data, dtype=types.FLOAT)
    # data = fn.copy(data, device=device)

    # label = fn.cast(label, dtype=types.INT32)
    # label = fn.copy(label, device=device)

    return data, label


def DaliNumpyExternalPipeline(
    files: List[str],
    labels: List[List[int]],
    batch_size: int,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    direct_store: bool = False,
    py_num_workers: int = -1,
    **kwargs,
):
    num_threads = num_threads if num_threads > 0 else torch.multiprocessing.cpu_count()
    py_num_workers = (
        py_num_workers if py_num_workers > 0 else torch.multiprocessing.cpu_count()
    )
    prefetch_factor = prefetch_factor if prefetch_factor is not None else 2
    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = "gpu" if device_id >= 0 else "cpu"

    external_source = ExternalInputIterator(
        files=files,
        num_shards=num_shards,
        shard_id=shard_id,
        shuffle=shuffle,
    )

    pipeline = external_source_pipeline(
        external_source=external_source,
        batch_size=batch_size,
        num_threads=num_threads,
        device=device,
        device_id=device_id,
        prefetch_queue_depth=prefetch_factor,
        # py_num_workers=py_num_workers,
        # py_start_method="spawn",
        # **kwargs,
    )

    # pipeline.start_py_workers()

    pipeline.build()

    device = f"cuda:{device_id}" if device_id >= 0 else "cpu"
    data_loader = PyTorchIterator(
        pipelines=[pipeline],
        labels=labels,
        device=device,
        output_map=["feature", "label"],
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        # last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
        # reader_name="Reader",
        size=len(files),
        prepare_first_batch=True,
    )

    return data_loader
