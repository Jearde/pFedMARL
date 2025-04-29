from typing import List

import lightning as pl
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class PyTorchIterator(DALIGenericIterator):
    def __init__(self, labels: List[List[int]], *kargs, **kvargs):
        """Overloading of DALIGenericIterator for Multi-Label

        This is for supporting multiple labels per file. The index of the file is used to get the value in the list.

        Args:
            labels (List[List[int]]): List of labels for each file. The index needs to be in sync with the file list.
        """
        super().__init__(*kargs, **kvargs)

        self.labels = [torch.tensor(label).squeeze(-1) for label in labels]

    def __next__(self):
        out = super().__next__()

        # Use only the output of the first pipeline (not suitable for multiple pipelines in parallel)
        out = out[0]

        # Concatenate along the last dimension
        labels = [
            self.labels[i][out["label"]].squeeze(1).to(out["audio"].device)
            for i in range(len(self.labels))
        ]

        # Get the label based on the index
        return out["audio"].movedim(-2, -1), labels


@pipeline_def
def wav_data_pipeline(
    files: List[str],
    target_sr: int = 16000,
    target_length: float = 10,
    mono: bool = True,
    shuffle: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    device: str = "cpu",
    dont_use_mmap=True,
    rnd_crop_size: float = None,
    start_sec: float = None,
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
        device (str, optional): Device to use like e.g. cuda. Defaults to 'cpu'.
        rnd_crop_size (float, optional): Length of slice in seconds that is cut out randomly from the audio file. Defaults to None.
        start_sec (float, optional): Start time in seconds to set an offset from where on the audio should be used. Defaults to None.
    """
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.html

    if rnd_crop_size is not None:
        rnd_choice_list = list(
            range(int(target_length * target_sr) - int(rnd_crop_size * target_sr))
        )
        if len(rnd_choice_list) == 0:
            rnd_choice_list = [0]
        start = fn.random.choice(rnd_choice_list, shape=[1])
        end = start + int(rnd_crop_size * target_sr)
    elif start_sec is not None:
        start = int(start_sec * target_sr)
        end = start + int(target_length * target_sr)
    else:
        start = 0
        end = target_length * target_sr

    # Load audio and return index as label
    encoded, label = fn.readers.file(
        files=files,
        labels=list(torch.arange(len(files))),
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        device="cpu",
        seed=42,
        name="Reader",
        dont_use_mmap=dont_use_mmap,  # True for local storage, False for network storage
        read_ahead=True,
        # prefetch_queue_depth=2,
    )

    # Decode audio, downmix reduces the dimension
    audio, sr = fn.decoders.audio(
        encoded,
        dtype=types.FLOAT,
        downmix=False,
        sample_rate=target_sr,
        device="cpu",
    )

    # If not using downmix for reducing the dimension, you can use mean and keep the dimension
    if mono:
        audio = fn.reductions.mean(audio, axes=[-1], keep_dims=True)

    # Copy the audio to the device (gpu)
    audio = fn.copy(audio, device=device)

    # Pad or cut to the target length
    audio = fn.slice(
        audio,
        start=start,
        end=end,
        axes=[0],
        out_of_bounds_policy="pad",
        device=device,
    )

    return audio, label


def DaliAudioPipeline(
    files: List[str],
    labels: List[List[int]],
    batch_size: int,
    target_sr: int = 16000,
    target_length: float = 10,
    mono: bool = True,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    random_crop_size: float = None,
    start_sec: float = None,
    **kwargs,
):
    """Create a PyTorch DataLoader with DALI

    Args:
        files (List[str]): List of file paths
        labels (List[List[int]]): List of labels for each file
        batch_size (int): Batch size
        target_sr (int, optional): Target sample rate. Defaults to 16000.
        target_length (float, optional): Target length in seconds. Defaults to 10.
        mono (bool, optional): Downmix to mono. Keeps the dimension. Defaults to True.
        num_threads (int, optional): Number of threads. Defaults to -1.
        shuffle (bool, optional): Shuffle the dataset. Defaults to False.
        local_rank (int, optional): Local rank in DDP. Defaults to 0.
        global_rank (int, optional): Global rank in DDP. Defaults to 0.
        world_size (int, optional): World size in DDP. Defaults to 1.
        random_crop_size (float, optional): Length of slice in seconds that is cut out randomly from the audio file. Defaults to None.
        start_sec (float, optional): Start time in seconds to set an offset from where on the audio should be used. Defaults to None.
    """

    num_threads = num_threads if num_threads > 0 else torch.multiprocessing.cpu_count()
    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = "gpu" if device_id >= 0 else "cpu"

    pipeline = wav_data_pipeline(
        files=files,
        target_sr=target_sr,
        target_length=target_length,
        mono=mono,
        batch_size=batch_size,
        num_threads=num_threads,
        shuffle=shuffle,
        device=device,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        rnd_crop_size=random_crop_size,
        start_sec=start_sec,
        prefetch_queue_depth=prefetch_factor if prefetch_factor is not None else 2,
        **kwargs,
    )
    pipeline.build()

    return PyTorchIterator(
        pipelines=[pipeline],
        labels=labels,
        output_map=["audio", "label"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        reader_name="Reader",
        prepare_first_batch=True,
    )


# PyTorch Lightning DataModule to use this dataset
class WavDataModule(pl.LightningDataModule):
    def __init__(self, file_paths: List[str], labels: List[str], batch_size: int = 4):
        super(WavDataModule, self).__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.device_id = device_id
        self.num_threads = num_threads
        self.labels = [labels, labels]

    def train_dataloader(self):
        dali_iterator = DaliAudioPipeline(
            files=self.file_paths,
            labels=self.labels,
            batch_size=self.batch_size,
            target_sr=16000,
            target_length=10,
            num_threads=4,
            shuffle=True,
            local_rank=0,
            global_rank=0,
            world_size=1,
        )

        return dali_iterator
