import os
import torch
import lightning as pl
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import (
    DALIGenericIterator,
    LastBatchPolicy
)
from typing import List
import numpy as np

class PyTorchIterator(DALIGenericIterator):
    def __init__(self, labels: dict[List[int]], *kargs, **kvargs):
        """ Overloading of DALIGenericIterator for Multi-Label

        This is for supporting multiple labels per file. The index of the file is used to get the value in the list.

        Args:
            labels (List[List[int]]): _description_
        """
        super().__init__(*kargs, **kvargs)
        self.labels = labels

        # self.labels = torch.tensor(labels) # Should this be moved to gpu already?
        # biggest_dim = np.argmax(list(self.labels.shape)) # Biggest dimension should be number of files to be first
        # self.labels = self.labels.movedim(biggest_dim, 0)

    def __next__(self):
        out = super().__next__()
        
        # Use only the output of the first pipeline (not suitable for multiple pipelines in parallel)
        out = out[0]

        # Convert int to string and then to int
        idcs = [int(''.join(map(chr, row))) for row in out['label']]
        
        labels = torch.cat(
            (
                torch.stack(list(map(self.labels.get, idcs)), dim=0).to(out['audio'].device),
                torch.tensor(idcs).unsqueeze(1).to(out['audio'].device),
            ),
            dim=-1
            ).to(out['audio'].device)
        

        # Get the label based on the index
        return out['audio'], labels
    
@pipeline_def
def numpy_data_pipeline(
    files: List[str],
    filename_len: int,
    target_sr: int = 16000,
    target_length: int = 10,
    shuffle: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
    device: str = 'cpu',
    direct_store: bool = False,
    mono: bool = True,
    rnd_crop_size: float = None,
    start_sec: float = None,
    ):
    """ Load audio files and return audio and label
    
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

    if rnd_crop_size is not None:
        rnd_choice_list = list(range(int(target_length * target_sr) - int(rnd_crop_size * target_sr)))
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

    # Load audio   
    audio = fn.readers.numpy(
        files=files,
        cache_header_information=True,
        out_of_bounds_policy='pad',
        fill_value=0.0,
        # roi_shape=[target_length * target_sr],
        roi_start=start,
        roi_end=end,
        roi_axes=[1],
        random_shuffle=shuffle,
        num_shards=num_shards,
        shard_id=shard_id,
        device=device if direct_store else 'cpu',
        seed=42,
        name="Reader",
    )

    label = fn.get_property(audio, key="source_info", name="Label")
    
    # Get only the name of the file. It is important that all file names have the same length
    label = label[-filename_len:-4]

    # If not using downmix for reducing the dimension, you can use mean and keep the dimension
    if mono:
        audio = fn.reductions.mean(audio, axes=[-2], keep_dims=True)

    return audio, label

def DaliNumpyPipeline(
    files: List[str],
    labels: List[List[int]],
    batch_size: int,
    target_sr: int = 16000,
    target_length: int = 10,
    num_threads: int = -1,
    prefetch_factor: int = 2,
    shuffle: bool = False,
    local_rank: int = 0,
    global_rank: int = 0,
    world_size: int = 1,
    mono: bool = True,
    random_crop_size: float = None,
    start_sec: float = None,
    direct_store: bool = False,
    **kwargs,
    ):

    num_threads = num_threads if num_threads > 0 else torch.multiprocessing.cpu_count()
    device_id = local_rank
    shard_id = global_rank
    num_shards = world_size

    device = 'gpu' if device_id >= 0 else 'cpu'

    filename_len = len(files[0].split("/")[-1])

    # Map the labels to the file index
    labels_dict = {}
    for idx, file in enumerate(files):
        labels_dict[int(file[-filename_len:-4])] = torch.tensor(labels)[:,idx]


    pipeline = numpy_data_pipeline(
        files=files,
        filename_len=filename_len,
        # label_lists=[external_data_source(label_lists[0]), external_data_source(label_lists[1])],
        target_sr=target_sr,
        target_length=target_length,
        batch_size=batch_size,
        num_threads=num_threads,
        shuffle=shuffle,
        device=device,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        direct_store=direct_store,
        mono=mono,
        rnd_crop_size=random_crop_size,
        start_sec=start_sec,
        # prefetch_factor=prefetch_factor,
        **kwargs,
    )
    pipeline.build()

    # TODO How to get labels?
    # img , labels = pipeline.run()
    # print(img)
    # print(labels)
    # ascii_array = labels.at(0) 
    # decoded_string = ''.join(chr(value) for value in ascii_array)
    # labels_array = labels.as_array()

    return PyTorchIterator(
        pipelines=[pipeline],
        labels=labels_dict,
        output_map=['audio', 'label'],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        reader_name="Reader",
        prepare_first_batch=True,
    )

# PyTorch Lightning DataModule to use this dataset
class NumpyDataModule(pl.LightningDataModule):
    def __init__(self, file_paths: List[str], labels: List[str], batch_size: int = 4):
        super(NumpyDataModule, self).__init__()
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.device_id = device_id
        self.num_threads = num_threads
        self.labels = [labels, labels]

    def train_dataloader(self):
        dali_iterator = DaliNumpyPipeline(
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

# Example usage with PyTorch Lightning DataModule
if __name__ == "__main__":
    import pickle
    from tqdm.auto import tqdm
    from pathlib import Path

    current_dir = Path(__file__).resolve().parent

    # A list of paths to WAV files
    file_list = [str(f) for f in Path('/mnt/nfs/datasets/audioset_numpy/eval/Speech').glob('*.npy')]
    label_lists = [[0, 1] * len(file_list)]

    # Copy all numpy files and name them with the index
    path_numbered = Path("/mnt/nfs/datasets/audioset_numpy/all/eval")
    path_numbered.mkdir(parents=True, exist_ok=True)

    digits_files = len(str(len(file_list)))
    for i, file in tqdm(enumerate(file_list), total=len(file_list)):
        file_name = f"{i}".zfill(digits_files)
        np.save(path_numbered / f"{file_name}.npy", np.load(file))

    files = [str(f) for f in path_numbered.glob('*.npy')]

    # Load file_list and label_list from a pickle file
    # with open(current_dir / 'tmp/file_list.pkl', 'rb') as f:
    #     file_list = pickle.load(f)

    # with open(current_dir / 'tmp/label_list.pkl', 'rb') as f:
    #     label_lists = pickle.load(f)

    batch_size = 256

    target_sr = 16000
    target_length = 10

    num_threads = os.cpu_count()
    shuffle = True
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
    num_gpus = torch.cuda.device_count()

    dali_iterator = DaliNumpyPipeline(
        files=files,
        # files_path=str(path_numbered),
        labels=label_lists,
        batch_size=batch_size,
        target_sr=target_sr,
        target_length=target_length,
        num_threads=num_threads,
        shuffle=shuffle,
        local_rank=device_id,
        global_rank=device_id,
        world_size=num_gpus,
        )

    # Iterate over the DataLoader
    for epoch in range(5):
        for batch_idx, batch in enumerate(tqdm(dali_iterator, total=len(dali_iterator), desc=f"Epoch {epoch}")):
            audio = batch[0]
            label = batch[1]

            if epoch == 0 and batch_idx == 0:
                print(f"audio: {audio.shape}, label: {label.shape}")
                print(f"Device: {audio.device}, {label.device}")