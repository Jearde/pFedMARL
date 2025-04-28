from typing import Union

from pydantic import BaseModel


class AudioParams(BaseModel):
    target_sr: int = 16000
    target_audio_length: int = 10
    audio_slice_length: Union[int, None] = None
    mono: bool = True
    use_dali: bool = True


class FrequencyAugmentationModel(BaseModel):
    use: bool = False
    fmin: float = 0.0
    fmax: float = 8000.0


class TimeAugmentationModel(BaseModel):
    use: bool = False
    time_stretch: float = 0.05
    pitch_shift: float = 0.05
    noise_level: float = 0.05


class FeatureAugmentationModel(BaseModel):
    percentage: float = 0
    label_index: Union[int, None] = None
    frequency: FrequencyAugmentationModel = FrequencyAugmentationModel()
    time: TimeAugmentationModel = TimeAugmentationModel()


# Used
class FeatureModel(BaseModel):
    sr: int = 16000
    n_mels: int = 128
    n_frames: int = 5
    frame_hop_length: float = 0.1
    frame_win_length: float = 1
    n_fft: int = 1024
    hop_length: int = 512
    fmax: Union[float, None] = 8000.0
    fmin: float = 0.0
    win_length: Union[int, None] = None
    channel: int = 1
    power: float = 2.0
    audio_params: AudioParams = AudioParams()
    augmentation: FeatureAugmentationModel = FeatureAugmentationModel()
