from dataclasses import dataclass, field
from typing import List, Union
from utils.helper import ModeType

default_patch_size: int = 256
default_stride: int = 128
default_aug_times: int = 1
default_mode: ModeType = 'color'
default_scales: List[Union[int, float]] = [1]


@dataclass
class DataPreparationConfiguration:
    patch_size: int = default_patch_size
    stride: int = default_stride
    aug_times: int = default_aug_times
    mode: ModeType = default_mode
    scales: List[Union[int, float]] = field(
        default_factory=lambda: default_scales)
