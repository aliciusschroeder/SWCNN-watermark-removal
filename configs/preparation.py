from dataclasses import dataclass, field
from typing import List, Literal, Tuple, Union
from utils.helper import ModeType

SamplingMethodType = Literal['default', 'mixed']


@dataclass
class DataPreparationConfiguration:
    # See DataPreparation.prepare_data() in ./utils/prepare_data.py for param descriptions
    patch_size: int = 256
    stride: int = 128
    aug_times: int = 1
    mode: ModeType = 'color'
    scales: List[Union[int, float]] = field(
        default_factory=lambda: [1])
    max_samples: Tuple[int, int] = (0, 0)
    sampling_method: SamplingMethodType = 'default'
    seed: Union[int, str] = 42
