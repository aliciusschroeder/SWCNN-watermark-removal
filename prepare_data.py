import argparse
from dataclasses import dataclass
import os
from typing import List, Union

from utils.helper import ModeType
from utils.helper import get_config
from utils.data_preparation import DataPreparation
from configs.preparation import DataPreparationConfiguration


@dataclass
class Args:
    scales: List[Union[int, float]]
    patch_size: int
    stride: int
    aug_times: int
    mode: ModeType


def str_to_mode(mode: str) -> ModeType:
    if mode == 'gray':
        return 'gray'
    elif mode == 'color':
        return 'color'
    else:
        raise ValueError(f"Invalid mode: {mode}")


parser = argparse.ArgumentParser(description="SWCNN")
parser.add_argument("--scales", nargs='+', default=[1],
                    help='scales for image pyramid')
parser.add_argument("--patch_size", type=int, default=256,
                    help="Patch size")
parser.add_argument("--stride", type=int, default=128,
                    help="Stride")
parser.add_argument("--aug_times", type=int, default=1,
                    help="Augmentation times (1 = no additional augmentation)")
parser.add_argument("--mode", type=str_to_mode, default='color',
                    help="Mode (gray or color)")

parsed_args = parser.parse_args()
params = DataPreparationConfiguration(
    patch_size=parsed_args.patch_size,
    stride=parsed_args.stride,
    aug_times=parsed_args.aug_times,
    mode=parsed_args.mode,
    scales=parsed_args.scales
)

config = get_config('configs/config.yaml')
data_path = config['data_path']
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data path not found: {data_path}")


def main():
    DataPreparation.prepare_data(
        data_path=config['data_path'],
        patch_size=params.patch_size,
        stride=params.stride,
        aug_times=params.aug_times,
        mode=params.mode,
        scales=params.scales
    )


if __name__ == "__main__":
    main()
