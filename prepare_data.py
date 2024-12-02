import argparse
from dataclasses import dataclass
from typing import List, Union

from dataset import prepare_data
from utils.get_config import get_config


@dataclass
class Args:
    scales: List[Union[int, float]]
    patch_size: int
    stride: int
    aug_times: int
    mode: str


parser = argparse.ArgumentParser(description="SWCNN")
parser.add_argument("--scales", nargs='+', default=[1],
                    help='scales for image pyramid')
parser.add_argument("--patch_size", type=int, default=256,
                    help="Patch size")
parser.add_argument("--stride", type=int, default=128,
                    help="Stride")
parser.add_argument("--aug_times", type=int, default=1,
                    help="Augmentation times (1 = no additional augmentation)")
parser.add_argument("--mode", type=str, default='color',
                    help="Mode (gray or color)")

parsed_args = parser.parse_args()
args = Args(
    scales=parsed_args.scales,
    patch_size=parsed_args.patch_size,
    stride=parsed_args.stride,
    aug_times=parsed_args.aug_times,
    mode=parsed_args.mode
)

config = get_config('configs/config.yaml')


def main():
    prepare_data(
        data_path=config['train_data_path'],
        patch_size=args.patch_size,
        stride=args.stride,
        aug_times=args.aug_times,
        mode=args.mode,
        scales=args.scales
    )
    # prepare_data(data_path=config['train_data_path'], patch_size=256, stride=128, aug_times=1, mode='color')


if __name__ == "__main__":
    main()
