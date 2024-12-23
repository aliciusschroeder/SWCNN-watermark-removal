import argparse
import os
from typing import List, Literal, Union

from utils.data_preparation import DataPreparation
from utils.helper import ModeType, get_config
from configs.preparation import DataPreparationConfiguration, SamplingMethodType
from dataclasses import dataclass


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


default_args = DataPreparationConfiguration()

parser = argparse.ArgumentParser(description="Prepare Data")
parser.add_argument("--scales", nargs='+', default=default_args.scales,
                    help='scales for image pyramid')
parser.add_argument("--patch_size", type=int, default=default_args.patch_size,
                    help="Patch size")
parser.add_argument("--stride", type=int, default=default_args.stride,
                    help="Stride")
parser.add_argument("--aug_times", type=int, default=default_args.aug_times,
                    help="Augmentation times (1 = no additional augmentation)")
parser.add_argument("--mode", type=str_to_mode, default=default_args.mode,
                    help="Mode (gray or color)")
parser.add_argument("--max_t_samples", type=int, default=default_args.max_samples[0],
                    help="Maximum number of training samples to use")
parser.add_argument("--max_v_samples", type=int, default=default_args.max_samples[1],
                    help="Maximum number of validation samples to use")
parser.add_argument("--sampling_method", type=str, default=default_args.sampling_method,
                    help="Sampling method (default or mixed)")
parser.add_argument("--seed", type=int, default=default_args.seed,
                    help="Random seed")


parsed_args = parser.parse_args()
sampling_method: SamplingMethodType = 'default' if parsed_args.sampling_method == 'default' else 'mixed'
max_samples = (parsed_args.max_t_samples, parsed_args.max_v_samples)

config = get_config('configs/config.yaml')
data_path = config['data_path']
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data path not found: {data_path}")


def main():
    step = input(
        "Prepare data for training + validation or test? (_train_/test/finetune): ")
    if step == "ft":
        step = "finetune"
    if not step in ['train', 'test', 'finetune']:
        if not step:
            step = 'train'
        else:
            raise ValueError(f"Invalid step: {step}")

    if step == 'train':
        return DataPreparation.prepare_data(
            data_path=config['data_path'],
            patch_size=parsed_args.patch_size,
            stride=parsed_args.stride,
            aug_times=parsed_args.aug_times,
            mode=parsed_args.mode,
            scales=parsed_args.scales,
            max_samples=max_samples,
            sampling_method=sampling_method,
            seed=parsed_args.seed
        )

    assert step in ['test', 'finetune'], f"Invalid step: {step}"
    purpose: Literal['test', 'finetune'] = 'test' if step == 'test' else 'finetune'
    clean_path = os.path.join(data_path, step, "clean")
    watermarked_path = os.path.join(data_path, step, "watermarked")
    return DataPreparation.prepare_test_data(
        data_path=data_path,
        clean_path=clean_path,
        watermarked_path=watermarked_path,
        mode=parsed_args.mode,
        purpose=purpose,
    )


if __name__ == "__main__":
    main()
