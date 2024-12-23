import os
import random
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from utils.data_preparation import DataPreparation
from utils.helper import ModeType, StepType


class Dataset(TorchDataset):
    """
    Dataset for loading image patches from HDF5 files.

    Attributes:
        keys (List[str]): List of keys in the HDF5 file.
        train (bool): Flag indicating training or validation mode.
        mode (ModeType): Color mode of the images ('gray' or 'color').
        data_path (str): Path to the dataset directory.
    """

    def __init__(self, train: bool = True, mode: ModeType = 'color', data_path: str = 'data/'):
        """
        Initializes the Dataset.

        Args:
            train (bool, optional): If True, loads training data, otherwise validation data. Defaults to True.
            mode (ModeType, optional): Mode of images, 'gray' or 'color'. Defaults to 'color'.
            data_path (str, optional): Path to the dataset. Defaults to 'data/'.
        """
        super().__init__()
        self.train = train
        self.mode: ModeType = mode
        self.data_path = data_path
        step: StepType = 'train' if train else 'validation'
        h5f_path = DataPreparation.get_h5_filepath(data_path, step, mode)

        if not os.path.exists(h5f_path):
            raise FileNotFoundError(f"HDF5 file not found at path: {h5f_path}")

        with h5py.File(h5f_path, 'r') as h5f:
            self.keys: List[str] = list(h5f.keys())

        random.shuffle(self.keys)

    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
            int: Number of samples.
        """
        return len(self.keys)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Retrieves the sample at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Tensor representation of the image patch.
        """
        step: StepType = 'train' if self.train else 'validation'
        h5f_path = DataPreparation.get_h5_filepath(
            self.data_path, step, self.mode)

        with h5py.File(h5f_path, 'r') as h5f:
            key = self.keys[index]
            data = np.array(h5f[key])

        return torch.tensor(data, dtype=torch.float32)


class PairedDataset(TorchDataset):
    """
    Dataset for loading paired image patches from an HDF5 file for testing.

    Attributes:
        keys (List[Tuple[str, str]]): List of key pairs (clean, watermarked) in the HDF5 file.
        mode (ModeType): Color mode of the images ('gray' or 'color').
        data_path (str): Path to the dataset directory.
    """

    def __init__(self, mode: ModeType = 'color', data_path: str = 'data/'):
        """
        Initializes the TestDataset.

        Args:
            mode (ModeType, optional): Mode of images, 'gray' or 'color'. Defaults to 'color'.
            data_path (str, optional): Path to the dataset. Defaults to 'data/'.
        """
        super().__init__()
        self.mode: ModeType = mode
        self.data_path = data_path
        h5f_path = DataPreparation.get_h5_filepath(data_path, 'test', mode)

        if not os.path.exists(h5f_path):
            raise FileNotFoundError(f"HDF5 file not found at path: {h5f_path}")

        with h5py.File(h5f_path, 'r') as h5f:
            # Assuming keys are stored in pairs (clean_key, watermarked_key)
            self.keys: List[Tuple[str, str]] = [
                (str(i), str(i + 1)) for i in range(0, len(h5f.keys()), 2)
            ]

        random.shuffle(self.keys)

    def __len__(self) -> int:
        """
        Returns the total number of paired samples.

        Returns:
            int: Number of paired samples.
        """
        return len(self.keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the paired sample at the specified index.

        Args:
            index (int): Index of the sample pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensor representations of the clean and watermarked image patches.
        """
        h5f_path = DataPreparation.get_h5_filepath(
            self.data_path, 'test', self.mode)

        with h5py.File(h5f_path, 'r') as h5f:
            clean_key, watermarked_key = self.keys[index]
            clean_data = np.array(h5f[clean_key])
            watermarked_data = np.array(h5f[watermarked_key])

        return (
            torch.tensor(clean_data, dtype=torch.float32),
            torch.tensor(watermarked_data, dtype=torch.float32)
        )
