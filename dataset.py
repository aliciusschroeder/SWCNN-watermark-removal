import os
import random
from typing import List

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
