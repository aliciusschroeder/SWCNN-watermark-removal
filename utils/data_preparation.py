import glob
import logging
import os
from typing import List, Literal, Union

import cv2
import h5py
import numpy as np

from utils.data_augmentation import data_augmentation
from utils.helper import ModeType
from utils.image import im2patch


StepType = Literal['train', 'validation']


class DataPreparation():
    """
    Handles data preparation tasks including normalization, patch extraction,
    data augmentation, and saving processed data to HDF5 files.
    """

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalizes the image data to the range [0, 1].

        Args:
            data (np.ndarray): Input image data.

        Returns:
            np.ndarray: Normalized image data.
        """
        return data / 255.0

    @staticmethod
    def get_h5_filepath(data_path: str, step: StepType, mode: ModeType = 'color') -> str:
        """
        Constructs the HDF5 file path based on the step and mode.

        Args:
            data_path (str): Base directory path.
            step (StepType): 'train' or 'validation'.
            mode (ModeType, optional): 'gray' or 'color'. Defaults to 'color'.

        Returns:
            str: Full path to the HDF5 file.
        """
        filename = "train" if step == 'train' else "val"
        if mode == 'color':
            filename += '_color'
        filename += '.h5'
        return os.path.join(data_path, filename)

    @staticmethod
    def input_files(data_path: str, step: StepType, mode: ModeType = 'color') -> tuple:
        """
        Retrieves the list of image files for the specified step and mode.

        Args:
            data_path (str): Base directory path.
            step (StepType): 'train' or 'validation'.
            mode (ModeType, optional): 'gray' or 'color'. Defaults to 'color'.

        Returns:
            tuple: (file_pattern, sorted list of files, HDF5 file path)
        """
        folder = 'train' if step == 'train' else 'validation'
        extension = 'png' if mode == 'gray' else 'jpg'
        file_pattern = os.path.join(data_path, folder, f'*.{extension}')
        files = sorted(glob.glob(file_pattern))
        h5f_path = DataPreparation.get_h5_filepath(data_path, step, mode)
        return file_pattern, files, h5f_path

    @staticmethod
    def prepare_data(
        data_path: str,
        patch_size: int,
        stride: int,
        aug_times: int = 1,
        mode: ModeType = 'color',
        scales: List[Union[int, float]] = [1],
    ) -> None:
        """
        Prepares the dataset by processing images and saving them into HDF5 files.

        Args:
            data_path (str): Path to the dataset directory.
            patch_size (int): Size of the image patches.
            stride (int): Stride for patch extraction.
            aug_times (int, optional): Number of augmentation times. Defaults to 1.
            mode (ModeType, optional): 'gray' or 'color'. Defaults to 'color'.
            scales (List[Union[int, float]], optional): List of scales for resizing. Defaults to [1].
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info('Begin to prepare data')
        logger.info(
            f"Creating patches of size {patch_size} with stride {stride} "
            f"and {aug_times - 1} additional augmentation times"
        )
        logger.info(f"Processing the following scales: {scales}")

        train_files_path, train_files, train_h5f_path = DataPreparation.input_files(
            data_path, 'train', mode)
        val_files_path, val_files, val_h5f_path = DataPreparation.input_files(
            data_path, 'validation', mode)

        logger.info(f"Collecting training files in: {train_files_path}")
        logger.info(f"Collecting validation files in: {val_files_path}")

        if not train_files or not val_files:
            logger.error(
                f"Error: Found {len(train_files)} training files and "
                f"{len(val_files)} validation files"
            )
            return

        logger.info(f"Found {len(train_files)} training files")
        logger.info(f"Found {len(val_files)} validation files")

        # Process Training Data
        logger.info('Processing training data')
        train_size = DataPreparation._process_files(
            files=train_files,
            h5f_path=train_h5f_path,
            patch_size=patch_size,
            stride=stride,
            scales=scales,
            aug_times=aug_times,
            mode=mode,
            logger=logger
        )
        print(f"Training set, # samples: {train_size}")

        # Process Validation Data
        logger.info('Processing validation data')
        val_size = DataPreparation._process_files(
            files=val_files,
            h5f_path=val_h5f_path,
            patch_size=patch_size,
            stride=stride,
            scales=scales,
            aug_times=1,
            mode=mode,
            logger=logger
        )
        print(f"Validation set, # samples: {val_size}")
        """ DataPreparation._process_validation_files(
            files=val_files,
            h5f_path=val_h5f_path,
            mode=mode
        ) """

    @staticmethod
    def _process_files(
        files: List[str],
        h5f_path: str,
        patch_size: int,
        stride: int,
        scales: List[Union[int, float]],
        aug_times: int,
        mode: ModeType,
        logger: logging.Logger,
    ) -> int:
        """
        Processes training or validation files and saves them into HDF5.

        Args:
            files (List[str]): List of image file paths.
            h5f_path (str): Path to the HDF5 file.
            patch_size (int): Size of the image patches.
            stride (int): Stride for patch extraction.
            scales (List[Union[int, float]]): List of scales for resizing.
            aug_times (int): Number of augmentation times.
            mode (ModeType): 'gray' or 'color'.
            logger (logging.Logger): Logger object.
        """
        with h5py.File(h5f_path, 'w') as h5f:
            sample_count = 0
            for file_path in files:
                img = cv2.imread(file_path)
                if img is None:
                    logger.warning(f"Failed to read image: {file_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                h, w, c = img.shape
                for scale in scales:
                    scaled_h, scaled_w = int(h * scale), int(w * scale)
                    if mode == 'color' and min(scaled_h, scaled_w) < 256:
                        logger.warning(f"Skipping image {file_path} at scale "+
                                       f"{scale} due to insufficient size.")
                        continue

                    resized_img = cv2.resize(
                        img,
                        (scaled_w, scaled_h),
                        interpolation=cv2.INTER_CUBIC
                    )

                    if mode == 'gray':
                        resized_img = np.expand_dims(
                            resized_img[:, :, 0], axis=0)
                    else:
                        resized_img = resized_img.transpose((2, 0, 1))

                    normalized_img = DataPreparation.normalize(
                        resized_img.astype(np.float32))
                    patches = im2patch(
                        normalized_img, win=patch_size, stride=stride)

                    logger.info(
                        f"Processing file: {file_path}, scale: {scale:.1f}, "
                        f"# samples: {patches.shape[3] * aug_times}"
                    )

                    for n in range(patches.shape[3]):
                        data = patches[:, :, :, n].copy()
                        h5f.create_dataset(str(sample_count), data=data)
                        sample_count += 1

                        for m in range(aug_times - 1):
                            augmented_data = data_augmentation(
                                data, np.random.randint(1, 8))
                            aug_key = f"{sample_count}_aug_{m + 1}"
                            h5f.create_dataset(aug_key, data=augmented_data)
                            sample_count += 1
            return sample_count

    # Use this method instead of _process_files for validation data if you want to skip patching

    @staticmethod
    def _process_validation_files(
        files: List[str],
        h5f_path: str,
        mode: ModeType,
        logger
    ) -> None:
        """
        Processes validation files and saves them into HDF5.

        Args:
            files (List[str]): List of validation image file paths.
            h5f_path (str): Path to the validation HDF5 file.
            mode (ModeType): 'gray' or 'color'.
            logger (logging.Logger): Logger object.
        """
        with h5py.File(h5f_path, 'w') as h5f:
            val_count = 0
            for file_path in files:
                logger.info(f"Processing validation file: {file_path}")
                img = cv2.imread(file_path)
                if img is None:
                    logger.warning(f"Failed to read image: {file_path}")
                    continue

                if mode == 'gray':
                    img = np.expand_dims(img[:, :, 0], axis=0)
                else:
                    img = img.transpose((2, 0, 1))

                normalized_img = DataPreparation.normalize(
                    img.astype(np.float32))
                h5f.create_dataset(str(val_count), data=normalized_img)
                val_count += 1
            logger.info(f"Validation set, # samples: {val_count}")
