import glob
import logging
import os
import random
from typing import List, Tuple, Union

import cv2
import h5py
import numpy as np

from configs.preparation import SamplingMethodType
from utils.data_augmentation import data_augmentation
from utils.helper import ModeType, StepType
from utils.image import im2patch


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
            step (StepType): 'train', 'validation' or 'test'.
            mode (ModeType, optional): 'gray' or 'color'. Defaults to 'color'.

        Returns:
            str: Full path to the HDF5 file.
        """

        if step == 'validation':
            filename = 'val'
        else:
            filename = step # 'train' or 'test'
            
        if mode == 'color':
            filename += '_color'
        filename += '.h5'
        return os.path.join(data_path, filename)

    @staticmethod
    def input_files(
        data_path: str,
        step: StepType,
        mode: ModeType = 'color'
    ) -> Tuple[str, List[str], str]:
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
        max_samples: Tuple[int, int] = (0, 0),
        sampling_method: SamplingMethodType = 'default',
        seed: Union[int, str] = 42
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
            max_samples (Tuple[int, int], optional): Maximum number of samples for training and validation. Defaults to (0, 0), meaning no limits.
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

        if sampling_method == 'default':
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
                logger=logger,
                max_samples=max_samples[0]

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
                logger=logger,
                max_samples=max_samples[1]
            )
            print(f"Validation set, # samples: {val_size}")
        elif sampling_method == 'mixed':
            # Process Training & Validation Data Together
            if train_files is None or val_files is None:
                raise ValueError("No training or validation files found")
            combined_files = train_files + val_files
            random.seed(seed)
            random.shuffle(combined_files)
            logger.info('Processing training data')
            train_size, val_size = DataPreparation._process_files_mixed(
                files=combined_files,
                h5f_path_train=train_h5f_path,
                h5f_path_val=val_h5f_path,
                patch_size=patch_size,
                stride=stride,
                scales=scales,
                aug_times=aug_times,
                mode=mode,
                logger=logger,
                max_samples=max_samples
            )
            print(f"Training set, # samples: {train_size}")
            print(f"Validation set, # samples: {val_size}")
        else:
            raise ValueError("Invalid sampling method")

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
        max_samples: int = 0
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
                        logger.warning(f"Skipping image {file_path} at scale " +
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
                        if max_samples > 0 and sample_count >= max_samples:
                            return sample_count

                        for m in range(aug_times - 1):
                            augmented_data = data_augmentation(
                                data, np.random.randint(1, 8))
                            aug_key = f"{sample_count}_aug_{m + 1}"
                            h5f.create_dataset(aug_key, data=augmented_data)
                            sample_count += 1
                            if max_samples > 0 and sample_count >= max_samples:
                                return sample_count

            return sample_count

    @staticmethod
    def select_bucket(sample_count: Tuple[int, int], max_samples: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
        results = [
            (0, sample_count),
            (1, (sample_count[0] + 1, sample_count[1])),
            (2, (sample_count[0], sample_count[1] + 1))
        ]

        a_full = sample_count[0] >= max_samples[0] and max_samples[0] > 0
        b_full = sample_count[1] >= max_samples[1] and max_samples[1] > 0

        if a_full and b_full:
            return results[0]
        if a_full:
            return results[2]
        if b_full:
            return results[1]
        if max_samples[0] == 0 and max_samples[1] == 0:
            choice = np.random.choice([1, 2])
        else:
            choice = np.random.choice(
                [1, 2], 
                p=[
                    (max_samples[0] - sample_count[0]) / (max_samples[0] + max_samples[1] - sum(sample_count)), 
                    (max_samples[1] - sample_count[1]) / (max_samples[0] + max_samples[1] - sum(sample_count))
                ]
            )
        return results[choice]

    @staticmethod
    def _process_files_mixed(
        files: List[str],
        h5f_path_train: str,
        h5f_path_val: str,
        patch_size: int,
        stride: int,
        scales: List[Union[int, float]],
        aug_times: int,
        mode: ModeType,
        logger: logging.Logger,
        max_samples: Tuple[int, int] = (10, 1)
    ) -> Tuple[int, int]:
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
        with h5py.File(h5f_path_train, 'w') as h5f_train:
            with h5py.File(h5f_path_val, 'w') as h5f_val:
                sample_count: Tuple[int, int] = (0, 0)
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
                            logger.warning(f"Skipping image {file_path} at scale " +
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
                            selection, sample_count = DataPreparation.select_bucket(sample_count, max_samples)
                            if selection == 0:
                                return sample_count
                            if selection == 1:
                                h5f_train.create_dataset(str(sample_count[0]), data=data)
                            elif selection == 2:
                                h5f_val.create_dataset(str(sample_count[1]), data=data)
                            else:
                                raise ValueError("Invalid selection")


                            for m in range(aug_times - 1):
                                selection, sample_count = DataPreparation.select_bucket(sample_count, max_samples)
                                augmented_data = data_augmentation(
                                    data, np.random.randint(1, 8))
                                aug_key = f"{sample_count[selection-1]}_aug_{m + 1}"
                                if selection == 0:
                                    return sample_count
                                if selection == 1:
                                    h5f_train.create_dataset(aug_key, data=augmented_data)
                                elif selection == 2:
                                    h5f_val.create_dataset(aug_key, data=augmented_data)
                                else:
                                    raise ValueError("Invalid selection")

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

    @staticmethod
    def prepare_test_data(
        data_path: str,
        clean_path: str,
        watermarked_path: str,
        mode: ModeType = 'color',
    ) -> None:
        """
        Prepares the test dataset by pairing clean and watermarked images and saving them into an HDF5 file.

        Args:
            data_path (str): Base directory path.
            clean_path (str): Path to the directory containing clean images.
            watermarked_path (str): Path to the directory containing watermarked images.
            mode (ModeType, optional): 'gray' or 'color'. Defaults to 'color'.
        """
        h5f_path = DataPreparation.get_h5_filepath(data_path, 'test', mode)
        clean_files = sorted(glob.glob(os.path.join(clean_path, "*.jpg")))
        watermarked_files = sorted(glob.glob(os.path.join(watermarked_path, "*.jpg")))

        if len(clean_files) != len(watermarked_files):
            raise ValueError("Number of clean and watermarked images must be the same.")

        with h5py.File(h5f_path, 'w') as h5f:
            for i, (clean_file, watermarked_file) in enumerate(zip(clean_files, watermarked_files)):
                print(f"Processing pair {i+1}/{len(clean_files)}: {clean_file}, {watermarked_file}")

                clean_img = cv2.imread(clean_file)
                watermarked_img = cv2.imread(watermarked_file)

                if clean_img is None or watermarked_img is None:
                    print(f"  Error: Could not read image files. Skipping pair.")
                    continue

                if clean_img.shape != watermarked_img.shape:
                    print(
                        f"  Error: Image shapes do not match. Skipping pair.")
                    continue
                
                if mode == 'gray':
                    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
                    watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
                    clean_img = np.expand_dims(clean_img, axis=0)
                    watermarked_img = np.expand_dims(watermarked_img, axis=0)
                else:
                    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                    watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
                    clean_img = clean_img.transpose((2, 0, 1))
                    watermarked_img = watermarked_img.transpose((2, 0, 1))

                clean_img = DataPreparation.normalize(clean_img.astype(np.float32))
                watermarked_img = DataPreparation.normalize(watermarked_img.astype(np.float32))

                h5f.create_dataset(str(i * 2), data=clean_img)
                h5f.create_dataset(str(i * 2 + 1), data=watermarked_img)