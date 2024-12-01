import os
import os.path
import random
import glob

from typing import List, Literal

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as udata

from utils import data_augmentation
# from PIL import Image

StepType = Literal['train', 'validation']
ModeType = Literal['gray', 'color']

SCALES = [1, 0.9, 0.8, 0.7]
DEFAULT_MODE : ModeType = 'color'

def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    print("img.shape", img.shape)
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def h5_filepath(data_path: str, step: StepType, mode: ModeType = DEFAULT_MODE) -> str:
    h5f_filename = 'train' if step == 'train' else 'val'
    h5f_filename += '_color' if mode == 'color' else ''
    h5f_filename += '.h5'
    h5f_filepath = f"{data_path}/{h5f_filename}"
    return h5f_filepath

def input_files(data_path: str, step: StepType, mode: ModeType = DEFAULT_MODE) -> tuple[str, List[str], str]:
    folder = 'train' if step == 'train' else 'validation'
    extension = 'png' if mode == 'gray' else 'jpg'
    file_path = os.path.join(data_path, folder, f'*.{extension}')
    files = glob.glob(file_path)
    files.sort()
    h5f_filepath = h5_filepath(data_path, step, mode)
    return file_path, files, h5f_filepath

def prepare_data(data_path, patch_size, stride, aug_times=1, mode=DEFAULT_MODE):
    print('Begin to prepare data')
    print(f"Creating patches of size {patch_size} with stride {stride} and {aug_times - 1} additional augmentation times")
    print(f"Will repeat that process for the following scales: {SCALES}")
    train_files_path, train_files, train_h5f_path = input_files(data_path, 'train', mode)
    val_files_path, val_files, val_h5f_path = input_files(data_path, 'validation', mode)
    print(f"Collecting training files in: {train_files_path}")
    print(f"Collecting validation files in: {val_files_path}")
    if min(len(train_files), len(val_files)) < 0:
        print(f"Error: Found only {len(train_files)} training files and {len(val_files)} validation files")
        return
    else:
        print(f"Found {len(train_files)} training files")
        print(f"Found {len(val_files)} validation files")

    # train
    print('process training data')
    files = train_files
    h5f = h5py.File(train_h5f_path, 'w')

    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = Image.open(files[i])

        h, w, c = img.shape
        # c = 3
        for k in range(len(SCALES)):
            if mode == 'color' and min(int(h * SCALES[k]), int(w * SCALES[k])) < 256:
                continue
            Img = cv2.resize(img, (int(w * SCALES[k]), int(h * SCALES[k])), interpolation=cv2.INTER_CUBIC)
            if mode =='gray':
                Img = np.expand_dims(Img[:, :, 0].copy(), 0)
            else:
                Img = np.transpose(Img, (2, 0, 1))
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], SCALES[k], patches.shape[3] * aug_times))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times - 1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(train_num) + "_aug_%d" % (m + 1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = val_files
    h5f = h5py.File(val_h5f_path, 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        if mode == 'gray':
            img = np.expand_dims(img[:, :, 0].copy(), 0)
        else:
            img = np.transpose(img, (2, 0, 1))
        # img = Image.open(files[i])
        # img = np.expand_dims(img[:, :, 0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    keys: List[str]
    train: bool
    mode: ModeType
    data_path: str

    def __init__(self, train : bool = True, mode : ModeType = DEFAULT_MODE, data_path : str = './'):
        super(Dataset, self).__init__()
        self.train = train
        self.mode = mode
        self.data_path = data_path
        step : StepType = 'train' if train else 'validation'
        h5f_path = h5_filepath(data_path, step, mode)
        h5f = h5py.File(h5f_path, 'r')
        self.keys : List[str] = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index : int) -> torch.Tensor:
        step : StepType = 'train' if self.train else 'validation'
        h5f_path = h5_filepath(self.data_path, step, self.mode)
        h5f = h5py.File(h5f_path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
