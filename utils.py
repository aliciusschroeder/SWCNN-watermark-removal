import math
import random

import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch import device as torchdevice
from torch.cuda import is_available as cuda_is_available
import torch

import yaml
from PIL import Image
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from models import VGG16


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    Img = np.transpose(Img, (0, 2, 3, 1))
    Iclean = np.transpose(Iclean, (0, 2, 3, 1))
    # print(Iclean.shape)
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range,
                             multichannel=True)
    return (SSIM / Img.shape[0])


def batch_RMSE(img, imclean, data_range):
    img = (img * 255).data.cpu().numpy().astype(np.uint8)
    imclean = (imclean * 255).data.cpu().numpy().astype(np.uint8)
    RMSE = np.mean([
        math.sqrt(compare_mse(imclean[i], img[i]))
        for i in range(img.shape[0])
    ])
    return RMSE

def load_watermark(random_img, alpha, data_path="data/watermarks/"):
    watermark = Image.open(f"{data_path}{random_img}.png").convert("RGBA")
    w, h = watermark.size
    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if color[3] != 0:
                transparence = int(255 * alpha)
                color = color[:-1] + (transparence,)
                watermark.putpixel((i, k), color)
    return watermark

def apply_watermark(base_image, watermark, scale, position):
    scaled_watermark = watermark.resize((int(watermark.width * scale), int(watermark.height * scale)))
    layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    layer.paste(scaled_watermark, position, scaled_watermark)
    return Image.alpha_composite(base_image, layer)

def calculate_occupancy(img_cnt, occupancy_ratio):
    sum_pixels = np.sum(img_cnt > 0)
    total_pixels = img_cnt.size
    return sum_pixels > total_pixels * occupancy_ratio / 100

def add_watermark_noise_generic(
    img_train,
    occupancy=50,
    self_supervision=False,
    same_random=0,
    alpha=0.3,
    img_id=None,
    scale_img=None,
    fixed_position=None,
    standalone=False
):
    if standalone:
        # Standalone processing for single image
        data_path = "water.png"
        watermark = Image.open(data_path).convert("RGBA")
        noise = img_train.numpy()
        _, h, w = noise.shape
        occupancy = np.random.uniform(0, occupancy)

        noise = np.ascontiguousarray(np.transpose(noise, (1, 2, 0)))
        noise = np.uint8(noise * 255)
        noise_pil = Image.fromarray(noise)
        img_for_cnt = Image.new("L", (w, h), 0)

        while True:
            angle = random.randint(-45, 45)
            scale = random.uniform(0.5, 1.0)
            rotated_watermark = watermark.rotate(angle, expand=1).resize(
                (int(watermark.width * scale), int(watermark.height * scale))
            )
            x = random.randint(-rotated_watermark.width, w)
            y = random.randint(-rotated_watermark.height, h)
            noise_pil = apply_watermark(noise_pil, rotated_watermark, 1.0, (x, y))
            img_for_cnt = apply_watermark(img_for_cnt.convert("RGBA"), rotated_watermark, 1.0, (x, y)).convert("L")
            img_cnt = np.array(img_for_cnt)
            if calculate_occupancy(img_cnt, occupancy):
                break
        return noise_pil

    # Batch processing
    if img_id is not None:
        random_img = img_id
    else:
        random_img = same_random if self_supervision else random.randint(1, 173)

    # Handle alpha for different scenarios
    if scale_img is not None:
        alpha = alpha  # Fixed alpha
    else:
        alpha = alpha + random.randint(0, 70) * 0.01 if 'add_watermark_noise_B' in add_watermark_noise_generic.__name__ else alpha

    watermark = load_watermark(random_img, alpha)
    img_train = img_train.numpy()
    _, _, img_h, img_w = img_train.shape
    occupancy = np.random.uniform(0, occupancy)

    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))

    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8)).convert("RGBA")
        img_for_cnt = Image.new("L", (img_w, img_h), 0)

        while True:
            # Determine scaling
            scale = scale_img if scale_img is not None else np.random.uniform(0.5, 1.0)
            scaled_watermark = watermark.resize((int(watermark.width * scale), int(watermark.height * scale)))

            # Determine position
            if fixed_position is not None:
                x, y = fixed_position
            else:
                x = random.randint(0, img_w - scaled_watermark.width)
                y = random.randint(0, img_h - scaled_watermark.height)

            # Apply watermark
            tmp = apply_watermark(tmp, scaled_watermark, 1.0, (x, y))
            img_for_cnt = apply_watermark(img_for_cnt.convert("RGBA"), scaled_watermark, 1.0, (x, y)).convert("L")
            img_cnt = np.array(img_for_cnt)

            if calculate_occupancy(img_cnt, occupancy):
                img_rgb = np.array(tmp).astype(float) / 255.0
                img_train[i] = img_rgb[:, :, :3]
                break

    img_train = np.transpose(img_train, (0, 3, 1, 2))
    return torch.tensor(img_train)

def add_watermark_noise(
    img_train, occupancy=50, self_supervision=False, same_random=0, alpha=0.3
):
    return add_watermark_noise_generic(
        img_train=img_train,
        occupancy=occupancy,
        self_supervision=self_supervision,
        same_random=same_random,
        alpha=alpha
    )

def add_watermark_noise_B(
    img_train, occupancy=50, self_supervision=False, same_random=0, alpha=0.3
):
    return add_watermark_noise_generic(
        img_train=img_train,
        occupancy=occupancy,
        self_supervision=self_supervision,
        same_random=same_random,
        alpha=alpha,
        # Additional parameters can be set here if needed
    )


def add_watermark_noise_test(
    img_train,
    occupancy=50,
    img_id=3,
    scale_img=1.5,
    self_supervision=False,
    same_random=0,
    alpha=0.3
):
    return add_watermark_noise_generic(
        img_train=img_train,
        occupancy=occupancy,
        self_supervision=self_supervision,
        same_random=same_random,
        alpha=alpha,
        img_id=img_id,
        scale_img=scale_img,
        fixed_position=(128, 128)
    )


def load_froze_vgg16():
    # Fine-tuning
    model_pretrain_vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)

    # Load VGG16
    net_vgg = VGG16()
    model_dict = net_vgg.state_dict()
    pretrained_dict = model_pretrain_vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Load Parameters
    net_vgg.load_state_dict(pretrained_dict)

    for child in net_vgg.children():
        for p in child.parameters():
            p.requires_grad = False
    device_ids = [0]

    device = torchdevice('cuda' if cuda_is_available() else 'cpu')
    model_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).to(device)
    return model_vgg


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
