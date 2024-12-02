import math

import numpy as np

from skimage.metrics import mean_squared_error as compare_mse, peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# TODO: Implement working batch_SSIM for test.py
""" def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    Img = np.transpose(Img, (0, 2, 3, 1))
    Iclean = np.transpose(Iclean, (0, 2, 3, 1))
    # print(Iclean.shape)
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range,
                             multichannel=True)
    return (SSIM / Img.shape[0]) """
def batch_SSIM(img, imclean, data_range):
    return 0

def batch_RMSE(img, imclean, data_range):
    img = (img * 255).data.cpu().numpy().astype(np.uint8)
    imclean = (imclean * 255).data.cpu().numpy().astype(np.uint8)
    RMSE = np.mean([
        math.sqrt(compare_mse(imclean[i], img[i]))
        for i in range(img.shape[0])
    ])
    return RMSE


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])



