import math

import numpy as np
from torch import Tensor as TorchTensorType

from skimage.metrics import (
    mean_squared_error as compare_mse,
    peak_signal_noise_ratio as compare_psnr,
    structural_similarity as compare_ssim,
)


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

def batch_RMSE(
    img: TorchTensorType, 
    imclean: TorchTensorType, 
    data_range: float
) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) for a batch of images.
    
    Args:
        img (torch.Tensor): Noisy images batch, values normalized between 0 and 1.
        imclean (torch.Tensor): Clean images batch, values normalized between 0 and 1.
        data_range (float): The data range of the input images.
        
    Returns:
        float: The average RMSE for the batch.
    """
    img_np = (img * data_range).data.cpu().numpy().astype(np.uint8)
    imclean_np = (imclean * data_range).data.cpu().numpy().astype(np.uint8)
    RMSE = np.mean([
        math.sqrt(compare_mse(imclean_np[i], img_np[i]))
        for i in range(img_np.shape[0])
    ])
    return float(RMSE)


def batch_PSNR(
    img: TorchTensorType, 
    imclean: TorchTensorType, 
    data_range: float
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) for a batch of images.
    
    Args:
        img (torch.Tensor): Noisy images batch, values normalized between 0 and 1.
        imclean (torch.Tensor): Clean images batch, values normalized between 0 and 1.
        data_range (float): The data range of the input images.
        
    Returns:
        float: The average PSNR for the batch.
    """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]

