import numpy as np
from numpy.typing import NDArray

def data_augmentation(image: NDArray[np.float32], mode: int) -> NDArray[np.float32]:
    """
    Perform data augmentation on the given image.

    Args:
        image (NDArray[np.float32]): The input image, expected to have shape (C, H, W).
        mode (int): The augmentation mode, an integer between 0 and 7 inclusive.

    Returns:
        NDArray[np.float32]: The augmented image, with the same shape as the input image (C, H, W).
    """

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