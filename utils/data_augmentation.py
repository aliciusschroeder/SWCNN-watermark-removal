import numpy as np
from numpy.typing import NDArray


def data_augmentation(image: NDArray[np.float32], mode: int) -> NDArray[np.float32]:
    """
    Perform data augmentation on the given image by applying one or multiple transformations based on the mode.

    Args:
        image (NDArray[np.float32]): The input image, expected to have shape (C, H, W).
        mode (int): The augmentation mode, an integer between 0 and 7 inclusive.

    Returns:
        NDArray[np.float32]: The augmented image, with the same shape as the input image (C, H, W).
    """
    if not (0 <= mode <= 7):
        raise ValueError(
            "Mode should be an integer between 0 and 7 inclusive.")

    # Transpose to (H, W, C) for OpenCV operations
    out = image.transpose(1, 2, 0)

    if mode == 0:
        # Original
        pass
    elif mode == 1:
        # Flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # Rotate counterclockwise 90 degrees
        out = np.rot90(out)
    elif mode == 3:
        # Rotate 90 degrees and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # Rotate 180 degrees
        out = np.rot90(out, k=2)
    elif mode == 5:
        # Rotate 180 degrees and flip up and down
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # Rotate 270 degrees
        out = np.rot90(out, k=3)
    elif mode == 7:
        # Rotate 270 degrees and flip up and down
        out = np.rot90(out, k=3)
        out = np.flipud(out)

    # Transpose back to (C, H, W)
    return out.transpose(2, 0, 1).astype(np.float32)
