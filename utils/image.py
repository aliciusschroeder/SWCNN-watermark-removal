import numpy as np

# Used for Gamma correction


def srgb_to_linear(color: np.ndarray) -> np.ndarray:
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(color: np.ndarray) -> np.ndarray:
    return np.where(color <= 0.0031308, color * 12.92, 1.055 * (color ** (1 / 2.4)) - 0.055)
