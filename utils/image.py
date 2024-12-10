import numpy as np
from numpy.typing import NDArray


# Core parameters for sRGB to linear and linear to sRGB conversion
# These numbers are derived from the sRGB standard (IEC 61966-2-1:1999), used for gamma correction in color spaces.
COLOR_SPACE_PARAMS = {
    "srgb_to_linear": {
        "threshold": 0.04045,  # Transition point between linear and non-linear scaling in sRGB.
        "scale_factor": 12.92, # Scaling factor for linear colors below the threshold.
        "gamma": 2.4,          # Gamma correction exponent for non-linear colors.
        "offset": 0.055,       # Offset added to normalize non-linear scaling.
        "multiplier": 1.055,   # Multiplier for non-linear scaling adjustment.
    },
    "linear_to_srgb": {
        "threshold": 0.0031308, # Transition point between linear and non-linear scaling in linear space.
        "scale_factor": 12.92,  # Scaling factor for linear colors below the threshold.
        "gamma": 2.4,           # Inverse gamma correction for non-linear colors.
        "offset": 0.055,        # Offset to normalize sRGB scaling.
        "multiplier": 1.055,    # Multiplier to match the sRGB curve.
    },
}

def introduce_random_variation(base_params: dict) -> dict:
    """
    Introduces controlled random variation to the sRGB and linear space transformation parameters.
    
    The random variations simulate inconsistencies or measurement errors in real-world displays or sensors,
    allowing for more robust simulations and testing of image processing algorithms.

    Parameters:
        base_params (dict): Standardized parameters for sRGB and linear space conversions.

    Returns:
        dict: Parameters with randomized deviations. 
        - `threshold`: Small perturbation around the transition point (±0.01).
        - `scale_factor`: Larger variability to simulate physical device inaccuracies (±1.0).
        - `gamma`: Perturbation to simulate imperfect gamma curve modeling (±0.3).
        - `offset` and `multiplier`: Minimal perturbation for minor calibration inconsistencies (±0.01).
    """
    return {
        "threshold": base_params["threshold"] + np.random.uniform(-0.01, 0.01),
        "scale_factor": base_params["scale_factor"] + np.random.uniform(-1.0, 1.0),
        "gamma": base_params["gamma"] + np.random.uniform(-0.3, 0.3),
        "offset": base_params["offset"] + np.random.uniform(-0.01, 0.01),
        "multiplier": base_params["multiplier"] + np.random.uniform(-0.01, 0.01),
    }


def srgb_to_linear(color: np.ndarray, randomize: bool = True) -> np.ndarray:
    """
    Converts sRGB color values to linear color space, optionally introducing parameter variability.

    The sRGB to linear transformation applies gamma correction, a standard technique in image processing 
    to linearize the perception of light intensity by the human eye.

    Parameters:
        color (np.ndarray): Array of sRGB values (range 0–1).
        randomize (bool): If True, introduces variability in conversion parameters.

    Returns:
        np.ndarray: Linear color space values.
    """
    params = (
        introduce_random_variation(COLOR_SPACE_PARAMS["srgb_to_linear"])
        if randomize
        else COLOR_SPACE_PARAMS["srgb_to_linear"]
    )

    return np.where(
        color <= params["threshold"],
        color / params["scale_factor"],
        ((color + params["offset"]) / params["multiplier"]) ** params["gamma"],
    )


def linear_to_srgb(color: np.ndarray, randomize: bool = True) -> np.ndarray:
    """
    Converts linear color space values to sRGB color space, optionally introducing parameter variability.

    The linear to sRGB transformation applies inverse gamma correction, mapping linear light intensities 
    to the non-linear sRGB space for display on standard devices.

    Parameters:
        color (np.ndarray): Array of linear color values (range 0–1).
        randomize (bool): If True, introduces variability in conversion parameters.

    Returns:
        np.ndarray: sRGB color space values.
    """
    params = (
        introduce_random_variation(COLOR_SPACE_PARAMS["linear_to_srgb"])
        if randomize
        else COLOR_SPACE_PARAMS["linear_to_srgb"]
    )

    return np.where(
        color <= params["threshold"],
        color * params["scale_factor"],
        params["multiplier"] * (color ** (1 / params["gamma"])) - params["offset"],
    )

