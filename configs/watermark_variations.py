"""
Customized watermark variations for the watermarking process. 
This only serves as an example, so make sure to write your own watermark variations.
"""

import random
from typing import Union
from utils.watermark import ArtifactsConfig


def artifacts_variation():
    return ArtifactsConfig(
        alpha=1.00,  # 0.66 #random.uniform(0.55, 0.77),
        intensity=random.uniform(0.50, 1.50),
        kernel_size=random.choice([5, 6, 7]),
    )


def base_stamp(name: str, scale: float = 1.0, alpha=0.5):
    return {
        'watermark_id': name,
        'occupancy': 0,
        'scale': scale,
        'alpha': alpha,
        'position': 'random',
        'application_type': 'stamp',
    }


def logo_by_name(name: str, scale: float = 1.0):
    scale = random.uniform(0.63, 0.93)  # => 0.78 +- 0.15
    alpha = random.uniform(0.33, 1)    # => 0.67 +- 0.33
    stamp = base_stamp(name, scale=scale, alpha=alpha)
    return stamp


def base_map(map: Union[str, int] = 43, scale: float = 0.5, alpha=0.5):
    return {
        'watermark_id': f'{map}',
        'occupancy': 0,
        'scale': scale,
        'application_type': 'map',
        'alpha': alpha,
        'artifacts_config': artifacts_variation(),
    }


def map_edgecase(id: str = 'map_43', scale: float = 0.5, alpha=0.5):
    """
    Simulates an edge case where the watermark is placed at the top or bottom of the image and not fully visible / cropped off.
    """
    move_to_bottom = random.choice(
        [0, -512 + 17])  # +17 to avoid moving the main watermark totally out of the image
    x = random.randint(383, 383 + 512) // 2
    y = (random.uniform(1393, 1431) + move_to_bottom) // 2
    map = base_map(id, scale=scale, alpha=alpha)
    map['position'] = (x, y)
    return map


def map_around_center(id: str = 'map_43', scale: float = 0.5, alpha=0.5):
    """
    Focusses on cases where the center of the watermark map (which differs a little from the rest of the watermark) is placed somewhere in the image.
    It is purposely possible to be cropped off to the left or right, but not to the top or bottom because that's handled by the edgecase.
    """

    # 512 is 2x the width of the base image => 256 after halving = exactly the width of the base image
    x = random.randint(383, 383 + 512) // 2
    # 75 is the height of the center feature in the watermark map
    y = random.uniform(1390 - 512 + 150, 1390) // 2

    map = base_map(id, scale=scale, alpha=alpha)
    map['position'] = (x, y)
    return map


def map_center(id: str = 'map_43', scale: float = 0.5, alpha=0.5):
    map = base_map(id, scale=scale, alpha=alpha)
    map['position'] = 'center'
    return map


# The following methods are used for the validation process on each image
val_relevant_methods = [0, 1, 2, 3]


def get_watermark_variations():
    scale = random.uniform(0.75, 1.25)

    variants = []
    variants.append(logo_by_name('logo_ppco'))  # 0
    variants.append(map_edgecase())             # 1
    variants.append(map_around_center())        # 2
    variants.append(map_center())               # 3

    weights = [0]*len(variants)

    # Set the probabilities for each watermark variation during training
    weights[0] = 5  # logo_by_name('logo_ppco')
    weights[1] = 1  # map_edgecase()
    weights[2] = 1  # map_around_center()
    weights[3] = 1  # map_center()

    if len(variants) != len(weights):
        raise ValueError(
            "The number of watermark variations and their weights must be the same.")

    return variants, weights
