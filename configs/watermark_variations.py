"""
Customized watermark variations for the watermarking process. Make sure to write your own watermark variations.
"""

import random
from utils.watermark import ArtifactsConfig

def artifacts_variation():
    return ArtifactsConfig(
        alpha=random.uniform(0.55, 0.77),
        intensity=random.uniform(1.00, 2.00),
        kernel_size=random.choice([7, 11, 15]),
    )


def logo_by_name(name: str, scale: float = 1.0):
    return {
        'watermark_id': name,
        'occupancy': 0,
        'scale': scale,
        'alpha': random.uniform(0.33, 1),
        'position': 'random',
        'application_type': 'stamp',
    }


def logo_milios():
    return {
        'watermark_id': 'logo_milios',
        'occupancy': 0,
        'scale': random.uniform(0.63, 0.93), # => 0.78 +- 0.15
        'alpha': random.uniform(0.33, 1),   # => 0.67 +- 0.33
        'position': 'random',
        'application_type': 'stamp',
    }


def milios_map():
    return {
        'watermark_id': 'map_43',
        'occupancy': 0,
        'scale': 0.5,
        'position': 'random',
        'application_type': 'map',
        'artifacts_config': artifacts_variation(),
    }


def milios_map_edgecase():
    """
    Simulates an edge case where the watermark is placed at the top or bottom of the image and not fully visible / cropped off.
    """
    move_to_bottom = random.choice([0, -512 + 17]) # +17 to avoid moving the main watermark totally out of the image
    x = random.randint(383, 383 + 512) // 2
    y = (random.uniform(1393, 1431) + move_to_bottom) // 2
    return {
        'watermark_id': 'map_43',
        'occupancy': 0,
        'scale': 0.5,
        'position': (x, y),
        'application_type': 'map',
        'artifacts_config': artifacts_variation(),
    }


def milios_map_around_center():
    """
    Focusses on cases where the center of the watermark map (which differs a little from the rest of the watermark) is placed somewhere in the image.
    It is purposely possible to be cropped off to the left or right, but not to the top or bottom because that's handled by the edgecase.
    """
    
    x = random.randint(383, 383 + 512) // 2 # 512 is 2x the width of the base image => 256 after halving = exactly the width of the base image
    y = random.uniform(1390 - 512 + 150, 1390) // 2 # 75 is the height of the center feature in the watermark map
    return {
        'watermark_id': 'map_43',
        'occupancy': 0,
        'scale': 0.5,
        'position': (x, y),
        'application_type': 'map',
        'artifacts_config': artifacts_variation(),
    }


def milios_map_center():
    return {
        'watermark_id': 'map_43',
        'occupancy': 0,
        'scale': 0.5,
        'alpha': 0.66,
        'position': 'center',
        'application_type': 'map',
        'artifacts_config': ArtifactsConfig(
            alpha=0.66,
            intensity=1.5,
            kernel_size=11
        ),
    }


def get_watermark_variations():
    scale = random.uniform(0.75, 1.25)
    variants = []
    variants.append(logo_by_name('logo_ppco', scale=scale))     # 0
    variants.append(logo_by_name('logo_mr', scale=scale))       # 1
    variants.append(logo_by_name('logo_mreb', scale=scale))     # 2
    variants.append(logo_by_name('logo_mrlnb', scale=scale))    # 3
    variants.append(logo_milios())                              # 4
    variants.append(milios_map())                   # 5
    variants.append(milios_map_edgecase())          # 6
    variants.append(milios_map_around_center())     # 7
    variants.append(milios_map_center())            # 8

    weights = [
        1, # logo_ppco
        1, # logo_mr
        1, # logo_mreb
        1, # logo_mrlnb
        1, # logo_milios
        6, # milios_map
        2, # milios_map_edgecase
        2, # milios_map_around_center
        1, # milios_map_center
    ]

    # Fine-tuning weights
    weights = [
        0, # logo_ppco
        0, # logo_mr
        0, # logo_mreb
        0, # logo_mrlnb
        0, # logo_milios
        16, # milios_map
        4, # milios_map_edgecase
        4, # milios_map_around_center
        1, # milios_map_center
    ]

    if len(variants) != len(weights):
        raise ValueError("The number of watermark variations and their weights must be the same.")

    return variants, weights


def get_watermark_validation_variation():
    return milios_map_center()

