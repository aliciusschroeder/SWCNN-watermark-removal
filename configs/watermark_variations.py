"""
Customized watermark variations for the watermarking process. Make sure to write your own watermark variations.
"""

import random
from typing import Union
from utils.watermark import ArtifactsConfig


def artifacts_variation():
    return ArtifactsConfig(
        alpha=1.00, #0.66 #random.uniform(0.55, 0.77),
        intensity=random.uniform(0.50, 1.50),
        kernel_size=random.choice([5, 6, 7]),
    )

def base_stamp(name: str, scale: float = 1.0, alpha = 0.5):
    return {
        'watermark_id': name,
        'occupancy': 0,
        'scale': scale,
        'alpha': alpha,
        'position': 'random',
        'application_type': 'stamp',
    }

def logo_by_name(name: str, scale: float = 1.0):
    stamp = base_stamp(name, scale)
    stamp['alpha'] = random.uniform(0.33, 1)
    return stamp


def logo_milios(id: str = 'logo_milios'):
    stamp = base_stamp(id)
    stamp['scale'] = random.uniform(0.63, 0.93) # => 0.78 +- 0.15
    stamp['alpha'] = random.uniform(0.33, 1)    # => 0.67 +- 0.33
    return stamp

def base_map(map: Union[str, int] = 43, scale: float = 0.5, alpha = 0.5):
    return {
        'watermark_id': f'{map}',
        'occupancy': 0,
        'scale': scale,
        'application_type': 'map',
        'alpha': alpha,
        'artifacts_config': artifacts_variation(),
    }

def milios_map(id: Union[str, int] = 43, scale: float = 0.5, alpha = 0.5):
    map = base_map(id, scale, alpha)
    map['position'] = 'random'
    return map


def milios_map_edgecase(id: str = 'map_43', scale: float = 0.5, alpha = 0.5):
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


def milios_map_around_center(id: str = 'map_43', scale: float = 0.5, alpha = 0.5):
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


def milios_map_center(id: str = 'map_43', scale: float = 0.5, alpha = 0.5):
    map = base_map(id, scale=scale, alpha=alpha)
    map['position'] = 'center'
    return map

def train_map(xtrabright: bool = False, continous_alpha: bool = False, black: bool = False, version: str = ''):
    id = f'map_train{version}_'
    id += 'bright' if xtrabright else 'normal'
    id += '_black' if black else ''
    alpha = (random.uniform(0.32, 0.8) + 0.1) if continous_alpha else random.choice([0.5, 0.66, 0.9])
    map = base_map(id, scale=1.0, alpha=alpha)
    del map['artifacts_config']
    map['position'] = 'random'
    return map

def train_map2(v: str = ''):
    id = f'map_train{v}'
    alpha = random.uniform(0.55, 0.77)
    map = base_map(id, scale=1.0, alpha=alpha)
    del map['artifacts_config']
    map['position'] = 'random'
    return map

# val_relevant_methods = [13, 9, 9, 9] # Stage I
# val_relevant_methods = [10, 11, 12, 13, 14] # Stage II
# val_relevant_methods = [15, 15, 16, 17, 17, 18] # Stage III
# val_relevant_methods = [22, 23, 24] # Stage IV
# val_relevant_methods = [19] # Stage V
val_relevant_methods = [25, 26, 27, 28, 29] # Stage Vb (epoch 011)

def get_watermark_variations():
    scale = random.uniform(0.75, 1.25)
    easy_alpha = random.choice([0.8, 1])
    easy_scale = random.choice([0.5, 0.75, 1])

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
    variants.append(logo_milios('logo_milios_white'))                                           # 9
    variants.append(milios_map('map_white', alpha=easy_alpha, scale=easy_scale))                # 10
    variants.append(milios_map('map_white_dull', alpha=easy_alpha, scale=easy_scale))           # 11
    variants.append(milios_map_edgecase('map_white', alpha=easy_alpha, scale=easy_scale))       # 12
    variants.append(milios_map_around_center('map_white', alpha=easy_alpha, scale=easy_scale))  # 13
    variants.append(milios_map_center('map_white', alpha=easy_alpha, scale=easy_scale))         # 14
    variants.append(train_map())                            # 15
    variants.append(train_map(xtrabright=True))             # 16
    variants.append(train_map(black=True))                  # 17
    variants.append(train_map(xtrabright=True, black=True)) # 18
    variants.append(train_map(version='2'))                 # 19
    variants.append(train_map(continous_alpha=True, version='2'))   # 20
    variants.append(train_map(continous_alpha=True, version='3'))   # 21
    variants.append(train_map(continous_alpha=True, version='4'))   # 22
    variants.append(train_map(continous_alpha=True, version='5'))   # 23
    variants.append(train_map(continous_alpha=True, version='6'))   # 24
    variants.append(train_map2(v='6'))  # 25
    variants.append(train_map2(v='6a')) # 26
    variants.append(train_map2(v='7'))  # 27
    variants.append(train_map2(v='7g')) # 28
    variants.append(train_map2(v='7alien')) # 29

    weights = [0]*len(variants)

    # Stage Vb (epoch 011)
    weights[25] = 1 # train_map2 - version 6
    weights[26] = 1 # train_map2 - version 6a
    weights[27] = 1 # train_map2 - version 7
    weights[28] = 1 # train_map2 - version 7g
    weights[29] = 1 # train_map2 - version 7alien


    """ # Stage V
    weights[20] = 1 # train_map - version 2
    weights[21] = 1 # train_map - version 3
    weights[22] = 1 # train_map - version 4
    weights[23] = 1 # train_map - version 5
    weights[24] = 1 # train_map - version 6
 """
    
    """ # Stage IV
    weights[19] = 1 # train_map - version 2 """


    """ # Stage I
    weights = [
        # 0 - 4
        0,  # logo_ppco
        0,  # logo_mr
        0,  # logo_mreb
        0,  # logo_mrlnb
        0,  # logo_milios
        # 5 - 8
        0,  # milios_map
        0,  # milios_map_edgecase
        0,  # milios_map_around_center
        0,  # milios_map_center
        # 9 - 14
        10,  # logo_milios - white
        0, # milios_map - white
        0, # milios_map - white_dull
        1,  # milios_map_edgecase - white
        3,  # milios_map_around_center - white
        1,  # milios_map_center - white
        0,  # train_map

    ] """

    """ # Stage II
    weights = [
        # 0 - 4
        0,  # logo_ppco
        0,  # logo_mr
        0,  # logo_mreb
        0,  # logo_mrlnb
        0,  # logo_milios
        # 5 - 8
        0,  # milios_map
        0,  # milios_map_edgecase
        0,  # milios_map_around_center
        0,  # milios_map_center
        # 9 - 14
        2,  # logo_milios - white
        3, # milios_map - white
        3, # milios_map - white_dull
        3,  # milios_map_edgecase - white
        3,  # milios_map_around_center - white
        3,  # milios_map_center - white
        0,  # train_map
    ] """

    """ # Stage III
    weights[15] = 2 # train_map
    weights[16] = 1 # train_map - bright
    weights[17] = 2 # train_map - black
    weights[18] = 1 # train_map - stronger black """


    if len(variants) != len(weights):
        raise ValueError(
            "The number of watermark variations and their weights must be the same.")

    return variants, weights


def get_watermark_validation_variation():
    return milios_map_center()
