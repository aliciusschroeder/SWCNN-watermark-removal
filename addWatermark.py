import random

import numpy as np
import torch
from PIL import Image

from utils import add_watermark_noise_generic

# import cv2
# import matplotlib.pyplot as plt


# angle = random.randint(-90,90)
# scale = random.random()
# scale *=2
# img = Image.open("water.png")
# img = img.rotate(angle,expand = 1)
# w,h = img.size
# img = img.resize((int(w*scale),int(h*scale)))
# img_three = Image.open("three.jpg")

# out = Image.alpha_composite(img_three,img.resize(img_three.size))
# img_three.paste(img,(1000,3000),img)
# plt.imshow(img_three)
#
# plt.show()
#     img = img.copy()
#     TRANSPARENCY = random.randint(28, 82)
#
#     image = Image.fromarray(img)
#     watermark = Image.open('./水印.png')  # 水印路径
#     # cv2.imshow(watermark)
#     plt.imshow(watermark)
#     if watermark.mode != 'RGBA':
#         alpha = Image.new('L', watermark.size, 255)
#         watermark.putalpha(alpha)
#
#     random_X = random.randint(-750, 45)
#     random_Y = random.randint(-500, 30)
#
#     paste_mask = watermark.split()[3].point(lambda i: i * TRANSPARENCY / 100.)
#     image.paste(watermark, (random_X, random_Y), mask=paste_mask)

def add_watermark_noise(noise, occupancy=50):
    return add_watermark_noise_generic(
        img_train=noise,
        occupancy=occupancy,
        standalone=True
    )


if __name__ == '__main__':
    noise = torch.FloatTensor(torch.ones((128, 3, 200, 200)).size()).normal_(mean=0, std=50 / 255.)
    max_noise = torch.max(noise)
    min_noise = torch.min(noise)
    noise = (noise - min_noise)/(max_noise-min_noise)
    add_watermark_noise(noise[0])
