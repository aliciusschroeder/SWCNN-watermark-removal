import os

import numpy as np
import torch
import thop
from thop import profile
from torch.autograd import Variable
from torch import device as torchdevice
from torch.cuda import is_available as cuda_is_available

# from models import FFDNet, DnCNN, IRCNN, HN, FastDerainNet, DRDNet, EAFN
# from models import UNet, FFDNet, DnCNN, IRCNN, UNet_Atten_4, FastDerainNet, DRDNet, EAFN
from models import DRDNet

device = torchdevice("cuda" if cuda_is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
noise_sigma = 0

noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(1)]))
noise_sigma = Variable(noise_sigma)
noise_sigma = noise_sigma.to(device)
# net = UNet_Atten_4()  # 定义好的网络模型
# net = HN()
# net = EAFN()
# net = FastDerainNet(3, 48)
net = DRDNet(3,48)
net = net.to(device)
input = torch.randn(1, 3, 256, 256)
input = input.to(device)
flops, params = profile(net, (input,))
flops, params = thop.clever_format([flops, params], "%.3f")  # 提升结果可读性
print('flops: ', flops, 'params: ', params)
