from models import VGG16


import torch.nn as nn
import torchvision.models as models
from torch import device as torchdevice
from torch.cuda import is_available as cuda_is_available
from torchvision.models import VGG16_Weights


def load_froze_vgg16():
    # Fine-tuning
    model_pretrain_vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)

    # Load VGG16
    net_vgg = VGG16()
    model_dict = net_vgg.state_dict()
    pretrained_dict = model_pretrain_vgg.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Load Parameters
    net_vgg.load_state_dict(pretrained_dict)

    for child in net_vgg.children():
        for p in child.parameters():
            p.requires_grad = False
    device_ids = [0]

    device = torchdevice('cuda' if cuda_is_available() else 'cpu')
    model_vgg = nn.DataParallel(net_vgg, device_ids=device_ids).to(device)
    return model_vgg