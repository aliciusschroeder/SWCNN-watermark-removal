import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch import device as torchdevice
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


from dataset import Dataset
from models import HN
from utils.helper import get_config
from utils.train_preparation import load_froze_vgg16
from utils.validation import batch_PSNR
from utils.watermark import WatermarkManager, ArtifactsConfig

parser = argparse.ArgumentParser(description="SWCNN")
config = get_config('configs/config.yaml')
parser.add_argument("--batchSize", type=int, default=8, 
                    help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, 
                    help="Number of total layers(DnCNN)")
parser.add_argument("--epochs", type=int, default=100, 
                    help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, 
                    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, 
                    help="Initial learning rate")
parser.add_argument("--alpha", type=float, default=0.6, 
                    help="The opacity of the watermark")
parser.add_argument("--outf", type=str, default=config['train_model_out_path_SWCNN'], 
                    help='path of model')
parser.add_argument("--net", type=str, default="HN", 
                    help='Network used in training')
parser.add_argument("--loss", type=str, default="L1", 
                    help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", 
                    help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=str, default="True", 
                    help='Whether to use perception network')
parser.add_argument("--GPU_id", type=str, default="0", 
                    help='GPU_id')
opt = parser.parse_args()

device = torchdevice("cuda" if cuda_is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_id

if opt.PN == "True":
    model_name_1 = "per"
else:
    model_name_1 = "woper"
if opt.loss == "L1":
    model_name_2 = "L1"
else:
    model_name_2 = "L2"
if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"
tensorboard_name = opt.net + model_name_1 + model_name_2 + model_name_3 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"
print()


def main():
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, mode='color', data_path=config['data_path'])
    dataset_val = Dataset(train=False, mode='color', data_path=config['data_path'])
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)  # 4
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # load network
    if opt.net == "HN":
        net = HN()
    else:
        assert False

    writer = SummaryWriter("runs/" + tensorboard_name)

    model_vgg = load_froze_vgg16()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)

    # load loss function
    if opt.loss == "L2":
        criterion = nn.MSELoss(reduction='sum')
    else:
        criterion = nn.L1Loss(reduction='sum')

    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    step = 0

    wmm = WatermarkManager(
        data_path = config['data_path'],
        swap_blue_red_channels=True,
        debug=True,
    )

    artifacts_config = ArtifactsConfig()

    def add_watermark_train(img, seed = None):
        result =  wmm.add_watermark_generic(
            img,
            occupancy=0,
            scale=(0.45, 0.55),
            # alpha=opt.alpha, <- Alpha not set as it's overwritten by artifact settings
            position = 'random',
            application_type='map',
            artifacts_config=artifacts_config,
            random_seed=seed,
            self_supervision=True,
        )
        return result

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            
            random_seed = random.getrandbits(128)
            # imgn_train = add_watermark_generic(img_train, 0, True, random_img, alpha=opt.alpha)
            imgn_train = add_watermark_train(img_train, random_seed)
            if opt.self_supervised == "True":
                imgn_train_2 = add_watermark_train(img_train, random_seed)
            else:
                imgn_train_2 = img_train

            imgn_train = torch.Tensor(imgn_train)
            imgn_train_2 = torch.Tensor(imgn_train_2)
            img_train, imgn_train = img_train.to(device), imgn_train.to(device)
            imgn_train_2 = imgn_train_2.to(device)
            if opt.net == "FFDNet":
                noise_sigma = 0 / 255.
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for _ in range(img_train.shape[0])]))
                noise_sigma = Variable(noise_sigma) # TODO: check if it needs to track gradients, ensure requires_grad=True is set on the tensor before removing Variable()
                noise_sigma = noise_sigma.to(device)
                out_train = model(imgn_train, noise_sigma)
            else:
                noise_sigma = None
                out_train = model(imgn_train)
            feature_out = model_vgg(out_train)
            feature_img = model_vgg(imgn_train_2)

            if opt.PN == "True":
                loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                    0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
            else:
                loss = (1.0 * criterion(out_train, img_train) / imgn_train.size()[
                    0] * 2) + (0.0 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            if opt.net == "FFDNet":
                out_train = torch.clamp(model(imgn_train, noise_sigma), 0., 1.)
            else:
                out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            step += 1
            if step % 10 == 0:
                writer.add_scalar("PSNR", psnr_train, step)
                writer.add_scalar("loss", loss.item(), step)

        ## the end of each epoch
        model.eval()
        # Save the trained network parameters
        torch.save(model.state_dict(), os.path.join(opt.outf, model_name))
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                # Cut the picture into multiples of 32
                _, _, w, h = img_val.shape
                w = int(int(w / 32) * 32)
                h = int(int(h / 32) * 32)
                img_val = img_val[:, :, 0:w, 0:h]
                imgn_val = add_watermark_train(img_val)
                img_val = torch.Tensor(img_val)
                imgn_val = torch.Tensor(imgn_val)
                with torch.no_grad():
                    img_val, imgn_val = img_val.to(device), imgn_val.to(device)
                if opt.net == "FFDNet":
                    noise_sigma = 0 / 255.
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(img_val.shape[0])]))
                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.to(device)
                    out_val = torch.clamp(model(imgn_val, noise_sigma), 0., 1.)
                else:
                    out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            writer.add_scalar("PSNR_val", psnr_val, epoch + 1)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
    writer.close()


if __name__ == "__main__":
    main()
