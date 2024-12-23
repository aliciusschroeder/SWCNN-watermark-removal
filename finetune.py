"""
Fine-tuning Module for SWCNN Watermark Removal

This module implements a fine-tuning pipeline for the SWCNN model using real-world 
paired data. It leverages a pre-trained model and a dataset of clean and 
watermarked image pairs to further refine the model's performance.

The fine-tuning process includes:
- Loading a pre-trained model checkpoint
- Preparing a dataset from an HDF5 file with paired clean and watermarked images
- Splitting the dataset into training and validation sets
- Fine-tuning the model using the training set
- Evaluating performance on the validation set
- Logging training progress to Tensorboard
"""

import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import PairedDataset
from models import HN
from utils.validation import batch_PSNR

# --- Configuration (Hardcoded/no config file for Fine-tuning) ---
PRETRAINED_MODEL_PATH = "output/models/finetune/base.pth"  # Path to the pre-trained model checkpoint
# PRETRAINED_MODEL_PATH = "output/models/finetune/HN_finetuned_epoch_075_best.pth"  # Path to the pretraining model checkpoint to resume
LOG_DIR = "output/runs-finetune"  # Directory for Tensorboard logs

START_EPOCH = 100
EPOCHS = 300
INITIAL_LR = 8e-5  # Lower learning rate for fine-tuning
MIN_LR = 0  # Minimum learning rate for the scheduler
BATCH_SIZE = 2
ITERATIONS_PER_RESTART = 50 # Iterations until restart for the CosineAnnealingWarmRestarts scheduler

GPU_ID = "0"
ARCHITECTURE = "HN"
LOSS_TYPE = "L1"  # Loss function type
USE_PERCEPTUAL_LOSS = True  # Whether to use perceptual loss

SAVE_CHECKPOINT_NTH_EPOCH = 1
SAVE_IMAGES_NTH_EPOCH = 5
TEST_NTH_EPOCH = 1
PRINT_PER_BATCH = False
PRINT_PER_EPOCH = True

USE_VALIDATION_SPLIT = False  # If False, a separate validation dataset must be present (test_color.h5)
VALIDATION_SPLIT = 0.10  # Percentage of data to use for validation
SPLIT_SEED = 6942  # Seed for dataset splitting
random.seed(SPLIT_SEED)

RESUME = False  # Whether to resume training from a Fine-Tuning checkpoint
RESUME_WRITER = False  # Whether to resume Tensorboard writer from a previous run
WRITER_RESUME_DIR = "None"  # Directory of the previous run to resume

# Encapsulate functionality into a FineTune class
class FineTune:
    def __init__(self):
        # Setup Environment
        torch.cuda.set_device(int(GPU_ID))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device.type}")
        
        # Create Output Directories
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs("output/models/finetune", exist_ok=True)
        
        # Load Pre-trained Model
        print(f"Loading pre-trained model from: {PRETRAINED_MODEL_PATH}")
        self.model = nn.DataParallel(HN(), device_ids=[0]).to(self.device)
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Initialize Tensorboard Writer
        current_time = datetime.now().strftime("%y-%m-%d-%H-%M")
        if RESUME_WRITER:
            self.writer = SummaryWriter(f"{LOG_DIR}/{WRITER_RESUME_DIR}", purge_step=int(checkpoint['global_step']))
        else:
            self.writer = SummaryWriter(f"{LOG_DIR}/{ARCHITECTURE}-finetune-{current_time}")
        
        # Prepare Dataset
        print(f"Loading fine-tuning dataset from: data/finetune_color.h5")
        full_dataset = PairedDataset(purpose='finetune', mode='color', data_path="data")
        if not USE_VALIDATION_SPLIT:
            val_dataset = PairedDataset(purpose='test', mode='color', data_path="data")
            train_dataset = full_dataset
        else:
            num_validation_samples = int(len(full_dataset) * VALIDATION_SPLIT)
            num_training_samples = len(full_dataset) - num_validation_samples
            train_dataset, val_dataset = random_split(
                full_dataset, [num_training_samples, num_validation_samples], generator=torch.Generator().manual_seed(SPLIT_SEED)
            )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        
        # Loss Function and Optimizer
        self.criterion = nn.L1Loss(reduction='sum').to(
            self.device) if LOSS_TYPE == "L1" else nn.MSELoss(reduction='sum').to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=INITIAL_LR)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=ITERATIONS_PER_RESTART, eta_min=MIN_LR)
        self.scaler = GradScaler()

        if RESUME:
            print("Resuming Fine-Tuning from checkpoint...")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step'] + 1
            print(f"Resuming will begin at epoch {self.start_epoch}, global step {self.global_step}")
        else:
            self.start_epoch = START_EPOCH
            self.global_step = 0
        
        # Load VGG16 for Perceptual Loss
        if USE_PERCEPTUAL_LOSS:
            from utils.train_preparation import load_froze_vgg16
            self.vgg_model = load_froze_vgg16()
        else:
            self.vgg_model = None

    def train(self):
        print("Starting fine-tuning...")
        best_val_psnr = 0.0
        global_step = self.global_step

        for epoch in range(self.start_epoch, EPOCHS):
            self.model.train()
            epoch_losses = {'reconstruction': 0.0, 'perceptual': 0.0, 'total': 0.0}
            epoch_psnr = 0.0

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            for batch_idx, (clean_img, watermarked_img) in enumerate(self.train_loader):
                clean_img, watermarked_img = clean_img.to(self.device), watermarked_img.to(self.device)

                self.optimizer.zero_grad()

                with autocast(device_type='cuda'):
                    output = self.model(watermarked_img)
                    reconstruction_loss = self.criterion(output, clean_img) / clean_img.size(0)

                    if USE_PERCEPTUAL_LOSS and self.vgg_model is not None:
                        output_features = self.vgg_model(output)
                        target_features = self.vgg_model(clean_img)
                        perceptual_loss = 0.024 * self.criterion(output_features, target_features) / (target_features.size(0) / 2)
                        total_loss = reconstruction_loss + perceptual_loss
                        epoch_losses['perceptual'] += perceptual_loss.item()
                    else:
                        total_loss = reconstruction_loss

                    epoch_losses['reconstruction'] += reconstruction_loss.item()
                    epoch_losses['total'] += total_loss.item()

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Calculate PSNR
                with torch.no_grad():
                    psnr = batch_PSNR(output, clean_img, 1.)
                    epoch_psnr += psnr

                # Log training losses and PSNR
                self.writer.add_scalar("Loss_Train/Total", total_loss.item(), global_step)
                self.writer.add_scalar("Loss_Train/Reconstruction", reconstruction_loss.item(), global_step)
                if USE_PERCEPTUAL_LOSS:
                    self.writer.add_scalar("Loss_Train/Perceptual", perceptual_loss.item(), global_step) # type: ignore
                self.writer.add_scalar("PSNR_Train", psnr, global_step)

                # Log sample images
                if batch_idx == 0 and (epoch + 1) % SAVE_IMAGES_NTH_EPOCH == 0:
                    img_grid = vutils.make_grid(
                        torch.cat([watermarked_img, output, clean_img]),
                        nrow=BATCH_SIZE,
                        normalize=True,
                    )
                    self.writer.add_image(f"Images_Train/Epoch_{epoch + 1}", img_grid, global_step)

                if PRINT_PER_BATCH:
                    print(
                        f"Epoch: {epoch + 1}/{EPOCHS}, Batch: {batch_idx + 1}/{len(self.train_loader)}, "
                        f"Loss: {total_loss.item():.4f}, PSNR: {psnr:.4f}"
                    )

                global_step += 1

            self.scheduler.step()

            # Log epoch-level training metrics
            num_train_batches = len(self.train_loader)
            for loss_name, loss_sum in epoch_losses.items():
                avg_loss = loss_sum / num_train_batches
                self.writer.add_scalar(f"Epoch_Metrics_Train/{loss_name}_loss_avg", avg_loss, epoch + 1)
            avg_epoch_psnr = epoch_psnr / num_train_batches
            self.writer.add_scalar("Epoch_Metrics_Train/PSNR_avg", avg_epoch_psnr, epoch + 1)

            avg_val_psnr = 0.0
            # --- Validation ---
            if (epoch + 1) % TEST_NTH_EPOCH == 0:
                self.model.eval()
                val_psnr = 0.0
                with torch.no_grad():
                    for clean_img, watermarked_img in self.val_loader:
                        clean_img, watermarked_img = clean_img.to(self.device), watermarked_img.to(self.device)
                        output = self.model(watermarked_img)
                        val_psnr += batch_PSNR(output, clean_img, 1.)

                avg_val_psnr = val_psnr / len(self.val_loader)
                self.writer.add_scalar("Epoch_Metrics_Val/PSNR_avg", avg_val_psnr, epoch + 1)

                if PRINT_PER_EPOCH:
                    print(f"Epoch: {epoch + 1}, Validation PSNR: {avg_val_psnr:.4f}")

            best_so_far = avg_val_psnr > best_val_psnr
            # Save model checkpoint periodically and when best so far
            if (epoch + 1) % SAVE_CHECKPOINT_NTH_EPOCH == 0 or best_so_far:
                if best_so_far:
                    best_val_psnr = avg_val_psnr


                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step
                    },
                    f"output/models/finetune/{ARCHITECTURE}_finetuned_epoch_{(epoch + 1):03d}"+
                    f"{"_best" if best_so_far else ""}.pth"
                )

        # --- Close Tensorboard Writer ---
        self.writer.close()
        print("Fine-tuning completed.")

def main():
    finetuner = FineTune()
    finetuner.train()

if __name__ == '__main__':
    main()