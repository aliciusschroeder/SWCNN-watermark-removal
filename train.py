"""
Self-Supervised Watermark Cleaning Neural Network (SWCNN) Training Module

This module implements a training pipeline for a neural network designed to remove
watermarks from images using self-supervised learning techniques. It supports both
traditional supervised and self-supervised training approaches, with optional
perceptual loss using VGG16 features.

The training process includes:
- Dynamic watermark application with configurable parameters
- Support for multiple network architectures
- Flexible loss function selection (L1/L2)
- Tensorboard logging for training monitoring
- Periodic validation on a separate dataset
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # type: ignore

from dataset import Dataset
from models import HN
from utils.helper import get_config
from utils.train_preparation import load_froze_vgg16
from utils.validation import batch_PSNR
from utils.watermark import WatermarkManager, ArtifactsConfig


@dataclass
class TrainingConfig:
    """Configuration parameters for the SWCNN training process."""
    batch_size: int = 8
    num_layers: int = 17
    epochs: int = 100
    milestone: int = 30  # Learning rate decay epoch
    initial_lr: float = 1e-3
    watermark_alpha: float = 0.6
    model_output_path: str = "models/SWCNN"
    architecture: Literal["HN"] = "HN"
    loss_type: Literal["L1", "L2"] = "L1"
    self_supervised: bool = True
    use_perceptual_loss: bool = True
    gpu_id: str = "0"
    data_path: str = "data"

    @property
    def model_name(self) -> str:
        """Generate a descriptive model name based on configuration."""
        components = [
            self.architecture,
            "per" if self.use_perceptual_loss else "woper",
            self.loss_type,
            "n2n" if self.self_supervised else "n2c",
            f"alpha{self.watermark_alpha}"
        ]
        return "_".join(components)


class WatermarkCleaner:
    """Manages the training of watermark removal neural networks."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "cpu")
        print(f"Using device: {self.device.type}")
        self._setup_environment()
        self._init_components()

    def _setup_environment(self) -> None:
        """Configure CUDA environment and create output directories."""
        torch.cuda.set_device(int(self.config.gpu_id))
        Path(self.config.model_output_path).mkdir(parents=True, exist_ok=True)

    def _init_components(self) -> None:
        """Initialize model, optimizer, loss functions, and data loaders."""
        self.model = self._create_model()
        self.vgg_model = load_froze_vgg16()
        self.criterion = (nn.MSELoss(reduction='sum') if self.config.loss_type == "L2" 
                         else nn.L1Loss(reduction='sum'))
        self.criterion.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.config.initial_lr)
        self.writer = SummaryWriter(f"runs/{self.config.model_name}")
        
        self.watermark_manager = WatermarkManager(
            data_path=f"{self.config.data_path}/watermarks",
            swap_blue_red_channels=True
        )
        
        self._init_datasets()

    def _create_model(self) -> nn.Module:
        """Initialize and prepare the neural network model."""
        if self.config.architecture != "HN":
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
            
        model = nn.DataParallel(HN(), device_ids=[0]).to(self.device)
        return model

    def _init_datasets(self) -> None:
        """Initialize training and validation datasets."""
        self.train_dataset = Dataset(train=True, mode='color', 
                                   data_path=self.config.data_path)
        self.val_dataset = Dataset(train=False, mode='color', 
                                 data_path=self.config.data_path)
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

    def _apply_watermark(
        self, 
        img: torch.Tensor, 
        seed: Optional[int] = None,
        variant_choice: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """Apply a random watermark to the input image.
        
        Args:
            img: Input image tensor
            seed: Random seed for reproducible watermark application
            variant_choice: Specific watermark variant to use (if None, chosen randomly)
            
        Returns:
            Tuple of (watermarked image, variant choice used)
        """
        variants = [
            {
                'watermark_id': 'logo_ppco',
                'occupancy': 0,
                'scale': 1.0,
                'alpha': random.uniform(0.33, 1),
                'position': 'random',
                'application_type': 'stamp',
                'same_random_wm_seed': seed if seed is not None else 0,
                'self_supervision': True,
            },
            {
                'watermark_id': 'map_43',
                'occupancy': 0,
                'scale': 0.5,
                'position': 'random',
                'application_type': 'map',
                'artifacts_config': ArtifactsConfig(
                    alpha=random.uniform(0.44, 0.88),
                    intensity=random.uniform(1.00, 2.00),
                    kernel_size=random.choice([7, 11, 15]),
                ),
                'self_supervision': True,
            }
        ]
        
        if variant_choice is None:
            variant_choice = random.randint(0, len(variants)-1)
            
        watermarked_img = self.watermark_manager.add_watermark_generic(
            img, 
            **variants[variant_choice]
        )
        return watermarked_img, variant_choice

    def _adjust_learning_rate(self, epoch: int) -> None:
        """Adjust learning rate based on epoch."""
        current_lr = (self.config.initial_lr 
                     if epoch < self.config.milestone 
                     else self.config.initial_lr / 10.)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def _train_step(
        self, 
        img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Execute a single training step.
        
        Args:
            img: Input image batch
            
        Returns:
            Tuple of (output image, target image, loss value, PSNR value)
        """
        self.model.train()
        self.optimizer.zero_grad()

        random_seed = random.getrandbits(128)
        watermarked_img, variant_choice = self._apply_watermark(img, random_seed)

        target_img = (self._apply_watermark(img, random_seed, variant_choice)[0]
                     if self.config.self_supervised else img)

        watermarked_img = watermarked_img.to(self.device)
        target_img = target_img.to(self.device)

        output = self.model(watermarked_img)
        
        # Calculate losses
        output_features = self.vgg_model(output)
        target_features = self.vgg_model(target_img)
        
        reconstruction_loss = self.criterion(output, target_img) / watermarked_img.size()[0] * 2
        
        if self.config.use_perceptual_loss:
            perceptual_loss = (0.024 * self.criterion(output_features, target_features) 
                              / (target_features.size()[0] / 2))
            total_loss = reconstruction_loss + perceptual_loss
        else:
            total_loss = reconstruction_loss

        total_loss.backward()
        self.optimizer.step()

        # Calculate PSNR for monitoring
        with torch.no_grad():
            output = torch.clamp(self.model(watermarked_img), 0., 1.)
            psnr = batch_PSNR(output, img.to(self.device), 1.)

        return output, target_img, total_loss.item(), psnr

    def validate(self, epoch: int) -> float:
        """Run validation on the validation dataset.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Average PSNR value across validation set
        """
        self.model.eval()
        total_psnr = 0

        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                img = torch.unsqueeze(self.val_dataset[i], 0)
                
                # Ensure dimensions are multiples of 32
                _, _, w, h = img.shape
                w, h = (int(dim // 32 * 32) for dim in (w, h))
                img = img[:, :, :w, :h]
                random.seed(i)
                watermarked_img, _ = self._apply_watermark(img)
                img, watermarked_img = img.to(self.device), watermarked_img.to(self.device)
                
                output = torch.clamp(self.model(watermarked_img), 0., 1.)
                total_psnr += batch_PSNR(output, img, 1.)

        avg_psnr = total_psnr / len(self.val_dataset)
        self.writer.add_scalar("PSNR/val", avg_psnr, epoch + 1)
        return avg_psnr
    
    def _save_model(self, epoch: int) -> None:
        """Save the current model state to disk."""
        pathname = Path(self.config.model_output_path) / f"{self.config.model_name}_{epoch:03}.pth"
        torch.save(
            {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1
        }, pathname)

    def train(self) -> None:
        """Execute the complete training pipeline."""
        print(f'Training on {len(self.train_dataset)} samples')
        
        step = 0
        for epoch in range(self.config.epochs):
            self._adjust_learning_rate(epoch)
            
            for i, img in enumerate(self.train_loader):
                output, target, loss, psnr = self._train_step(img)
                
                print(f"[epoch {epoch + 1}][{i + 1}/{len(self.train_loader)}] "
                      f"loss: {loss:.4f} PSNR_train: {psnr:.4f}")
                
                if step % 10 == 0:
                    self.writer.add_scalar("PSNR/train", psnr, step)
                    self.writer.add_scalar("Loss/train", loss, step)
                step += 1

            # Save model and validate after each epoch
            self._save_model(epoch)
            random.seed("validation")
            val_psnr = self.validate(epoch)
            random.seed()
            print(f"\n[epoch {epoch + 1}] PSNR_val: {val_psnr:.4f}")

        self.writer.close()


def main():
    """Entry point for training the watermark removal model."""
    yaml_config = get_config('configs/config.yaml')
    config = TrainingConfig(
        model_output_path=yaml_config['train_model_out_path_SWCNN'],
        data_path=yaml_config['data_path'],
        batch_size=8,
    )
    
    trainer = WatermarkCleaner(config)
    trainer.train()


if __name__ == "__main__":
    main()