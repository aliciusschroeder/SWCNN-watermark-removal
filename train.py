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

# TODO(medium): Implement validation loss calculation
# TODO(medium): Look out for possible performance improvements in the training loop (partly done)

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import random

import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
from torch import autocast, GradScaler # Mixed precision training can improve performance. If it causes problems, remove GradScaler/autocast related lines
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from configs.tensorboard import TensorBoardConfig
from configs.training import TrainingConfig, ResumeOptions
from configs.watermark_variations import get_watermark_variations, val_relevant_methods
from dataset import Dataset, PairedDataset
from models import HN
from utils.helper import get_config
from utils.train_preparation import load_froze_vgg16
from utils.validation import batch_PSNR
from utils.watermark import WatermarkManager

PRINT_DURING_TRAINING = False

class WatermarkCleaner:
    """Manages the training of watermark removal neural networks."""

    def __init__(
            self, 
            config: TrainingConfig, 
            tensorboard_config: TensorBoardConfig = TensorBoardConfig(),
            resume_options: Optional[ResumeOptions] = None
        ):
        self.start_epoch = 0
        self.global_step = 0
        self.config = config
        self.tb_config = tensorboard_config
        self.resume_options = resume_options
        self.watermark_variants, self.watermark_weights = get_watermark_variations()
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "cpu")
        print(f"Using device: {self.device.type}")
        self._setup_environment()
        self._init_components()
        self._setup_tensorboard()


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

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=50, eta_min=0.0)
        # self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=1)
        
        self.scaler = GradScaler() # Remove this line if not using mixed precision training
        
        self.watermark_manager = WatermarkManager(
            data_path=f"{self.config.data_path}/watermarks",
            swap_blue_red_channels=False,
        )
        
        self._init_datasets()

        if self.resume_options is not None:
            self._load_checkpoint()


    def _load_checkpoint(self) -> None:
        """Load model and optimizer state from a checkpoint."""
        if self.resume_options is None:
            raise ValueError("No resume options provided")
        print(f"Resuming from checkpoint: {self.resume_options.checkpoint_filepath}")
        checkpoint = torch.load(self.resume_options.checkpoint_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step'] + 1
        
        print(f"Resuming training from epoch {self.start_epoch + 1} at step {self.global_step}")
                                              

    def _setup_tensorboard(self) -> None:
        """Initialize Tensorboard logging for training monitoring."""
        current_time = datetime.now().strftime("%y-%m-%d-%H-%M")
        log_dir = f"{self.tb_config.log_dir}/{self.config.model_name}-{current_time}"
        purge_step = None
        if self.resume_options is not None:
            if self.resume_options.log_dir is not None:
                log_dir = self.resume_options.log_dir
                print(f"Resuming Tensorboard logs from {log_dir}")
                if self.global_step > 0:
                    purge_step = self.global_step
                    print(f"And purging logs after step {purge_step}")
        
        self.writer = SummaryWriter(log_dir, purge_step=purge_step)

        if self.start_epoch == 0 and self.tb_config.log_model_architecture:
            # Log model architecture
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            self.writer.add_graph(self.model, dummy_input)

        if self.tb_config.log_hyperparameters:
            # Log hyperparameters
            hparams = {
                "batch_size": self.config.batch_size,
                "num_layers": self.config.num_layers,
                "epochs": self.config.epochs,
                "milestone": self.config.milestone,
                "initial_lr": self.config.initial_lr,
                "architecture": self.config.architecture,
                "loss_type": self.config.loss_type,
                "self_supervised": self.config.self_supervised,
                "use_perceptual_loss": self.config.use_perceptual_loss,
                "gpu_id": self.config.gpu_id,
                "data_path": self.config.data_path,
            }
            self.writer.add_hparams(hparams, {})


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
        self.test_dataset = PairedDataset(purpose='test',
                                        mode='color',
                                        data_path=self.config.data_path)
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            # os.cpu_count() // 2 seems like a good starting point
            num_workers=16,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

    def freeze_block(self, block: nn.Module, freeze: bool) -> None:
        """Freeze or unfreeze a block of the model.

        Args:
            block: The block of the model to freeze or unfreeze
            freeze: Whether to freeze the block

        Returns:
            None

        Example usage (freezing block 1.0):
            ```
            self.freeze_block(self.model.module._block1[0], freeze=True)
            ```      
        """
        for param in block.parameters():
            param.requires_grad = not freeze


    def _process_watermark_application(
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
        
        if variant_choice is None:
            variant_choice = random.choices(range(len(self.watermark_variants)), weights=self.watermark_weights, k=1)[0]
            
        watermarked_img = self.watermark_manager.add_watermark_generic(
            img,
            self_supervision=self.config.self_supervised,
            same_random_wm_seed=seed,
            **self.watermark_variants[variant_choice]
        )
        return watermarked_img, variant_choice


    def _adjust_learning_rate(self, epoch: int) -> None:
        """Adjust learning rate based on epoch."""
        current_lr = (self.config.initial_lr 
                     if epoch < self.config.milestone 
                     else self.config.initial_lr / 10.)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    
    def log_learning_rate(self, epoch: int) -> None:
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        # Don't add one to epoch here as it's called at the start of the epoch
        self.writer.add_scalar('Learning_Rate', current_lr, epoch) 


    def _train_step(
        self, 
        img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], float]:
        """Execute a single training step.
        
        Args:
            img: Input image batch
            
        Returns:
            Tuple of (output image, target image, loss value, PSNR value)
        """
        self.model.train()
        self.optimizer.zero_grad()

        random_seed = random.getrandbits(128)
        watermarked_img, variant_choice = self._process_watermark_application(img, random_seed)

        target_img = (self._process_watermark_application(img, random_seed, variant_choice)[0]
                     if self.config.self_supervised else img)

        watermarked_img = watermarked_img.to(self.device)
        target_img = target_img.to(self.device)

        # Use GradScaler/autocast for mixed precision training. Remove context manager if causing issues
        with autocast(device_type='cuda'): 
            output = self.model(watermarked_img)
            
            # Calculate and track individual losses
            losses = {}

            # Reconstruction loss
            reconstruction_loss = self.criterion(output, target_img) / watermarked_img.size()[0] * 2
            losses['reconstruction'] = reconstruction_loss.item()
            
            if self.config.use_perceptual_loss:
                output_features = self.vgg_model(output)
                target_features = self.vgg_model(target_img)
                perceptual_loss = (0.024 * self.criterion(output_features, target_features) 
                                / (target_features.size()[0] / 2))
                losses['perceptual'] = perceptual_loss.item()
                total_loss = reconstruction_loss + perceptual_loss
            else:
                total_loss = reconstruction_loss

            losses['total'] = total_loss.item()

        # total_loss.backward() # Use this line if not using mixed precision training
        self.scaler.scale(total_loss).backward()

        # Log parameter gradients before clipping
        if self.writer is not None and self.tb_config.log_parameter_histograms:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients_before_clip/{name}", 
                                              param.grad, self.global_step)

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # self.optimizer.step() # Use this line if not using mixed precision training
        self.scaler.step(self.optimizer)

        # Calculate PSNR
        with torch.no_grad():
            output = torch.clamp(self.model(watermarked_img), 0., 1.)
            psnr = batch_PSNR(output, img.to(self.device), 1.)

        self.scaler.update() # Remove this line if not using mixed precision training

        return output, target_img, losses, psnr


    def validate(self, epoch: int) -> float:
        """Run validation on the validation dataset.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Average PSNR value across validation set
        """
        self.model.eval()
        total_psnr = 0
        val_losses = {'reconstruction': 0.0, 'perceptual': 0.0, 'total': 0.0}

        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                for variant in val_relevant_methods:
                    img = torch.unsqueeze(self.val_dataset[i], 0)
                    
                    # Ensure dimensions are multiples of 32
                    _, _, w, h = img.shape
                    w, h = (int(dim // 32 * 32) for dim in (w, h))
                    img = img[:, :, :w, :h]
                    random.seed(i+variant)
                    watermarked_img, _ = self._process_watermark_application(img, seed=i, variant_choice=variant)
                    img, watermarked_img = img.to(self.device), watermarked_img.to(self.device)
                    
                    output = torch.clamp(self.model(watermarked_img), 0., 1.)
                    total_psnr += batch_PSNR(output, img, 1.)

        avg_psnr = total_psnr / (len(self.val_dataset) * len(val_relevant_methods))
        self.writer.add_scalar("Epoch_Metrics_Val/PSNR_avg", avg_psnr, epoch + 1)
        return avg_psnr

    def evaluate_test(self, epoch: int) -> None:
        """
        Evaluates the model on the test dataset and logs the results to Tensorboard.
        
        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        avg_psnr_test = 0.0
        test_losses = {'reconstruction': 0.0, 'perceptual': 0.0, 'total': 0.0}

        with torch.no_grad():
            for i, (clean_img, watermarked_img) in enumerate(self.test_loader):
                # Ensure dimensions are multiples of 32
                _, _, w, h = clean_img.shape
                w, h = (int(dim // 32 * 32) for dim in (w, h))
                clean_img = clean_img[:, :, :w, :h]
                watermarked_img = watermarked_img[:, :, :w, :h]

                clean_img, watermarked_img = clean_img.to(self.device), watermarked_img.to(self.device)
                output = torch.clamp(self.model(watermarked_img), 0., 1.)

                # Calculate PSNR for test set
                avg_psnr_test += batch_PSNR(output, clean_img, 1.)

                # Calculate losses for test set
                reconstruction_loss = self.criterion(output, clean_img) / clean_img.size(0) * 2
                test_losses['reconstruction'] += reconstruction_loss.item()

                if self.config.use_perceptual_loss:
                    output_features = self.vgg_model(output)
                    target_features = self.vgg_model(clean_img)
                    perceptual_loss = 0.024 * self.criterion(output_features, target_features) / (
                                target_features.size(0) / 2)
                    test_losses['perceptual'] += perceptual_loss.item()
                    test_losses['total'] += reconstruction_loss.item() + perceptual_loss.item()
                else:
                    test_losses['total'] += reconstruction_loss.item()

                # Log a few sample images from the test set to Tensorboard
                if i < self.tb_config.test_batches_to_save:  # Log images from the first batch
                    img_grid = vutils.make_grid(
                        torch.cat([watermarked_img, output, clean_img]),
                        nrow=self.config.batch_size,
                        normalize=True
                    )
                    self.writer.add_image(f'Images_Test/Epoch_{epoch + 1}_Sample_{i + 1}', img_grid, self.global_step)

        # Calculate and log average PSNR for the test set
        avg_psnr_test /= len(self.test_loader)
        self.writer.add_scalar('Epoch_Metrics_Test/PSNR_avg', avg_psnr_test, epoch + 1)

        # Calculate and log average losses for the test set
        num_test_batches = len(self.test_loader)
        for loss_name, loss_sum in test_losses.items():
            if loss_name == 'total' or self.tb_config.log_detailed_losses_test:
                avg_loss = loss_sum / num_test_batches
                self.writer.add_scalar(f'Epoch_Metrics_Test/{loss_name}_loss_avg', avg_loss, epoch + 1)
        if PRINT_DURING_TRAINING:
            print(f"[epoch {epoch + 1}] Test PSNR: {avg_psnr_test:.4f}")    

    def _epoch_is_saveworthy(self, epoch: int, is_best: bool) -> bool:
        """Determine if the current epoch should be saved to disk."""
        is_nth_checkpoint = epoch % self.tb_config.save_checkpoint_nth_epoch == 0 and epoch > 0
        is_last_epoch = epoch == self.config.epochs - 1
        is_best = is_best and epoch > self.config.epochs // 20
        return is_nth_checkpoint or is_best or is_last_epoch

    
    def _save_model(self, epoch: int, is_best: bool = False) -> None:
        """Save the current model state to disk."""
        if not self._epoch_is_saveworthy(epoch, is_best):
            return

        filename = f"{self.config.model_name}_"
        filename += f"{epoch:03}"
        filename += "_best" if is_best else ""
        pathname = Path(self.config.model_output_path) / f"{filename}.pth"
        torch.save(
            {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step
        }, pathname)


    def _log_training_batch(
    self, 
    epoch: int, 
    batch_step: int, 
    watermarked_img: torch.Tensor,
    output: torch.Tensor, 
    target_img: torch.Tensor,
    losses: Dict[str, float],
    psnr: float,
    global_step: int
) -> None:
        """Log detailed training metrics and visualizations to tensorboard."""
        if self.writer is None:
            return
        
        # Log loss (components/total)
        for loss_name, loss_value in losses.items():
            if self.tb_config.log_detailed_losses_step or loss_name == 'total':
                self.writer.add_scalar(f"Loss_Train/loss_{loss_name}", loss_value, global_step)
        
        # Log PSNR
        self.writer.add_scalar("PSNR_Train/PSNR", psnr, global_step)
        
        # Log gradient norms
        if self.tb_config.log_gradient_norms:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    self.writer.add_scalar(f"Gradients/{name}_norm", grad_norm, global_step)
        
        is_nth_global_step = global_step % self.tb_config.save_images_nth_global_step == 0 and global_step > 0
        is_nth_epoch = epoch % self.tb_config.save_images_nth_epoch == 0 and epoch > 0 and batch_step == 0

        # Periodically log images
        if is_nth_global_step or is_nth_epoch:
            # Create a grid of sample images
            img_grid = vutils.make_grid([
                watermarked_img[0].cpu(),  # Input watermarked image
                output[0].cpu(),           # Model output
                target_img[0].cpu()        # Target image
            ], normalize=True, nrow=3)
            
            self.writer.add_image(
                f'Images_Train/epoch_{epoch}_batch_{batch_step}',
                img_grid,
                global_step
            )
            
            # Log histograms of model parameters
            if self.tb_config.log_parameter_histograms:
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f"Parameters/{name}", param, global_step)


    def train(self) -> None:
        """Execute the complete training pipeline."""
        print(f'Training on {len(self.train_dataset)} samples')

        input(f"Please confirm the following training configuration:\n" +
              f"=== Training Config ===\n{self.config}\n\n" +
              f"=== Tensorboard Config ===\n{self.tb_config}\n\n" +
              f"=== Resume Options ===\n{self.resume_options}\n\n" +
              f"=== Scheduler State ===\n{self.scheduler.state_dict()}\n\n" +
              f"Optimizer LR: {self.optimizer.state_dict()['param_groups'][0]['lr']}\n" +
              f"Press Enter to continue...")
        print("Starting training!")

        best_psnr = 0.0
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_losses = {
                'reconstruction': 0.0,
                'perceptual': 0.0,
                'total': 0.0
            }
            epoch_psnr = 0.0

            self.log_learning_rate(epoch)
            
            for i, img in enumerate(self.train_loader):
                output, target, losses, psnr = self._train_step(img)

                for k, v in losses.items():
                    epoch_losses[k] += v
                epoch_psnr += psnr

                self._log_training_batch(
                    epoch, i, img, output, target, losses, psnr, self.global_step
                )
                
                if PRINT_DURING_TRAINING:
                    print(f"[epoch {epoch + 1}][{i + 1}/{len(self.train_loader)}] "
                        f"loss: {losses['total']:.4f} PSNR_train: {psnr:.4f}")
                
                self.global_step += 1

            self.scheduler.step()

            # Log epoch-level metrics
            num_batches = len(self.train_loader)
            for loss_name, loss_sum in epoch_losses.items():
                if self.tb_config.log_detailed_losses_epoch or loss_name == 'total':
                    avg_loss = loss_sum / num_batches
                    self.writer.add_scalar(f"Epoch_Metrics_Train/{loss_name}_loss_avg",
                                            avg_loss, epoch + 1)

            avg_epoch_psnr = epoch_psnr / num_batches
            self.writer.add_scalar("Epoch_Metrics_Train/PSNR_avg", avg_epoch_psnr, epoch + 1)

            # Validation
            random.seed("validation")
            val_psnr = self.validate(epoch)
            random.seed()
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                self._save_model(epoch, is_best=True)
            else:
                self._save_model(epoch, is_best=False)

            if (epoch + 1) % self.tb_config.test_nth_epoch == 0:
                self.evaluate_test(epoch)

            if PRINT_DURING_TRAINING:
                print(f"\n[epoch {epoch + 1}] "
                    f"Avg_loss: {epoch_losses['total']/num_batches:.4f} "
                    f"Avg_PSNR: {avg_epoch_psnr:.4f} "
                    f"Val_PSNR: {val_psnr:.4f}")

        self.writer.close()


def main():
    """Entry point for training the watermark removal model."""
    resume_options = None
    yaml_config = get_config('configs/config.yaml')
    # resume_options = ResumeOptions(checkpoint_filepath='output/models/HN_per_L1_n2n_035_best.pth', log_dir='output/runs/004-finetune-001') # , log_dir='output/runs/002'
    config = TrainingConfig(
        model_output_path=yaml_config['train_model_out_path_SWCNN'],
        data_path=yaml_config['data_path'],
    )
    
    trainer = WatermarkCleaner(config, resume_options=resume_options)
    trainer.train()


if __name__ == "__main__":
    main()
