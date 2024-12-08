from dataclasses import dataclass


@dataclass
class TensorBoardConfig:
    """Configuration parameters for Tensorboard logging."""
    log_dir: str = "output/runs"
    log_detailed_losses: bool = False
    log_parameter_histograms: bool = False
    log_gradient_norms: bool = False
    save_checkpoint_nth_epoch: int = 5  # exclusive of epoch 0
    save_images_nth_epoch: int = 5  # exclusive of epoch 0
    save_images_nth_batch: int = 100  # inclusive of batch 0
