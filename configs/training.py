from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainingConfig:
    """Configuration parameters for the SWCNN training process."""
    batch_size: int = 32
    num_layers: int = 17  # Number of total layers(DnCNN)
    epochs: int = 500
    milestone: int = 30  # Learning rate decay epoch
    initial_lr: float = 0.002
    model_output_path: str = "output/models"
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
        ]
        return "_".join(components)


@dataclass
class ResumeOptions:
    """Configuration parameters for resuming training from a checkpoint."""
    checkpoint_filepath: str
    log_dir: Optional[str] = None
