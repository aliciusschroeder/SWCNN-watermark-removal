from dataclasses import dataclass


@dataclass
class TensorBoardConfig:
    """Configuration parameters for Tensorboard logging."""
    log_dir: str = "output/runs"
    # log distinct loss components each step / epoch / test
    log_detailed_losses_step: bool = False
    log_detailed_losses_epoch: bool = False
    log_detailed_losses_test: bool = False

    log_model_architecture: bool = False
    log_parameter_histograms: bool = False
    log_gradient_norms: bool = False
    log_hyperparameters: bool = False
