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

    # exclusive of epoch/step 0
    save_checkpoint_nth_epoch: int = 1
    save_images_nth_epoch: int = 5
    save_images_nth_global_step: int = 500
    test_nth_epoch: int = 1

    # number of batches to save to tensorboard during testing
    test_batches_to_save: int = 3
