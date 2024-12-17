**This SWCNN uses a self-supervised way to construct reference watermarked images rather than given paired training samples, according to watermark distribution.**

A heterogeneous U-Net architecture is used to extract more complementary structural information via simple components for image watermark removal. Taking into account texture information, a mixed loss is exploited to improve visual effects of image watermark removal.

Experimental results show that the proposed SWCNN is superior to popular CNNs in image watermark removal.

This model is based on a paper by Chunwei Tian, Menghua Zheng, Tiancai Jiao, Wangmeng Zuo, Yanning Zhang, Chia-Wen Lin which can be found [here](https://arxiv.org/html/2403.05807v1).

This is a fork of the reference implementation of that paper aimed at improving several aspects:

**I. Major Improvements and New Features:**

1. **Refactoring and Code Structure:**
  
    *   The codebase has been significantly refactored for improved readability, maintainability, and modularity. This includes:
        *   Moving several modules into separate files.
        *   Separating the config logic.
        *   Creating `configs` for training and tensorboard.
    *   Further reference: [`6cc09d0`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/6cc09d0)

2. **Enhanced Watermark Handling:**
    *   **Watermark Manager Class:** Introduced a `WatermarkManager` class to centralize watermark loading, caching, and application. This improves code organization and efficiency.
    *   **Artifacts Configuration:** Added support for configuring and applying artifacts around watermarks, simulating real-world imperfections.
    *   **New Watermark Application:** Implemented `add_watermark_noise_generic` for flexible watermark application with various options like scaling, positioning, and alpha blending. Replaced old, less flexible methods.
    *   **Watermark Variation:** Added more watermark options and control over the watermark application process.
    *   **Deterministic Watermarks for Validation:** Added the ability to generate deterministic watermarks for consistent validation. ([`8e0c4ca`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8e0c4ca))
    *   **Watermark Map Caching**: Watermark maps are now resized and cached for faster usage. ([`790c4ca`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/790c4ca))
    *   **Watermark Opacity:** The `--alpha` parameter now controls the opacity of the watermark.
    *   Further reference: [`8e0c4ca`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8e0c4ca), [`d8f3183`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/d8f3183), [`b8db4c3`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/b8db4c3), [`9b0f729`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/9b0f729), [`7cd7477`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/7cd7477), [`f1339af`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/f1339af), [`1a15d39`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/1a15d39), [`c93778d`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/c93778d), [`0305673`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/0305673)

3. **Training Process Enhancements:**
    *   **TensorBoard Integration:** Implemented comprehensive TensorBoard logging for monitoring training progress. Metrics logged include:
        *   Loss components (reconstruction and perceptual, if enabled)
        *   Learning rate
        *   PSNR
        *   Histograms of model parameters and gradients (optional)
        *   Images at different stages of processing
    *   **Configurable Logging:** Added options to control the granularity of TensorBoard logging.
    *   **Mixed Precision Training:** Introduced mixed precision training using `GradScaler` to potentially speed up training and reduce memory usage.
    *   **Resumable Training:** Added option to resume training from a checkpoint. ([`2ae85f5`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/2ae85f5))
    *   **Learning Rate Scheduler:** Implemented a learning rate scheduler for dynamic adjustment during training.
    *   Further reference: [`83c5024`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/83c5024), [`6906d07`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/6906d07), [`1c053a4`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/1c053a4), [`e904da6`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/e904da6)

4. **Performance Optimizations:**
    *   **Optimized Data Loading:** Improved data loading efficiency by using `pin_memory=True` and adjusting `num_workers` in the `DataLoader`.
    *   **Reduced Redundancy:** Removed unnecessary computations and memory allocations.
    *   **Refactored Watermark Application:** Optimized the watermark application process for better performance.
    *   **GPU Usage:** Improved GPU utilization.
    *   **CUDA:** Implemented CUDA-accelerated color space conversions. ([`dfbca74`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/dfbca74))
    *   **Performance Benchmarks**: The repository now includes benchmarks for measuring the speedup achieved with CUDA and other optimizations.
    *   **Patching:** Implemented patching for inference, similar to training. ([`5af24a4`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/5af24a4))
    *   Further reference: [`887da9d`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/887da9d), [`98be395`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/98be395), [`787870f`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/787870f), [`dfbca74`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/dfbca74), [`5af24a4`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/5af24a4)

**II. Bug Fixes:**

1. **Image Distortion:** Fixed an issue causing image distortion during resizing by correcting the order of width and height parameters. ([`65eae1a`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/65eae1a))
2. **Model Saving:** Resolved a problem where models were not saving correctly by adjusting the saving logic.
3. **Watermark Application:** Fixed various bugs related to watermark positioning and application.
4. **Alpha Blending:** Corrected the alpha blending implementation for more accurate watermark application.
5. **Path Issues:** Addressed incorrect paths for loading watermarks and saving models.
6. **Typos:** Fixed several typos.
7. **Parameter Naming:** Standardized parameter names for consistency.
8. **`Variable` Deprecation**: Removed redundant usage of `Variable` in favor of `torch.no_grad()`. ([`7f1e460`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/7f1e460))
9. **`volatile` Replacement**: Replaced deprecated `volatile=True` with `torch.no_grad()`. ([`d2ad14a`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/d2ad14a))

**III. Documentation and Usability:**

1. **README Update:** Updated the README to reflect the changes and provide clearer instructions.
2. **Docstrings:** Added detailed docstrings to functions and classes for better code understanding.
3. **Type Hinting:** Added type hints throughout the codebase to improve code clarity and maintainability.
4. **Helper Functions:** Introduced helper functions for common tasks like configuration loading and metric calculations.
5. **Inference Script:** Added a PowerShell script (`inference_folder.ps1`) for running inference on a folder of images.
6. **Output Directory Structure:** Established a new output directory structure for better organization of results.
7. **Distribution Metrics:** Implemented `DistributionMetrics` class for computing and visualizing distribution similarity metrics.

**IV. Removed Functionality:**

1. **DRDNet Model:** Removed the DRDNet model as it was not being used in the core training pipeline.
2. **Unused Code:** Eliminated unused code, variables, and imports to streamline the codebase.
3. **`utils.py` Removal**: split up into multiple files, `config.py` and `data_augmentation.py` moved into their respective folders. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))
4. **`basicblock.py` and `batchrenorm` folder**: Removed unused code related to external libraries. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))
5. **`model_common/filters.py` Removal**: Removed unused filters. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))
6. **`addWatermark.py` Removal**: Functionality merged into `utils/watermark.py`. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))
7. **`flops.py` Removal**: Removed the code for calculating FLOPs as it was not being used. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))
8. **`model_common` folder**: Removed. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))
9. **`train_noisy_L.py` Removal**: Removed the script as it's not used. ([`8d1f777`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/8d1f777))

**V. Other Changes:**

1. **Configuration:** Moved configuration parameters to a YAML file (`configs/config.yaml`) for easier management.
2. **Test Data:** Added a dedicated folder for paired test data (`data/test`) and a text file indicating where to place test images.
3. **Gitignore:** Updated `.gitignore` to exclude unnecessary files and directories.
4. **.vsconfig:** Added `.vsconfig` for development environment.
5. **Requirements:** Added a `requirements.txt` file for easy dependency management.

These changes collectively enhance the SWCNN codebase, making it more robust, efficient, and easier to use. The added documentation and type hinting also improve code readability and maintainability. This list is up to date until 17th December 2024 at commit [`ef85170`](https://github.com/aliciusschroeder/SWCNN-watermark-removal/commit/ef85170).

# Getting Started

This guide will help you set up the environment, install necessary dependencies, and walk you through the steps of training, testing, and performing inference with the SWCNN model.

## Prerequisites

- Python 3.x
- CUDA-enabled GPU (recommended for training)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/aliciusschroeder/SWCNN-watermark-removal.git
    cd SWCNN-watermark-removal
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The SWCNN model is trained on image patches extracted from a dataset of clean images. You'll need to prepare your training and validation data in HDF5 format.

1. **Place your training data** in the `data/train` directory.
2. **Place your validation data** in the `data/validation` directory.
3. **Place your watermark images** in the `data/watermarks` directory.
4. **Place your test images** in the `data/test/clean` directory. The watermarked versions should go to `data/test/watermarked`. Skip this step if you don't have paired test data.
5. **Run the data preparation script:**

    ```bash
    python prepare_data.py
    ```

    This script will:

    -   Process images in `data/train` and `data/validation`.
    -   Create patches of a specified size (default: 256x256) with a given stride (default: 128).
    -   Apply data augmentation (optional).
    -   Save the processed data into HDF5 files (`train_color.h5` and `val_color.h5`) in the `data` directory.
    -   (If you have paired test data and run it again) create an HDF5 file for testing (`test_color.h5`) in the `data` directory.

    You can customize the data preparation process using the following command-line arguments:

    -   `--scales`: List of scaling factors for image augmentation (default: `[1]`).
    -   `--patch_size`: Size of the image patches (default: `256`).
    -   `--stride`: Stride for patch extraction (default: `128`).
    -   `--aug_times`: Number of augmentation operations per patch (default: `1`, no augmentation).
    -   `--mode`: Color mode, either 'gray' or 'color' (default: `color`).
    -   `--max_t_samples`: Maximum number of training samples (default: `0`, no limit).
    -   `--max_v_samples`: Maximum number of validation samples (default: `0`, no limit).
    -   `--sampling_method`: Sampling method, either 'default' or 'mixed' (default: `default`).
    -   `--seed`: Random seed (default: `42`).

## Training

To train the SWCNN model, use the `train.py` script. You can customize the training process using command-line arguments or by modifying the `configs/config.yaml` file.

**Basic Usage:**

```bash
python train.py
```

**Example with custom settings:**

```bash
python train.py --batchSize 16 --epochs 50 --lr 0.0005 --gpu_id 0
```

**Available Options:**

-   `--batchSize`: Training batch size (default: `8`).
-   `--num_of_layers`: Number of layers in the DnCNN (default: `17`).
-   `--epochs`: Number of training epochs (default: `100`).
-   `--milestone`: Epoch at which to decay learning rate (default: `30`).
-   `--lr`: Initial learning rate (default: `1e-3`).
-   `--model_output_path`: Path to save trained models (default: `output/models`).
-   `--net`: Network architecture to use (default: `HN`).
-   `--loss`: Loss function to use (default: `L1`).
-   `--self_supervised`: Whether to use self-supervised training (default: `True`).
-   `--PN`: Whether to use a perceptual network as additional loss component (default: `True`).
-   `--GPU_id`: ID of the GPU to use (default: `0`).

**Monitoring Training:**

Training progress, including loss values and image samples, can be monitored using Tensorboard. Start Tensorboard with:

```bash
tensorboard --logdir output/runs
```

**Resuming Training:**

To resume training from a checkpoint, change the `resume_options` from `None` to the desired settings in `train.py`. The script can automatically read last epoch trained, how many global_steps have been performed and will even purge newer steps from the most recent unsaved epoch in TensorBoard for a clean monitoring experience.

## Inference

To perform inference on a single image, use the `inference.py` script.

**Usage:**

```bash
python inference.py --model_path output/models/your_model.pth --input_image data/test/watermarked/your_image.jpg --output_image output/inference_runs/your_output_image.jpg
```

**Options:**

-   `--model_path`: Path to the trained model file.
-   `--input_image`: Path to the input image with a watermark.
-   `--output_image`: Path to save the output image.
-   `--device`: Device to run inference on (`cuda` or `cpu`).

**Inference on a Folder of Images:**

You can use the provided PowerShell script `scripts/inference_folder.ps1` to perform inference on an entire folder of images.

1. **Modify the script:**
    -   Set the `$InputDirectory` variable to the path of your input folder containing watermarked images.
    -   Set the `$ModelPath` variable to the path of your trained model.

2. **Run the script:**

    ```powershell
    .\scripts\inference_folder.ps1
    ```

The script will process all `.jpg` images in the input directory and save the results in a new folder within `output/inference_runs` named with the current timestamp.

# Network architecture
![image-20240304100316953](assets/network-architecture.png)
