"""
This script loads a series of model checkpoints and evaluates the average PSNR on a test dataset for each checkpoint.
The average PSNR values are then plotted against the epoch numbers to visualize the model performance over training.
Models in the specified directory should be named as 3-digit, zero-padded '[epoch_number].pth'.
"""

import os
import re
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from models import HN
from utils.validation import batch_PSNR
from typing import List, Tuple

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a trained model from the specified path.

    Args:
        model_path (str): Path to the .pth model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = HN()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def get_checkpoint_epochs(model_history_dir: str) -> List[int]:
    """
    Extracts the epoch numbers from the checkpoint files in the given directory.

    Args:
        model_history_dir (str): Path to the directory containing checkpoint files.

    Returns:
        List[int]: Sorted list of epoch numbers.
    """
    epochs = []
    for filename in os.listdir(model_history_dir):
        if filename.endswith(".pth"):
            match = re.match(r"(\d{3})\.pth", filename)
            if match:
                epoch = int(match.group(1))
                epochs.append(epoch)
    epochs.sort()
    return epochs

def calculate_avg_psnr(
    model: torch.nn.Module,
    data_file: str,
    device: torch.device
) -> float:
    """
    Calculates the average PSNR for the watermarked images in the given data file using the provided model.

    Args:
        model (torch.nn.Module): The trained model.
        data_file (str): Path to the HDF5 file containing paired clean and watermarked images.
        device (torch.device): Device to perform calculations on.

    Returns:
        float: Average PSNR value.
    """
    total_psnr = 0.0
    num_images = 0

    with h5py.File(data_file, 'r') as f:
        num_pairs = len(f.keys()) // 2
        for i in range(num_pairs):
            clean_data = np.array(f[str(i * 2)])
            watermarked_data = np.array(f[str(i * 2 + 1)])

            clean_tensor = torch.tensor(clean_data, dtype=torch.float32).unsqueeze(0).to(device)
            watermarked_tensor = torch.tensor(watermarked_data, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = torch.clamp(model(watermarked_tensor), 0.0, 1.0)
                psnr = batch_PSNR(output, clean_tensor, 1.0)
                total_psnr += psnr
                num_images += 1

    avg_psnr = total_psnr / num_images if num_images > 0 else 0.0
    return avg_psnr

def main():
    model_history_dir = "output/model-history"
    data_file = "data/test_color.h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = get_checkpoint_epochs(model_history_dir)
    avg_psnr_values = []

    for epoch in epochs:
        model_path = os.path.join(model_history_dir, f"{epoch:03d}.pth")
        model = load_model(model_path, device)
        avg_psnr = calculate_avg_psnr(model, data_file, device)
        avg_psnr_values.append(avg_psnr)
        print(f"Epoch {epoch}: Average PSNR = {avg_psnr:.4f}")

    # Plotting
    plt.plot(epochs, avg_psnr_values, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average PSNR")
    plt.title("Average PSNR vs. Epoch")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
