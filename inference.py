#!/usr/bin/env python

"""
inference.py

This script performs inference using a trained SWCNN model to remove watermarks from input images by processing the image in overlapping patches.

Usage:
    python inference.py --config configs/config.yaml --model_path path/to/model.pth --input_image path/to/input.jpg --output_image path/to/output.jpg
"""

import os
import argparse
import torch
import cv2
import numpy as np
from models import HN
from math import ceil


def parse_args():
    parser = argparse.ArgumentParser(description="SWCNN Inference Script with Patch Processing")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help="Path to the configuration YAML file.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained .pth model file.")
    parser.add_argument('--input_image', type=str, required=True,
                        help="Path to the input image with watermark.")
    parser.add_argument('--output_image', type=str, required=True,
                        help="Path where the output image will be saved.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to run the model on. Options: 'cuda', 'cpu'.")
    return parser.parse_args()


def load_model(model_path, device):
    """
    Loads the trained model from the specified path, supporting both legacy (only weight) and new (weights, optimizer, epoch) formats.

    Args:
        model_path (str): Path to the .pth model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize the model architecture
    model = HN()

    # Load the state dictionary
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Detected new checkpoint format.")
        model_state_dict = checkpoint['model_state_dict']
    else:
        print("Detected legacy checkpoint format.")
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
    print(f"Model loaded successfully from {model_path}")
    return model


def pad_image_for_patching(image, patch_size=256, stride=128):
    """
    Pads the image so that it can be divided into patches of size patch_size x patch_size with the given stride.

    Args:
        image (numpy.ndarray): Input image in HWC format.
        patch_size (int): Size of each square patch.
        stride (int): Stride between patches.

    Returns:
        padded_image (numpy.ndarray): Padded image.
        pad (tuple): Padding applied as (pad_left, pad_right, pad_top, pad_bottom).
    """
    H, W, C = image.shape
    num_patches_h = ceil((H - patch_size) / stride) + 1
    num_patches_w = ceil((W - patch_size) / stride) + 1

    total_coverage_h = stride * (num_patches_h - 1) + patch_size
    total_coverage_w = stride * (num_patches_w - 1) + patch_size

    pad_h = max(total_coverage_h - H, 0)
    pad_w = max(total_coverage_w - W, 0)

    # Apply padding (use reflect padding to minimize border artifacts)
    padded_image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded_image, (0, pad_w, 0, pad_h), num_patches_h, num_patches_w


def remove_padding(image, pad):
    """
    Removes the padding from the image.

    Args:
        image (numpy.ndarray): Padded image.
        pad (tuple): Padding applied as (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        cropped_image (numpy.ndarray): Image with padding removed.
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    if pad_bottom == 0 and pad_right == 0:
        return image
    h, w = image.shape[:2]
    return image[0:h - pad_bottom, 0:w - pad_right]


def extract_patches(image, patch_size=256, stride=128):
    """
    Extracts overlapping patches from the image.

    Args:
        image (numpy.ndarray): Padded image in HWC format.
        patch_size (int): Size of each square patch.
        stride (int): Stride between patches.

    Returns:
        patches (list): List of patch numpy arrays.
        positions (list): List of (x, y) positions for each patch.
    """
    H, W, C = image.shape
    patches = []
    positions = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch)
            positions.append((x, y))

    return patches, positions


def reconstruct_image(patches, positions, image_shape, patch_size=256, stride=128):
    """
    Reconstructs the image from processed patches by blending overlapping regions.

    Args:
        patches (list): List of processed patch numpy arrays.
        positions (list): List of (x, y) positions for each patch.
        image_shape (tuple): Shape of the padded image (H, W, C).
        patch_size (int): Size of each square patch.
        stride (int): Stride between patches.

    Returns:
        reconstructed_image (numpy.ndarray): The blended reconstructed image.
    """
    H, W, C = image_shape
    accumulator = np.zeros((H, W, C), dtype=np.float32)
    weight = np.zeros((H, W, C), dtype=np.float32)

    for patch, (x, y) in zip(patches, positions):
        accumulator[y:y + patch_size, x:x + patch_size, :] += patch
        weight[y:y + patch_size, x:x + patch_size, :] += 1.0

    # Avoid division by zero
    weight[weight == 0] = 1.0
    reconstructed_image = accumulator / weight
    reconstructed_image = np.clip(reconstructed_image, 0.0, 1.0)
    return reconstructed_image


def preprocess_image(image_path, patch_size=256, stride=128, device='cuda'):
    """
    Loads and preprocesses the image for inference by splitting it into patches.

    Args:
        image_path (str): Path to the input image.
        patch_size (int): Size of each square patch.
        stride (int): Stride between patches.
        device (torch.device): Device to load the image tensor on.

    Returns:
        patches_tensor (list): List of preprocessed patch tensors.
        positions (list): List of (x, y) positions for each patch.
        padded_size (tuple): Padded image size as (H, W).
        pad (tuple): Padding applied as (pad_left, pad_right, pad_top, pad_bottom).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")

    # Load image using OpenCV
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read the image from {image_path}")

    original_size = image_bgr.shape[:2]  # (H, W)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image_normalized = image_rgb.astype(np.float32) / 255.0

    # Pad the image to fit patches
    padded_image, pad, num_patches_h, num_patches_w = pad_image_for_patching(
        image_normalized, patch_size, stride)

    # Extract patches
    patches, positions = extract_patches(padded_image, patch_size, stride)

    # Convert patches to tensors and move to device
    patches_tensor = []
    for patch in patches:
        tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(
            0).to(device)  # Shape: 1 x C x H x W
        patches_tensor.append(tensor)

    return patches_tensor, positions, padded_image.shape[:2], pad, original_size


def postprocess_image(reconstructed_padded, original_size, pad):
    """
    Postprocesses the reconstructed padded image to convert it into the final output image.

    Args:
        reconstructed_padded (numpy.ndarray): Reconstructed padded image in RGB format.
        original_size (tuple): Original image size as (H, W).
        pad (tuple): Padding applied as (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        output_image (numpy.ndarray): Final output image in BGR format.
    """
    # Remove padding
    output_cropped = remove_padding(reconstructed_padded, pad)

    # Denormalize to [0, 255]
    output_denormalized = (output_cropped * 255.0).astype(np.uint8)

    return output_denormalized


def main():
    args = parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    model = load_model(args.model_path, device)

    # Preprocess the image: extract patches
    patches_tensor, positions, padded_size, pad, original_size = preprocess_image(
        args.input_image, patch_size=256, stride=128, device=device.type)
    print(f"Input image loaded and preprocessed. Original size: {original_size}, Padded size: {padded_size}")

    processed_patches = []
    batch_size = 16  # Adjust based on GPU memory
    num_patches = len(patches_tensor)
    print(f"Total patches to process: {num_patches}")

    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_patches = patches_tensor[i:i + batch_size]
            batch = torch.cat(batch_patches, dim=0)  # Shape: batch_size x C x H x W
            outputs = model(batch)  # Assuming model outputs in the same scale
            outputs = torch.clamp(outputs, 0.0, 1.0)
            outputs_np = outputs.cpu().numpy()
            # Convert back to list of numpy arrays
            for j in range(outputs_np.shape[0]):
                patch = np.transpose(outputs_np[j], (1, 2, 0))  # C x H x W -> H x W x C
                processed_patches.append(patch)
            print(f"Processed batch {i // batch_size + 1} / {ceil(num_patches / batch_size)}")

    print("All patches processed. Reconstructing the final image.")

    # Reconstruct the image by blending patches
    reconstructed_padded = reconstruct_image(
        processed_patches, positions, (*padded_size, 3), patch_size=256, stride=128)
    print("Image reconstruction completed.")

    # Postprocess to get the final output image
    output_image = postprocess_image(
        reconstructed_padded, original_size, pad)
    print(f"Postprocessing completed. Saving output image to {args.output_image}")

    cv2.imwrite(args.output_image, output_image)
    print("Output image saved successfully.")


if __name__ == "__main__":
    main()
