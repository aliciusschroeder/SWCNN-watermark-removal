#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py

This script performs inference using a trained SWCNN model to remove watermarks from input images.

Usage:
    python inference.py --config configs/config.yaml --model_path path/to/model.pth --input_image path/to/input.jpg --output_image path/to/output.jpg
"""

import os
import argparse
import torch
import cv2
import numpy as np
from models import HN


def parse_args():
    parser = argparse.ArgumentParser(description="SWCNN Inference Script")
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

    # TODO: Check if the model was trained using DataParallel and wether removal of 'module.' prefix is needed

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


def pad_image(image, multiple=32):
    """
    Pads the image so that its dimensions are multiples of 'multiple'.

    Args:
        image (numpy.ndarray): Input image in HWC format.
        multiple (int): The multiple to pad the dimensions to.

    Returns:
        padded_image (numpy.ndarray): Padded image.
        pad (tuple): Padding applied as (pad_left, pad_right, pad_top, pad_bottom).
    """
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) if h % multiple != 0 else 0
    pad_w = (multiple - w % multiple) if w % multiple != 0 else 0

    # Apply padding (use reflect padding to minimize border artifacts)
    padded_image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded_image, (0, pad_w, 0, pad_h)


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


def preprocess_image(image_path, device):
    """
    Loads and preprocesses the image for inference.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to load the image tensor on.

    Returns:
        input_tensor (torch.Tensor): Preprocessed image tensor.
        original_size (tuple): Original image size as (H, W).
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

    # Pad the image
    padded_image, pad = pad_image(image_normalized, multiple=32)

    # Convert to tensor and rearrange dimensions to C x H x W
    input_tensor = torch.from_numpy(padded_image).permute(
        2, 0, 1).unsqueeze(0)  # Shape: 1 x C x H x W
    input_tensor = input_tensor.to(device)

    return input_tensor, original_size, pad


def postprocess_image(output_tensor, original_size, pad):
    """
    Postprocesses the model output tensor to convert it into an image.

    Args:
        output_tensor (torch.Tensor): Model output tensor.
        original_size (tuple): Original image size as (H, W).
        pad (tuple): Padding applied as (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        output_image (numpy.ndarray): Final output image in BGR format.
    """
    # Remove batch dimension and move to CPU
    output_tensor = output_tensor.squeeze(0).cpu()

    # Clamp the values to [0, 1]
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    # Convert to numpy array and rearrange to H x W x C
    output_np = output_tensor.permute(1, 2, 0).numpy()

    # Remove padding
    output_cropped = remove_padding(output_np, pad)

    # Denormalize to [0, 255]
    output_denormalized = (output_cropped * 255.0).astype(np.uint8)

    # Convert RGB back to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_denormalized, cv2.COLOR_RGB2BGR)

    return output_bgr


def main():
    args = parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    model = load_model(args.model_path, device)

    input_tensor, original_size, pad = preprocess_image(
        args.input_image, device)
    print(f"Input image loaded and preprocessed."+
          f"Original size: {original_size},"+
          f"Padded: {input_tensor.shape[2:]}")

    with torch.no_grad():
        output_tensor = model(input_tensor)
        print("Inference completed.")

    output_image = postprocess_image(output_tensor, original_size, pad)
    print(f"Postprocessing completed. Saving "+
          f"output image to {args.output_image}")

    cv2.imwrite(args.output_image, output_image)
    print("Output image saved successfully.")


if __name__ == "__main__":
    main()
