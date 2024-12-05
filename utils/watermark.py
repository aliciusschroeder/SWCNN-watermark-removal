import random
from typing import Union, Optional, Tuple

from PIL import Image
import numpy as np
import torch

ApplicationType = Literal["stamp", "map"]

DEBUG = True

def print_debug(
    *values: object,
    debug = False
) -> None:
    print(*values) if debug else None

def load_watermark(
    random_img: Union[str, int],
    alpha: float,
    data_path: str = "data/watermarks/"
) -> Image.Image:
    """
    Load a watermark image and adjust its opacity.

    Args:
        random_img (str or int): Filename of the watermark image without extension.
        alpha (float): Opacity of the watermark (between 0 and 1).
        data_path (str, optional): Path to the watermark images directory. Defaults to "data/watermarks/".

    Returns:
        Image.Image: The watermark image with adjusted opacity.
    """
    watermark = Image.open(f"{data_path}{random_img}.png").convert("RGBA")
    w, h = watermark.size

    for i in range(w):
        for k in range(h):
            color = watermark.getpixel((i, k))
            if not isinstance(color, tuple):
                raise ValueError(f"Unexpected pixel value at ({i}, {k}): {color}")
            if color[3] != 0:
                transparency = int(255 * alpha)
                color = color[:-1] + (transparency,)
                watermark.putpixel((i, k), color)

    # Note: For performance improvement, consider using numpy arrays to adjust the alpha channel.
    return watermark


def apply_watermark(
    base_image: Image.Image,
    watermark: Image.Image,
    scale: float,
    position: Tuple[int, int]
) -> Image.Image:
    """
    Apply a watermark to a base image at a specified position and scale.

    Args:
        base_image (Image.Image): The base image to which the watermark will be applied.
        watermark (Image.Image): The watermark image.
        scale (float): Scaling factor for the watermark.
        position (Tuple[int, int]): (x, y) position where the watermark will be placed.

    Returns:
        Image.Image: The image with the watermark applied.
    """
    # Scale the watermark image
    scaled_width = int(watermark.width * scale)
    scaled_height = int(watermark.height * scale)
    scaled_watermark = watermark.resize((scaled_width, scaled_height))

    # Create a transparent layer the size of the base image
    layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    # Paste the scaled watermark onto the layer at the specified position
    layer.paste(scaled_watermark, position, scaled_watermark)

    # Composite the base image and the watermark layer
    return Image.alpha_composite(base_image, layer)


def calculate_occupancy(img_cnt: np.ndarray, occupancy_ratio: float) -> bool:
    """
    Determine if the occupied pixels exceed the specified occupancy ratio.

    Args:
        img_cnt (np.ndarray): Image array where non-zero pixels are considered occupied.
        occupancy_ratio (float): Desired occupancy ratio in percentage (0-100).

    Returns:
        bool: True if occupied pixels exceed the occupancy ratio, False otherwise.
    """
    sum_pixels = np.sum(img_cnt > 0)
    total_pixels = img_cnt.size
    occupancy_threshold = total_pixels * occupancy_ratio / 100
    threshold_exceeded = False
    if sum_pixels > occupancy_threshold:
        threshold_exceeded = True
    if DEBUG:
        print(f"Occupied pixels: {sum_pixels}, Total pixels: {total_pixels}, Occupancy threshold: {occupancy_threshold}")
    return threshold_exceeded

def show_tmp_img(tmp: Image.Image) -> None:
    """
    Display the temporary image for debugging purposes.

    Args:
        tmp (Image.Image): Temporary image to display.
    """
    r, g, b, a = tmp.split()
    preview = Image.merge("RGBA", (b, g, r, a))
    plt.imshow(preview)
    plt.axis("off")
    plt.show()

def add_watermark_noise_generic(
    img_train: torch.Tensor,
    occupancy: float = 50,
    self_supervision: bool = False,
    same_random: int = 0,
    alpha: float = 0.3,
    img_id: Optional[int] = None,
    scale_img: Optional[float] = None,
    fixed_position: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Add watermark noise to images.

    This function applies watermarks to images, either in standalone mode for a single image,
    or in batch processing mode for multiple images.

    Args:
        img_train (torch.Tensor): Input image tensor(s). In standalone mode, shape [C, H, W].
            In batch processing mode, shape [N, C, H, W].
        occupancy (float, optional): Desired occupancy ratio percentage (0-100). Defaults to 50.
        self_supervision (bool, optional): Whether to use self-supervision mode. Defaults to False.
        same_random (int, optional): Random seed or image index to use when self_supervision is True. Defaults to 0.
        alpha (float, optional): Opacity of the watermark (0 to 1). Defaults to 0.3.
        img_id (int, optional): Specific watermark image ID to use. If None, a random image is selected. Defaults to None.
        scale_img (float, optional): Fixed scale for the watermark image. If None, random scaling is used. Defaults to None.
        fixed_position (Tuple[int, int], optional): Fixed position (x, y) to place the watermark. If None, random position is used. Defaults to None.
        standalone (bool, optional): Whether to run in standalone mode for a single image. Defaults to False.

    Returns:
        Union[Image.Image, torch.Tensor]: In standalone mode, returns a PIL Image with the watermark applied.
            In batch processing mode, returns a torch.Tensor of images with watermarks applied.
    """

    # Batch processing
    if img_id is not None:
        random_img = img_id
    else:
        random_img = same_random if self_supervision else random.randint(1, 173)

    # Adjust alpha value based on whether scaling is fixed or not
    if scale_img is not None:
        # Fixed alpha when a specific scale is provided
        alpha = alpha
    else:
        # Adjust alpha randomly for certain function variants
        if 'add_watermark_noise_B' in add_watermark_noise_generic.__name__:
            alpha = alpha + random.randint(0, 70) * 0.01

    # Load the watermark image with adjusted opacity
    watermark = load_watermark(random_img, alpha)

    # Convert input tensor to NumPy array and adjust dimensions
    img_train_np = img_train.numpy()
    _, _, img_h, img_w = img_train_np.shape
    # Randomly select an occupancy level between 0 and the specified occupancy
    occupancy = np.random.uniform(0, occupancy)

    # Rearrange dimensions for processing
    img_train_np = np.ascontiguousarray(np.transpose(img_train_np, (0, 2, 3, 1)))

    if DEBUG:
        print(f"Adding watermark noise with occupancy {occupancy}, alpha {alpha}")
        print(f"Watermark size: {watermark.size}")
        print(f"img_train_np size: {img_w} x {img_h}")
        print(f"Number of images: {len(img_train_np)}\n\n")

    for i in range(len(img_train_np)):
        # Convert the image to PIL format
        tmp = Image.fromarray((img_train_np[i] * 255).astype(np.uint8)).convert("RGBA")
        # Initialize an empty image for counting occupied pixels
        img_for_cnt = Image.new("L", (img_w, img_h), 0)

        while True:
            # Determine scaling factor
            scale = scale_img if scale_img is not None else np.random.uniform(0.5, 1.0)
            scaled_watermark = watermark.resize((int(watermark.width * scale), int(watermark.height * scale)))

            # Determine position to paste the watermark
            if fixed_position is not None:
                x, y = fixed_position
            else:
                x = random.randint(0, img_w - scaled_watermark.width)
                y = random.randint(0, img_h - scaled_watermark.height)

            # Apply the watermark to the image (debug messages are already included)
            tmp = apply_watermark(tmp, scaled_watermark, 1.0, (x, y))

            if DEBUG:
                show_tmp_img(tmp)

            img_for_cnt = apply_watermark(img_for_cnt.convert("RGBA"), scaled_watermark, 1.0, (x, y)).convert("L")
            img_cnt = np.array(img_for_cnt)

            # Check if the occupancy condition is met
            if calculate_occupancy(img_cnt, occupancy):
                # Update the image in the array
                img_rgb = np.array(tmp).astype(float) / 255.0
                img_train_np[i] = img_rgb[:, :, :3]
                break

    # Rearrange dimensions back to original and convert to tensor
    img_train_np = np.transpose(img_train_np, (0, 3, 1, 2))
    return torch.tensor(img_train_np)


def add_watermark_noise(
    img_train: torch.Tensor,
    occupancy: float = 50,
    self_supervision: bool = False,
    same_random: int = 0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    Add watermark noise to images using default parameters.

    Args:
        img_train (torch.Tensor): Input image tensor(s).
        occupancy (float, optional): Desired occupancy ratio percentage. Defaults to 50.
        self_supervision (bool, optional): Whether to use self-supervision mode. Defaults to False.
        same_random (int, optional): Random seed or image index. Defaults to 0.
        alpha (float, optional): Opacity of the watermark. Defaults to 0.3.

    Returns:
        torch.Tensor: Images with watermarks applied.
    """
    return add_watermark_noise_generic(
        img_train=img_train,
        occupancy=occupancy,
        self_supervision=self_supervision,
        same_random=same_random,
        alpha=alpha
    )


def add_watermark_noise_B(
    img_train: torch.Tensor,
    occupancy: float = 50,
    self_supervision: bool = False,
    same_random: int = 0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    Add watermark noise to images with adjusted alpha value.

    Args:
        img_train (torch.Tensor): Input image tensor(s).
        occupancy (float, optional): Desired occupancy ratio percentage. Defaults to 50.
        self_supervision (bool, optional): Whether to use self-supervision mode. Defaults to False.
        same_random (int, optional): Random seed or image index. Defaults to 0.
        alpha (float, optional): Base opacity of the watermark. Actual opacity may vary. Defaults to 0.3.

    Returns:
        torch.Tensor: Images with watermarks applied.
    """
    return add_watermark_noise_generic(
        img_train=img_train,
        occupancy=occupancy,
        self_supervision=self_supervision,
        same_random=same_random,
        alpha=alpha,
        # Additional parameters can be set here if needed
    )


def add_watermark_noise_test(
    img_train: torch.Tensor,
    occupancy: float = 50,
    img_id: int = 3,
    scale_img: float = 1.5,
    self_supervision: bool = False,
    same_random: int = 0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    Add watermark noise to images for testing purposes.

    Args:
        img_train (torch.Tensor): Input image tensor(s).
        occupancy (float, optional): Desired occupancy ratio percentage. Defaults to 50.
        img_id (int, optional): Specific watermark image ID to use. Defaults to 3.
        scale_img (float, optional): Fixed scale for the watermark image. Defaults to 1.5.
        self_supervision (bool, optional): Whether to use self-supervision mode. Defaults to False.
        same_random (int, optional): Random seed or image index. Defaults to 0.
        alpha (float, optional): Opacity of the watermark. Defaults to 0.3.

    Returns:
        torch.Tensor: Images with watermarks applied.
    """


    return add_watermark_noise_generic(
        img_train=img_train,
        occupancy=occupancy,
        self_supervision=self_supervision,
        same_random=same_random,
        alpha=alpha,
        img_id=img_id,
        scale_img=scale_img,
        fixed_position=(0, 0)  # Uncomment and set position if needed
    )


def add_watermark_noise_standalone(
    img_train: torch.Tensor,
    occupancy: float = 50,
    data_path = "data/watermarks/2.png",
):
    # Standalone processing for a single image
    watermark = Image.open(data_path).convert("RGBA")
    # Convert the input tensor to a NumPy array and prepare the image
    noise = img_train.numpy()
    _, h, w = noise.shape
    # Randomly select an occupancy level between 0 and the specified occupancy
    occupancy = np.random.uniform(0, occupancy)

    # Prepare the image
    noise = np.ascontiguousarray(np.transpose(noise, (1, 2, 0)))
    noise = np.uint8(noise * 255)
    noise_pil = Image.fromarray(noise)

    # Initialize an empty image for counting occupied pixels
    img_for_cnt = Image.new("L", (w, h), 0)

    while True:
        # Randomly rotate and scale the watermark
        angle = random.randint(-45, 45)
        scale = random.uniform(0.5, 1.0)
        rotated_watermark = watermark.rotate(angle, expand=1).resize(
            (int(watermark.width * scale), int(watermark.height * scale))
        )
        # Randomly choose a position to paste the watermark
        x = random.randint(-rotated_watermark.width, w)
        y = random.randint(-rotated_watermark.height, h)
        # Apply the watermark to the image
        noise_pil = apply_watermark(noise_pil, rotated_watermark, 1.0, (x, y))
        img_for_cnt = apply_watermark(img_for_cnt.convert("RGBA"), rotated_watermark, 1.0, (x, y)).convert("L")
        img_cnt = np.array(img_for_cnt)
        # Check if the occupancy condition is met
        if calculate_occupancy(img_cnt, occupancy):
            break
    return noise_pil
