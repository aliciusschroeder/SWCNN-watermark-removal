# @filename: watermark.py

from dataclasses import dataclass
import os
import random
import sys
from typing import Literal, Union, Optional, Tuple, Dict

if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Add the parent directory to sys.path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.ndimage import convolve

from utils.helper import print_debug
from utils.image_numba import linear_to_srgb, srgb_to_linear
from utils.preview import PreviewManager

ApplicationType = Literal["stamp", "map"]
PositionType = Union[Tuple[int, int], Literal["random", "center"]]


@dataclass
class DebugConfig():
    # Debug printing flags
    print_load_all_watermarks: bool = False
    print_load_watermark_image: bool = False
    print_apply_watermark: bool = False
    print_add_watermark_generic: bool = False
    # Debug visualization flags
    show_previews_add_watermark_generic: bool = False


@dataclass
class ArtifactsConfig:
    """
    Configuration for adding artifacts around the watermark area.
    """
    alpha: float = 0.66
    intensity: float = 1
    kernel_size: int = 7


class WatermarkManager:
    """
    Manages loading and caching of watermark images to optimize performance.
    """

    def __init__(
        self,
        data_path: str = "data/watermarks/",
        swap_blue_red_channels: bool = True,
        debug: DebugConfig = DebugConfig()
    ):
        """
        Initializes the WatermarkManager by loading all watermark images into memory.

        :param data_path: Directory where watermark images are stored.
        :param swap_blue_red_channels: Whether to swap blue and red channels.
        :param debug: Enables debug printing if set to True.
        """
        self.data_path = data_path
        self.swap_blue_red_channels = swap_blue_red_channels
        self.debug = debug
        self.watermarks: Dict[Union[str, int], Image.Image] = {}
        self.watermark_maps: Dict[Union[str, int], Image.Image] = {}
        self.watermark_resize_cache: Dict[str, Image.Image] = {}
        self.preview_manager = PreviewManager()
        self._load_all_watermarks()

    def _load_all_watermarks(self) -> None:
        """
        Loads all .png watermark images from the data_path directory into memory.
        """
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f"Watermark directory not found: {self.data_path}")

        print_debug(f"Found files in watermark directory: {os.listdir(self.data_path)}",
                    self.debug.print_load_all_watermarks)
        for file in os.listdir(self.data_path):
            print_debug(f"Checking watermark file: {file}",
                        self.debug.print_load_all_watermarks)
            filename = file.lower()
            if filename.endswith(".png"):
                watermark_id = os.path.splitext(file)[0]
                if watermark_id.startswith("map_"):
                    self.watermark_maps[watermark_id] = self._load_watermark_image(
                        file)
                else:
                    try:
                        # Attempt to convert watermark_id to integer if possible
                        watermark_id = int(watermark_id)
                    except ValueError:
                        pass  # Keep as string if not an integer
                    self.watermarks[watermark_id] = self._load_watermark_image(
                        file)

        if self.debug.print_load_all_watermarks:
            print(f"Loaded {len(self.watermarks)} watermarks " +
                  f"and {len(self.watermark_maps)} watermark maps" +
                  "into memory.")

    def _load_watermark_image(self, filename: str, alpha: float = 1.0) -> Image.Image:
        """
        Loads a single watermark image, adjusts its alpha, and swaps channels if necessary.

        :param filename: Name of the watermark file.
        :param alpha: Transparency level (0.0 to 1.0).
        :return: Processed watermark Image.
        """
        filepath = os.path.join(self.data_path, filename)
        if self.debug.print_load_watermark_image:
            print(f"Loading watermark from: {filepath} with alpha {alpha}")

        watermark = Image.open(filepath).convert("RGBA")

        # Adjust the alpha channel
        r, g, b, a_channel = watermark.split()
        if alpha != 1.0:
            # TODO: Check later for type hinting
            a_channel = a_channel.point(
                lambda i: int(i * alpha))  # type: ignore
        if self.swap_blue_red_channels:
            # Swap blue and red channels intentionally
            watermark = Image.merge("RGBA", (b, g, r, a_channel))
        else:
            watermark = Image.merge("RGBA", (r, g, b, a_channel))
        return watermark

    def get_watermark(
        self,
        watermark_id: Union[str, int],
        pool: ApplicationType = 'stamp',
        alpha: float = 1.0
    ) -> Image.Image:
        """
        Retrieves a watermark image by its ID, applying the specified alpha transparency.

        :param watermark_id: Identifier of the watermark.
        :param alpha: Transparency level (0.0 to 1.0).
        :return: Watermark Image with adjusted transparency.
        """
        if pool == 'stamp':
            watermark = self.watermarks.get(watermark_id)
        elif pool == 'map':
            watermark = self.watermark_maps.get(watermark_id)
        else:
            raise ValueError(f"Invalid pool: {pool}")

        if watermark is None:
            raise ValueError(f"Watermark with ID '{watermark_id}' not found.")

        if alpha != 1.0:
            # Create a copy to adjust alpha without modifying the cached watermark
            watermark = watermark.copy()
            r, g, b, a_channel = watermark.split()
            # TODO: Check later for type hinting
            a_channel = a_channel.point(
                lambda i: int(i * alpha))  # type: ignore
            watermark = Image.merge("RGBA", (r, g, b, a_channel))

        return watermark

    def get_random_watermark_id(
        self,
        application_type: ApplicationType = "map",
        seed: Optional[int] = None
    ) -> Union[str, int]:
        """
        Retrieves a random watermark ID. Optionally, seeds the random number generator.

        :param seed: Seed for the random number generator.
        :return: Random watermark ID.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()
        if application_type == "stamp":
            return random.choice(list(self.watermarks.keys()))
        elif application_type == "map":
            return random.choice(list(self.watermark_maps.keys()))
        else:
            raise ValueError(f"Invalid application type: {application_type}")

    def prepare_watermark_stamp(
        self,
        watermark_id: Union[str, int],
        base_width: int,
        base_height: int,
        alpha: float = 1.0,
        scale: float = 1.0,
    ) -> Image.Image:
        """
        Prepares a watermark image for stamping on a base image.

        :param watermark_id: Identifier of the watermark.
        :param alpha: Transparency level (0.0 to 1.0).
        :param scale: Scaling factor for the watermark.
        :return: Watermark Image prepared for stamping.
        """
        watermark = self.get_watermark(watermark_id, pool='stamp', alpha=alpha)

        watermark_resized = watermark.resize(
            (
                int(watermark.width * scale),
                int(watermark.height * scale)
            ),
            resample=Image.Resampling.LANCZOS
        ) if scale != 1.0 else watermark

        # Crop a random section of the watermark if it is larger than the base image
        if watermark_resized.width > base_width or watermark_resized.height > base_height:
            x = random.randint(0, max(watermark_resized.width - base_width, 0))
            y = random.randint(
                0, max(watermark_resized.height - base_height, 0))
            watermark_resized = watermark_resized.crop(
                (x, y, min(x + base_width, watermark_resized.width),
                 min(y + base_height, watermark_resized.height))
            )

        return watermark_resized

    def prepare_watermark_map(
        self,
        watermark_id: Union[str, int],
        base_width: int,
        base_height: int,
        position: PositionType = "center",
        alpha: float = 1.0,
        scale: float = 1.0,
        random_seed: Optional[int] = None
    ) -> Image.Image:
        """
        Prepares a watermark image for overlaying on a base image as a map.

        :param watermark_id: Identifier of the watermark.
        :param base_width: Width of the base image.
        :param base_height: Height of the base image.
        :param alpha: Transparency level (0.0 to 1.0).
        :param scale: Scaling factor for the watermark.
        :return: Watermark Image prepared for overlaying as a map.
        """
        watermark = self.get_watermark(watermark_id, pool='map', alpha=alpha)
        if scale == 1.0:
            watermark_resized = watermark
        else:
            identifier = f"{watermark_id}_map_{scale}"
            if identifier in self.watermark_resize_cache:
                watermark_resized = self.watermark_resize_cache[identifier]
            else:
                watermark_resized = watermark.resize(
                    (
                        int(watermark.width * scale),
                        int(watermark.height * scale)
                    ),
                    resample=Image.Resampling.LANCZOS
                )
                self.watermark_resize_cache[identifier] = watermark_resized

        # Ensure the watermark is at least as large as the base image
        if watermark_resized.width < base_width or watermark_resized.height < base_height:
            watermark_resized = watermark_resized.resize(
                (
                    max(watermark_resized.width, base_width),
                    max(watermark_resized.height, base_height)
                ),
                resample=Image.Resampling.LANCZOS
            )
            print("Warning: Upscaling a watermark map was necessary!")

        if position == "center":
            x = (watermark_resized.width - base_width) // 2
            y = (watermark_resized.height - base_height) // 2
            position = (int(x), int(y))
        elif position == "random":
            if random_seed is not None:
                random.seed(random_seed)
            else:
                random.seed()
            x = random.randint(0, max(watermark_resized.width - base_width, 0))
            y = random.randint(
                0, max(watermark_resized.height - base_height, 0))
            position = (x, y)

        crop_box = (
            position[0],
            position[1],
            position[0] + base_width,
            position[1] + base_height
        )

        return watermark_resized.crop(crop_box)

    def apply_watermark(
        self,
        base_image: Image.Image,
        watermark_id: Union[str, int],
        scale: float = 1.0,
        alpha: float = 1.0,
        position: PositionType = "center",
        application_type: ApplicationType = "map",
        artifacts_config: Optional[ArtifactsConfig] = None,
        random_seed: Optional[int] = None,
    ) -> Image.Image:
        print_debug(f"Applying watermark at position {position} with scale {scale}x" +
                    f"Base image size: {base_image.size}",
                    self.debug.print_apply_watermark)

        base_w, base_h = base_image.size
        layer = Image.new("RGBA", base_image.size, color=0) # color=0 is fully transparent black        

        if application_type == "stamp":
            wm = self.prepare_watermark_stamp(
                watermark_id, base_w, base_h, alpha, scale
            )
            if position == "random":
                if random_seed is not None:
                    random.seed(random_seed)
                else:
                    random.seed()
                x = random.randint(0, max(base_w - wm.width, 0))
                y = random.randint(0, max(base_h - wm.height, 0))
                position = (x, y)
            elif not (isinstance(position, tuple) and len(position) == 2):
                raise ValueError(
                    f"Invalid position: {position}" +
                    "stamp mode only supports 'random' or (x, y) position"
                )
            layer.paste(wm, position, mask=wm)
        elif application_type == "map":
            wm = self.prepare_watermark_map(
                watermark_id, base_w, base_h, position, alpha, scale
            )
            if artifacts_config is not None:
                print_debug(f"Applying artifacts using config: {artifacts_config}",
                            self.debug.print_apply_watermark)
                result = apply_watermark_with_artifacts(
                    base_image,
                    wm,
                    artifacts_config.alpha,
                    artifacts_config.intensity,
                    artifacts_config.kernel_size
                )
                return result
            else:
                layer.paste(wm, (0, 0), mask=wm)
        else:
            raise ValueError(
                f"Invalid application type: {application_type}"
            )

        # Composite the watermark layer onto the base image
        result = Image.alpha_composite(base_image, layer)
        return result

    def add_watermark_generic(
        self,
        img_train: torch.Tensor,
        watermark_id: Optional[Union[str, int]] = None,
        occupancy: float = 0,
        self_supervision: bool = False,
        same_random_wm_seed: Optional[int] = None,
        scale: Union[float, Tuple[float, float]] = 1.0,
        # Ignored, if artifacts_config is provided as their alpha works differently
        alpha: float = 1.0,
        position: PositionType = "center",
        application_type: ApplicationType = "map",
        artifacts_config: Optional[ArtifactsConfig] = None,
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add watermark noise to images.

        This function applies watermarks to images, either in standalone mode for a single image,
        or in batch processing mode for multiple images.

        Args:
            img_train (torch.Tensor): Input image tensor(s), shape [N, C, H, W].
            watermark_manager (WatermarkManager): Instance of WatermarkManager for accessing watermarks.
            self_supervision (bool, optional): Whether to use self-supervision mode. Defaults to False.
            same_random_wm_seed (int, optional): Random seed or image index to use when self_supervision is True. Defaults to 0.
            scale_img (float, optional): Fixed scale for the watermark image. If None, random scaling is used. Defaults to None.
            fixed_position (Tuple[int, int], optional): Fixed position (x, y) to place the watermark. If None, random position is used. Defaults to None.

        Returns:
            torch.Tensor: Images with watermarks applied.
        """
        # Determine watermark ID
        if watermark_id is not None:
            selected_watermark_id = watermark_id
        else:
            if self_supervision:
                selected_watermark_id = self.get_random_watermark_id(
                    application_type, same_random_wm_seed)
            else:
                selected_watermark_id = self.get_random_watermark_id(
                    application_type, random_seed)

        # Convert input tensor to NumPy array and adjust dimensions
        img_train_np = img_train.cpu().numpy()  # Ensure tensor is on CPU
        n_images, _, img_h, img_w = img_train_np.shape

        # Randomly select an occupancy level between 0 and the specified occupancy
        occupancy = np.random.uniform(0, occupancy)

        # Rearrange dimensions for processing: [N, H, W, C]
        img_train_np = np.ascontiguousarray(
            np.transpose(img_train_np, (0, 2, 3, 1)))

        if self.debug.print_add_watermark_generic:
            print(
                "Adding watermark with " +
                f"occupancy {occupancy:.2f}%, " +
                f"scale {scale}, " +
                f"alpha {alpha}, " +
                f"position {position}" +
                f"application type {application_type}" +
                f"artifacts config {artifacts_config}" +
                f"random seed for watermark selection {same_random_wm_seed}" +
                f"random seed {random_seed}"
            )
            print(f"Selected watermark ID: {selected_watermark_id}")
            print(f"Input images size: {img_w}x{img_h}")
            print(f"Number of images: {n_images}\n\n")

        for i in range(n_images):
            # Convert the image to PIL format
            tmp = Image.fromarray(
                (img_train_np[i] * 255).astype(np.uint8)).convert("RGBA")

            # Initialize an empty image for counting occupied pixels
            img_for_cnt = Image.new("L", (img_w, img_h), 0)

            while True:
                # Determine scaling factor
                if isinstance(scale, tuple):
                    scale = random.uniform(scale[0], scale[1])

                # Apply the watermark to the image
                tmp = self.apply_watermark(
                    tmp,
                    selected_watermark_id,
                    scale,
                    alpha,
                    position,
                    application_type,
                    artifacts_config,
                    random_seed
                )

                if self.debug.show_previews_add_watermark_generic:
                    # show_tmp_img(tmp)
                    self.preview_manager.add_image(tmp)

                if occupancy != 0:
                    # Apply the watermark to the counting image
                    img_for_cnt = self.apply_watermark(
                        img_for_cnt.convert("RGBA"),
                        selected_watermark_id,
                        scale,
                        alpha,
                        position,
                        application_type,
                        artifacts_config,
                        random_seed
                    ).convert("L")
                    img_cnt = np.array(img_for_cnt)

                    # Check if the occupancy condition is met
                    if confirm_occupancy(img_cnt, occupancy):
                        # Update the image in the array
                        img_rgb = np.array(tmp).astype(np.float32) / 255.0
                        img_train_np[i] = img_rgb[:, :, :3]
                        break
                else:
                    img_rgb = np.array(tmp).astype(np.float32) / 255.0
                    img_train_np[i] = img_rgb[:, :, :3]
                    break

        # Rearrange dimensions back to original: [N, C, H, W]
        img_train_np = np.transpose(img_train_np, (0, 3, 1, 2))
        return torch.tensor(img_train_np, dtype=img_train.dtype, device=img_train.device)


def confirm_occupancy(img_cnt: np.ndarray, occupancy_ratio: float, debug: bool = False) -> bool:
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
    if sum_pixels > occupancy_threshold:
        threshold_exceeded = True
    else:
        threshold_exceeded = False

    if debug:
        print(f"Occupied pixels: {sum_pixels}, " +
              f"Total pixels: {total_pixels}, " +
              f"Occupancy threshold: {occupancy_threshold}"
              )

    return threshold_exceeded


def apply_overlay_with_artifacts(
    base_arr: np.ndarray,
    expanded_mask: np.ndarray,
    artifact_intensity: float
) -> np.ndarray:
    for i in range(3):  # For each RGB channel
        channel = base_arr[:, :, i]
        noise = np.random.normal(0, 1, channel.shape)
        artifact_mask = expanded_mask * noise * artifact_intensity
        channel += artifact_mask * channel  # Apply relative noise
        channel = np.clip(channel, 0, 255)
        base_arr[:, :, i] = channel
    return base_arr

def apply_watermark_with_artifacts(
    base: Image.Image,
    watermark: Image.Image,
    alpha: float,
    artifact_intensity: float,
    kernel_size: int,
    convert_to_linear_space: bool = False
) -> Image.Image:
    """
    Applies a watermark to the base image and introduces artifacts around the watermark area.

    Args:
        base (Image.Image): The base image.
        watermark (Image.Image): The watermark image.
        alpha (float, optional): Opacity of the watermark.
        artifact_intensity (float, optional): Intensity of the artifacts.
        kernel_size (int, optional): Size of the convolution kernel for mask expansion.

    Returns:
        Image.Image: Image with watermark and artifacts applied.
    """
    base_arr = np.array(base).astype(np.float32)
    overlay_arr = np.array(watermark).astype(np.float32)
    alpha_mask = overlay_arr[:, :, 3] / 255.0

    # Create convolution kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    expanded_mask = convolve(alpha_mask > 0, kernel) / kernel.sum()

    # Apply artifacts only where the mask is expanded
    base_arr = apply_overlay_with_artifacts(base_arr, expanded_mask, artifact_intensity)

    # Convert to linear color space
    if convert_to_linear_space:
        overlay_linear = srgb_to_linear(overlay_arr[:, :, :3] / 255.0)
        base_linear = srgb_to_linear(base_arr[:, :, :3] / 255.0)
    else:
        overlay_linear = overlay_arr[:, :, :3] / 255.0
        base_linear = base_arr[:, :, :3] / 255.0

    # Blend the images based on alpha
    mask_3d = np.stack([alpha_mask] * 3, axis=-1) * alpha
    blended = overlay_linear * mask_3d + base_linear * (1 - mask_3d)

    # Convert back to sRGB
    if convert_to_linear_space:
        blended_srgb = linear_to_srgb(blended) * 255.0
    else:
        blended_srgb = blended * 255.0
    blended_srgb = np.clip(blended_srgb, 0, 255).astype(np.uint8)

    # Combine with alpha channel
    result = np.dstack((blended_srgb, np.ones_like(
        alpha_mask) * 255)).astype(np.uint8)
    return Image.fromarray(result)


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


def main():
    testfile = input("Enter the image filename (test.jpg): ")
    if not testfile:
        testfile = "test.jpg"
    base = Image.open("data/test/clean/" + testfile).convert("RGBA")
    reference = Image.open(("data/test/watermarked/" + testfile).replace("source", "target")).convert("RGBA")
    wmm = WatermarkManager(swap_blue_red_channels=False)

    num_samples = 3
    width, height = base.size
    grid_width = width * 2
    grid_height = height * 2
    grid_image = Image.new('RGB', (grid_width, grid_height))

    previews = []
    for i in range(num_samples):
        result = wmm.apply_watermark(
            base,
            'map_train7_normal',
            scale=1.0,
            alpha=0.5,
            position="random",
            application_type="map",
            artifacts_config=ArtifactsConfig(
                alpha=0.66,
                intensity=1.0,
                kernel_size=7
            )
        )
        previews.append(result)
    
    grid_image.paste(reference, (0, 0))             # Top-left
    grid_image.paste(previews[0], (width, 0))       # Top-right
    grid_image.paste(previews[1], (0, height))      # Bottom-left
    grid_image.paste(previews[2], (width, height))  # Bottom-right

    grid_image.show()

if __name__ == "__main__":
    main()
