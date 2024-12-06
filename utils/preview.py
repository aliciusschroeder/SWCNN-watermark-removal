from typing import Optional
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class PreviewManager:
    def __init__(self):
        self.pool_a = []  # List to store images and their metadata for Pool A
        self.pool_b = []  # List to store images and their metadata for Pool B
        self.image = 1  # Counter to keep track of the number of images added
        self.toggle = True  # Toggle to alternate between pools

    def add_image(
        self,
        img: Image.Image,
        meta: Optional[str] = None,
        swap_rb: bool = True
    ) -> None:
        """
        Add an image to a pool. Alternates between Pool A and Pool B.
        :param img: A PIL Image object.
        :param meta: Optional metadata string to be displayed above the image.
        """
        if swap_rb:
            r, g, b, a = img.split()
            img = Image.merge("RGBA", (b, g, r, a))
        if self.toggle:
            if meta is None:
                meta = f"Image {self.image}A"

            self.pool_a.append((img, meta))
        else:
            if meta is None:
                meta = f"Image {self.image}B"
            self.image += 1
            self.pool_b.append((img, meta))
        self.toggle = not self.toggle

    def show_pools(self) -> None:
        """
        Displays both pools side by side in a grid and clears the pools afterward.
        """
        # Define grid dimensions
        num_images_a = len(self.pool_a)
        num_images_b = len(self.pool_b)
        grid_size_a = int(np.ceil(np.sqrt(num_images_a)))
        grid_size_b = int(np.ceil(np.sqrt(num_images_b)))

        # Create subplots for Pool A and Pool B
        fig, axes = plt.subplots(
            max(grid_size_a, grid_size_b),
            grid_size_a + grid_size_b,
            figsize=(15, 8),
        )
        axes = axes.flatten()

        # Display images from Pool A
        for i, (img, meta) in enumerate(self.pool_a):
            ax = axes[i]
            ax.imshow(img)
            ax.axis("off")
            if meta:
                ax.set_title(meta, fontsize=10)

        # Display images from Pool B
        offset = grid_size_a
        for i, (img, meta) in enumerate(self.pool_b):
            ax = axes[offset + i]
            ax.imshow(img)
            ax.axis("off")
            if meta:
                ax.set_title(meta, fontsize=10)

        # Hide unused axes
        for ax in axes[len(self.pool_a) + len(self.pool_b):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        # Clear both pools after showing
        self.pool_a.clear()
        self.pool_b.clear()
