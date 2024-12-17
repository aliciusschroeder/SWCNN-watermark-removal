from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
        :param swap_rb: Whether to swap the red and blue channels.
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
        Displays both pools side by side with Pool A on the left and Pool B on the right,
        separated by a vertical divider, and clears the pools afterward.
        """
        num_images_a = len(self.pool_a)
        num_images_b = len(self.pool_b)

        # Determine grid size for each pool
        # int(np.ceil(np.sqrt(num_images_a))) if num_images_a > 0 else 1
        grid_size_a = num_images_a
        # int(np.ceil(np.sqrt(num_images_b))) if num_images_b > 0 else 1
        grid_size_b = num_images_b

        cols = 3
        # Determine the number of rows based on the larger grid size
        rows = max(grid_size_a, grid_size_b) // cols + 1
        width_ratios = [1] * cols + [0.05] + [1] * cols

        # Create a figure with GridSpec: 3 columns (Pool A, Divider, Pool B)
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(
            rows, cols*2+1, width_ratios=width_ratios, wspace=0.05)

        # Add images to Pool A (Column 0)
        for i, (img, meta) in enumerate(self.pool_a):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.axis('off')
            if meta:
                ax.set_title(meta, fontsize=10)

        # Add images to Pool B (Column 2)
        for i, (img, meta) in enumerate(self.pool_b):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, cols+col+1])
            ax.imshow(img)
            ax.axis('off')
            if meta:
                ax.set_title(meta, fontsize=10)

        # Add vertical divider
        ax_divider = fig.add_subplot(gs[:, cols])
        ax_divider.axis('off')  # Hide axis
        # Draw a vertical line
        ax_divider.plot([0.5, 0.5], [0, 1], color='black', linewidth=1)

        plt.tight_layout()
        plt.show()

        # Clear both pools after showing
        self.pool_a.clear()
        self.pool_b.clear()


if __name__ == "__main__":
    pm = PreviewManager()
    for i in range(16):
        pm.add_image(Image.new("RGBA", (255, 255), color="red")) # type: ignore
    pm.show_pools()
