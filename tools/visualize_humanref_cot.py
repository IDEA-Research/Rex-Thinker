import copy
import json
import os
import random
from base64 import b64decode
from io import BytesIO

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


def parse_args():
    """Parse command line arguments for the visualization script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - img_tsv (str): Path to image TSV file
            - ann_tsv (str): Path to annotation TSV file
            - ann_lineidx (str): Path to annotation lineidx file
            - idx (int): Index of the sample to visualize
            - output (str): Output path for visualization image
    """
    parser = argparse.ArgumentParser(
        description="Visualize human reference data with reasoning process"
    )
    parser.add_argument(
        "--img_tsv",
        type=str,
        default="data/humanref_cot.images.tsv",
        help="Path to image TSV file",
    )
    parser.add_argument(
        "--ann_tsv",
        type=str,
        default="data/humanref_cot.annotations.tsv",
        help="Path to annotation TSV file",
    )
    parser.add_argument(
        "--ann_lineidx",
        type=str,
        default="data/humanref_cot.annotations.tsv.lineidx",
        help="Path to annotation lineidx file",
    )
    parser.add_argument(
        "--num_vis", type=int, default=50, help="number of data to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vis/",
        help="Output path for visualization",
    )
    return parser.parse_args()


class TSVDataset(Dataset):
    """Dataset class for loading images and annotations from TSV files.

    This dataset class handles loading of images and annotations from TSV format files,
    where images are stored as base64 encoded strings and annotations are stored as JSON.

    Args:
        img_tsv_file (str): Path to the TSV file containing images
        ann_tsv_file (str): Path to the TSV file containing annotations
        ann_lineidx_file (str): Path to the line index file for annotations

    Attributes:
        data (list): List of line indices for annotations
        img_handle (file): File handle for image TSV file
        ann_handle (file): File handle for annotation TSV file
        img_tsv_file (str): Path to image TSV file
        ann_tsv_file (str): Path to annotation TSV file
    """

    def __init__(self, img_tsv_file: str, ann_tsv_file: str, ann_lineidx_file: str):
        super(TSVDataset, self).__init__()
        self.data = []
        f = open(ann_lineidx_file)
        for line in f:
            self.data.append(int(line.strip()))
        # shuffle(self.data)
        random.shuffle(self.data)

        self.img_handle = None
        self.ann_handle = None
        self.img_tsv_file = img_tsv_file
        self.ann_tsv_file = ann_tsv_file

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (image, data_dict) where:
                - image (PIL.Image): RGB image
                - data_dict (dict): Dictionary containing:
                    - gt_boxes (list): List of bounding boxes [x0, y0, x1, y1]
                    - region_map (dict): Mapping from referring expressions to box indices
                    - think (str): Reasoning process text
        """
        ann_line_idx = self.data[idx]

        if self.ann_handle is None:
            self.ann_handle = open(self.ann_tsv_file)
        self.ann_handle.seek(ann_line_idx)

        img_line_idx, ann = self.ann_handle.readline().strip().split("\t")
        img_line_idx = int(img_line_idx)
        if self.img_handle is None:
            self.img_handle = open(self.img_tsv_file)
        self.img_handle.seek(img_line_idx)
        img = self.img_handle.readline().strip().split("\t")[1]
        if img.startswith("b'"):
            img = img[1:-1]
        img = BytesIO(b64decode(img))
        image = Image.open(img).convert("RGB")
        data_dict = json.loads(ann)

        return image, data_dict


def visualize(image, data_dict, output_path="visualization.png"):
    """Visualize an image with bounding boxes and reasoning process.

    This function creates a visualization with two panels:
    - Left panel: Original image with ground truth boxes (red) and answer boxes (green)
    - Right panel: Reasoning process text

    Args:
        image (PIL.Image): Input image to visualize
        data_dict (dict): Dictionary containing:
            - gt_boxes (list): List of bounding boxes [x0, y0, w, h]
            - region_map (dict): Mapping from referring expressions to box indices
            - think (str): Reasoning process text
        output_path (str, optional): Path to save the visualization. Defaults to "visualization.png".
    """
    # Create figure with two subplots side by side
    plt.rcParams["figure.dpi"] = 300
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display image on the left subplot
    ax1.imshow(image)

    # Plot all ground truth boxes in red with indices
    gt_boxes = data_dict.get("gt_boxes", [])
    for idx, box in enumerate(gt_boxes):
        x0, y0, width, height = box

        # Create rectangle patch
        rect = patches.Rectangle(
            (x0, y0), width, height, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax1.add_patch(rect)

        # Add index number
        ax1.text(
            x0,
            y0 - 5,
            str(idx),
            color="red",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # Plot answer boxes from region_map in green
    region_map = data_dict.get("region_map", {})
    for referring_exp, answer_indices in region_map.items():
        # Display referring expression at the top of the image
        ax1.text(
            10,
            30,
            referring_exp,
            color="blue",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Plot answer boxes in green
        for idx in answer_indices:
            if idx < len(gt_boxes):
                box = gt_boxes[idx]
                x0, y0, width, height = box
                # Create rectangle patch for answer box
                rect = patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=3,
                    edgecolor="green",
                    facecolor="none",
                )
                ax1.add_patch(rect)

    # Remove axis ticks from image
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Image with Bounding Boxes")

    # Display reasoning text on the right subplot
    ax2.text(0.05, 0.95, data_dict.get("think", ""), wrap=True, fontsize=12, va="top")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Reasoning Process")

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    import argparse

    # Parse arguments
    args = parse_args()

    # Initialize dataset
    dataset = TSVDataset(args.img_tsv, args.ann_tsv, args.ann_lineidx)

    vis_root = args.output_dir
    os.makedirs(vis_root, exist_ok=True)
    for i in range(args.num_vis):
        image, data_dict = dataset[i]
        # Save the visualization
        output_path = os.path.join(vis_root, f"visualization_{i}.png")
        visualize(image, data_dict, output_path)
        print(f"Visualization saved to {output_path}")
