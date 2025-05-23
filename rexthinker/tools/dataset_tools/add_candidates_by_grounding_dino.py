import argparse
import json
import os
from typing import Any, Dict, List, Optional, Union

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Add candidates by Grounding DINO")
    parser.add_argument(
        "--anno_path",
        type=str,
        required=True,
        help="Path to input annotation file in JSONL format",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory containing the images",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the output annotation file",
    )
    parser.add_argument(
        "--gdino_config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to Grounding DINO config",
    )
    parser.add_argument(
        "--gdino_weights",
        type=str,
        default="GroundingDINO/weights/groundingdino_swint_ogc.pth",
        help="Path to Grounding DINO weights",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text threshold for Grounding DINO",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Box threshold for Grounding DINO",
    )
    return parser.parse_args()


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def return_maximum_overlap(gt_box, candidate_boxes, min_iou=0.5):
    max_iou = 0.0
    best_box = None
    for box in candidate_boxes:
        iou = compute_iou(gt_box, box)
        if iou >= min_iou and iou > max_iou:
            max_iou = iou
            best_box = box
    return best_box


def gdino_load_image(image: Union[str, Image.Image]) -> torch.Tensor:
    """Load and transform image for Grounding DINO model.

    Args:
        image (Union[str, Image.Image]): Input image path or PIL Image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image, str):
        image_source = Image.open(image).convert("RGB")
    else:
        image_source = image
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image_transformed


def inference_gdino(
    image: Image.Image,
    prompts: List[str],
    gdino_model: Any,
    TEXT_TRESHOLD: float = 0.25,
    BOX_TRESHOLD: float = 0.35,
) -> torch.Tensor:
    """Process an image with Grounding DINO model to detect objects.

    Args:
        image (Image.Image): Input PIL image.
        prompts (List[str]): List of text prompts for object detection.
        gdino_model (Any): The Grounding DINO model instance.
        TEXT_TRESHOLD (float, optional): Text confidence threshold. Defaults to 0.25.
        BOX_TRESHOLD (float, optional): Box confidence threshold. Defaults to 0.35.

    Returns:
        torch.Tensor: Tensor containing detected object bounding boxes in format (x1, y1, x2, y2).
    """
    text_labels = ".".join(prompts)
    image_transformed = gdino_load_image(image)
    boxes, _, _ = predict(
        model=gdino_model,
        image=image_transformed,
        caption=text_labels,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )
    # the output boxes is in the format of (x,y,w,h), in [0,1]
    boxes = boxes * torch.tensor([image.width, image.height, image.width, image.height])
    # convert to the format of (x1,y1,x2,y2)
    boxes = torch.cat(
        (boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2), dim=1
    )
    return boxes.tolist()


def add_candidate_boxes(
    anno_path: str,
    image_root: str,
    save_path: str,
    gdino_model: Any,
    TEXT_TRESHOLD: float = 0.25,
    BOX_TRESHOLD: float = 0.25,
) -> None:
    """Add candidate boxes to annotations using Grounding DINO model.

    This function processes a custom dataset for referring expression comprehension task in Rex-Thinker-GRPO.
    The input annotation file should be in JSONL format, where each line contains a JSON object with the following structure:

    Example input format:
    ```json
    {
        "image_name": "COCO_train2014_000000292799.jpg",
        "ans": [82.92, 201.37, 479.07, 586.99], # x0, y0, x1, y1
        "category": "turkey",
        "referring": "two turkeys near a tree"
    }
    ```

    The output will add a "candidate_boxes" field to each annotation, containing detected boxes from Grounding DINO.

    Args:
        anno_path (str): Path to input annotation file in JSONL format.
        image_root (str): Root directory containing the images.
        save_path (str): Path to save the output annotation file.
        gdino_model (Any): The Grounding DINO model instance.
        TEXT_TRESHOLD (float, optional): Text confidence threshold. Defaults to 0.25.
        BOX_TRESHOLD (float, optional): Box confidence threshold. Defaults to 0.35.
    """
    with open(anno_path, "r") as f:
        annos = [json.loads(line) for line in f]
    for anno in annos:
        image_path = os.path.join(image_root, anno["image_name"])
        image = Image.open(image_path).convert("RGB")
        boxes = inference_gdino(
            image, [anno["category"]], gdino_model, TEXT_TRESHOLD, BOX_TRESHOLD
        )
        gt_box = anno["ans"]
        # we need to make sure that the gt_box is inside the candidate boxes
        best_box = return_maximum_overlap(gt_box, boxes)
        if best_box is None:
            boxes.append(gt_box)

        anno["candidate_boxes"] = [
            {"bbox": box, "category": anno["category"]} for box in boxes
        ]
    with open(save_path, "w") as f:
        for anno in annos:
            json.dump(anno, f)
            f.write("\n")


def main() -> None:
    """Main function to run the script."""
    args = parse_args()
    gdino_model = load_model(
        args.gdino_config,
        args.gdino_weights,
    ).to("cuda")
    add_candidate_boxes(
        args.anno_path,
        args.image_root,
        args.save_path,
        gdino_model,
        args.text_threshold,
        args.box_threshold,
    )


if __name__ == "__main__":
    main()
