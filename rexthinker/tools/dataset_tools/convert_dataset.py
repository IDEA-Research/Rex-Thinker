import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from qwen_vl_utils import smart_resize
from tqdm import tqdm


def image_to_bytes(img_path: str) -> Optional[bytes]:
    """Convert an image file to bytes.

    Args:
        img_path (str): Path to the image file.

    Returns:
        Optional[bytes]: Image bytes in PNG format, or None if conversion fails.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        with BytesIO() as output:
            image.save(output, format="PNG")
            return output.getvalue()
    except Exception as e:
        return None


def process_line(line: Dict, image_root: str) -> Optional[Dict]:
    """Process a single data line and convert image to bytes.

    Args:
        line (Dict): A dictionary containing image and annotation data.
        image_root (str): Root directory containing the images.

    Returns:
        Optional[Dict]: Processed data with image bytes, or None if processing fails.
    """
    try:
        img_path = os.path.join(image_root, line["image"])
        if not os.path.exists(img_path):
            return None
        image_bytes = image_to_bytes(img_path)
        if image_bytes is None:
            return None
        return {
            "images": image_bytes,
            "problem": "<image>" + line["problem"],
            "answer": line["solution"],
        }
    except Exception as e:
        print(e)
        return None


def convert_boxes_from_absolute_to_qwen25_format(
    gt_boxes: List[List[float]],
    ori_width: int,
    ori_height: int,
    min_pixels: int = 16 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> List[List[int]]:
    """Convert bounding boxes from absolute coordinates to Qwen25 format.

    Args:
        gt_boxes (List[List[float]]): List of bounding boxes in [x0, y0, x1, y1] format.
        ori_width (int): Original image width.
        ori_height (int): Original image height.
        min_pixels (int, optional): Minimum number of pixels. Defaults to 16 * 28 * 28.
        max_pixels (int, optional): Maximum number of pixels. Defaults to 1280 * 28 * 28.

    Returns:
        List[List[int]]: Resized bounding boxes in Qwen25 format.
    """
    resized_height, resized_width = smart_resize(
        ori_height,
        ori_width,
        28,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_gt_boxes = []
    for box in gt_boxes:
        x0, y0, x1, y1 = box
        x0 = int(x0 / ori_width * resized_width)
        x1 = int(x1 / ori_width * resized_width)
        y0 = int(y0 / ori_height * resized_height)
        y1 = int(y1 / ori_height * resized_height)

        x0 = max(0, min(x0, resized_width - 1))
        y0 = max(0, min(y0, resized_height - 1))
        x1 = max(0, min(x1, resized_width - 1))
        y1 = max(0, min(y1, resized_height - 1))
        resized_gt_boxes.append([x0, y0, x1, y1])
    return resized_gt_boxes


def compose_answer(
    referring: str,
    gt_boxes: List[List[float]],
    ori_width: int,
    ori_height: int,
    min_pixels: int = 16 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> str:
    """Compose answer in JSON format with resized bounding boxes.

    Args:
        referring (str): Referring expression for the object.
        gt_boxes (List[List[float]]): Ground truth bounding boxes.
        ori_width (int): Original image width.
        ori_height (int): Original image height.
        min_pixels (int, optional): Minimum number of pixels. Defaults to 16 * 28 * 28.
        max_pixels (int, optional): Maximum number of pixels. Defaults to 1280 * 28 * 28.

    Returns:
        str: JSON formatted answer string.
    """
    answer = []
    resized_height, resized_width = smart_resize(
        ori_height,
        ori_width,
        28,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    for box in gt_boxes:
        x0, y0, x1, y1 = box
        x0 = int(x0 / ori_width * resized_width)
        x1 = int(x1 / ori_width * resized_width)
        y0 = int(y0 / ori_height * resized_height)
        y1 = int(y1 / ori_height * resized_height)

        x0 = max(0, min(x0, resized_width - 1))
        y0 = max(0, min(y0, resized_height - 1))
        x1 = max(0, min(x1, resized_width - 1))
        y1 = max(0, min(y1, resized_height - 1))
        answer.append(
            {
                "bbox_2d": [x0, y0, x1, y1],
                "label": referring,
            }
        )
    return f"```json\n{json.dumps(answer)}\n```"


def convert_custom_dataset(
    anno_path: str, image_root: str, save_path: str, max_workers: int = os.cpu_count()
) -> None:
    """Convert custom dataset to parquet format.

    This function processes a custom dataset for referring expression comprehension task in Rex-Thinker-GRPO.
    The input annotation file should be in JSONL format, where each line contains a JSON object with the following structure:

    Example input format:
    ```json
    {
        "image_name": "COCO_train2014_000000292799.jpg",
        "ans": [82.92, 201.37, 479.07, 586.99],
        "referring": "two turkeys near a tree",
        "candidate_boxes": [
            {
                "bbox": [82.92, 201.37, 479.07, 586.99],
                "category": "turkey"
            },
            {
                "bbox": [233.32, 164.58, 357.44, 344.42],
                "category": "turkey"
            },
            {
                "bbox": [233.58, 164.63, 451.35, 357.78],
                "category": "turkey"
            },
            {
                "bbox": [116.42, 212.40, 430.18, 536.94],
                "category": "turkey"
            }
        ]
    }
    ```

    Args:
        anno_path (str): Path to the annotation file in JSONL format.
        image_root (str): Root directory containing the images.
        save_path (str): Path to save the output parquet file.
        max_workers (int, optional): Number of worker processes. Defaults to CPU count.
    """
    with open(anno_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f.readlines()]

    # step1: compose boxes hints
    new_annos = []
    for line in lines:
        image_name = line["image_name"]
        referring = line["referring"]
        gt_boxes = line["ans"]
        candidate_boxes = line["candidate_boxes"]
        main_object = candidate_boxes[0]["category"]
        boxes = [can["bbox"] for can in candidate_boxes]
        image_path = os.path.join(image_root, image_name)
        width, height = Image.open(image_path).convert("RGB").size

        hint = json.dumps(
            {
                f"{main_object}": convert_boxes_from_absolute_to_qwen25_format(
                    boxes, width, height
                )
            }
        )
        answer = compose_answer(referring, [gt_boxes], width, height)
        problem = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {referring} in this image."
        solution = answer
        new_annos.append(
            {
                "image": os.path.basename(image_path),
                "problem": problem,
                "solution": solution,
                "ori_width": width,
                "ori_height": height,
            }
        )

    # step2: convert to parquet format
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_line, line, image_root) for line in new_annos
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            result = future.result()
            if result:
                records.append(result)

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, save_path)
    print(f"数据集已保存为 Parquet 格式: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom dataset to parquet format"
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        required=True,
        help="Path to the annotation file in JSONL format",
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
        help="Name of the saved parquet file",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes",
    )

    args = parser.parse_args()
    convert_custom_dataset(
        anno_path=args.anno_path,
        image_root=args.image_root,
        save_path=args.save_path,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
