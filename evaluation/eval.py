import argparse
import json
import os
import re

import torch
from PIL import Image
from qwen_vl_utils import smart_resize
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

THINK_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="IDEA-Research/Rex-Thinker-GRPO-7B",
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        default="data/IDEA-Research/HumanRef/annotations.jsonl",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="data/IDEA-Research/HumanRef/images",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="IDEA-Research/Rex-Thinker-GRPO-7B/eval_humanref/eval.jsonl",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=6001)
    parser.add_argument(
        "--system_prompt", type=str, default="You are a helpful assistant"
    )

    return parser.parse_args()


def inference(image, prompt, system_prompt="DEFAULT_PROMPT", max_new_tokens=2048):
    system_prompt = THINK_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    return output_text[0], input_height, input_width, text


def parse_json(json_output):
    pattern = r"\[([0-9\.]+(?:, ?[0-9\.]+)*)\]"

    # 提取所有匹配的列表（作为字符串）
    matches = re.findall(pattern, json_output)

    # 将每个匹配的列表字符串转换为数字列表
    coordinates = [
        [float(num) if "." in num else int(num) for num in match.split(",")]
        for match in matches
    ]

    return coordinates


def convert_boxes_from_absolute_to_qwen25_format(
    min_pixels, max_pixels, gt_boxes, ori_width, ori_height
):
    resized_height, resized_width = smart_resize(
        ori_height,
        ori_width,
        28,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_gt_boxes = []
    for box in gt_boxes:
        # resize the box
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


if __name__ == "__main__":
    args = get_args()
    if not (args.start_idx == -1 and args.end_idx == -1):
        args.save_path = args.save_path.replace(
            ".jsonl",
            f"_{args.start_idx}_{args.end_idx}.jsonl",
        )

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    # promtp for none candidate
    prompt_wo_candidate = "Locate <REFERRING> in the image and answer the question in json format. If there is no such object, please answer with 'None'."
    prompt_w_candidate = "<CANDIDATES>\nLocate <REFERRING> in the image and answer the question in json format."

    with open(args.anno_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]

    lines = lines[args.start_idx : args.end_idx]
    predictions = []
    for line in tqdm(lines):
        id = line["id"]
        image_name = line["image_name"]
        image_path = os.path.join(args.image_root_dir, image_name)
        image = Image.open(image_path)
        w, h = image.size

        referring = line["referring"]
        candidate_boxes = line["candidate_boxes"]

        candidate_boxes = convert_boxes_from_absolute_to_qwen25_format(
            args.min_pixels, args.max_pixels, candidate_boxes, w, h
        )
        hint = json.dumps({"person": candidate_boxes})
        # dump_dict = {}
        # for i, box in enumerate(candidate_boxes):
        #     box_name = f"Person {i + 1}"
        #     dump_dict[box_name] = box
        # hint = json.dumps({"person": dump_dict})
        prompt = prompt_w_candidate.replace("<REFERRING>", referring)
        prompt = prompt.replace("<CANDIDATES>", f"Hint: {hint}")

        output, input_height, input_width, text = inference(
            image,
            prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            json_output = parse_json(output)
            final_boxes = []
            input_height = input_height.item()
            input_width = input_width.item()
            for box in json_output:
                x0, y0, x1, y1 = box
                x0 = x0 / input_width * w
                y0 = y0 / input_height * h
                x1 = x1 / input_width * w
                y1 = y1 / input_height * h

                final_boxes.append([x0, y0, x1, y1])
            prediction = {
                "id": id,
                "extracted_predictions": final_boxes,
                "prompt": prompt,
                "raw_response": output,
            }
        except Exception as e:
            print(f"Parse faile, error is {e}")
            prediction = {
                "id": id,
                "extracted_predictions": [],
                "prompt": prompt,
                "raw_response": output,
            }
        predictions.append(prediction)

    with open(args.save_path, "a") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")
