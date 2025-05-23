import argparse
import json
import os

import torch
from groundingdino.util.inference import load_model
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tools.inference_tools import (
    convert_boxes_from_absolute_to_qwen25_format,
    inference_gdino,
    postprocess_and_vis_inference_out,
)

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."


def get_args():
    parser = argparse.ArgumentParser(description="Inference script for Qwen-2.5-VL")
    parser.add_argument(
        "--image_path",
        type=str,
        default="demo/example_images/demo_helmet.png",
        help="Path to the input image",
    )
    parser.add_argument(
        "--cate_name",
        type=str,
        default="helmet",
        help='text prompt for grounding dino, e.g. "cat", "dog", "car"',
    )
    parser.add_argument(
        "--ref_exp",
        type=str,
        default="the forth helmet from left",
        help="Reference expression for Rex-Thinker, e.g. 'the cat on the left'",
    )
    parser.add_argument(
        "--vis_path",
        type=str,
        default="vis/example_output.jpg",
        help="Path to save the visualization result",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="IDEA-Research/Rex-Thinker-GRPO-7B",
        help="Path to the input image",
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
        "--qwen_model_path",
        type=str,
        default="IDEA-Research/Rex-Thinker-GRPO-7B",
        help="Path to Qwen-2.5-VL model or model identifier from Hugging Face Hub",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    image_path = args.image_path
    cate_name = args.cate_name
    ref_exp = args.ref_exp
    gdino_config = args.gdino_config
    gdino_weights = args.gdino_weights
    qwen_model_path = args.qwen_model_path

    # Load the Grounding DINO model
    gdino_model = load_model(gdino_config, gdino_weights)
    gdino_model.eval()

    # Load Rex-Thinker-GRPO
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=16 * 28 * 28, max_pixels=1280 * 28 * 28
    )

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Prepare the text prompts for Grounding DINO
    prompts = [cate_name]

    # Run inference with Grounding DINO to get box hint
    gdino_boxes = inference_gdino(image, prompts, gdino_model)

    proposed_box = convert_boxes_from_absolute_to_qwen25_format(
        gdino_boxes, image.width, image.height
    )
    hint = json.dumps(
        {
            f"{cate_name}": proposed_box,
        }
    )
    question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {ref_exp} in the image."

    # compose input
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_text[0]
    print(output_text)

    ref_vis_result, gdino_vis_result = postprocess_and_vis_inference_out(
        image,
        output_text,
        proposed_box,
        gdino_boxes,
        font_size=20,
        draw_width=10,
        input_height=input_height,
        input_width=input_width,
    )

    # Create a new image with white background for the combined result
    combined_width = gdino_vis_result.width + ref_vis_result.width
    combined_height = max(gdino_vis_result.height, ref_vis_result.height)
    combined_image = Image.new("RGB", (combined_width, combined_height), "white")

    # Paste the images side by side
    combined_image.paste(gdino_vis_result, (0, 0))
    combined_image.paste(ref_vis_result, (gdino_vis_result.width, 0))

    # Add titles
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("tools/Tahoma.ttf", 30)

    # Add Grounding DINO title
    draw.text((10, 10), "Grounding DINO Output", fill="black", font=font)

    # Add Rex-Thinker title
    draw.text(
        (gdino_vis_result.width + 10, 10), "Rex-Thinker Output", fill="black", font=font
    )

    # Save the combined visualization result
    os.makedirs(os.path.dirname(args.vis_path), exist_ok=True)
    combined_image.save(args.vis_path)
