import argparse
import json
import threading

import gradio as gr
import numpy as np
import torch
from groundingdino.util.inference import load_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

from tools.inference_tools import (
    convert_boxes_from_absolute_to_qwen25_format,
    inference_gdino,
    postprocess_and_vis_inference_out,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="IDEA-Research/Rex-Thinker-GRPO-7B"
    )
    parser.add_argument(
        "--gdino_config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--gdino_weights",
        type=str,
        default="GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument(
        "--server_ip",
        type=str,
        default="192.168.81.138",
        help="IP address to bind the server to",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=2512,
        help="Port to run the server on",
    )
    return parser.parse_args()


def initialize_models(args):
    # Load GDINO model
    gdino_model = load_model(args.gdino_config, args.gdino_weights).to("cuda")
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

    return (gdino_model, processor, model)


def process_image_with_streaming(
    image,
    system_prompt,
    cate_name,
    referring_expression,
    draw_width,
    font_size,
    gdino_model,
    rexthinker_processor,
    rexthinker_model,
):
    """
    Process image with streaming-like updates using a regular function.
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Run GDINO inference
    gdino_boxes = inference_gdino(
        image,
        [cate_name],
        gdino_model,
        TEXT_TRESHOLD=0.25,
        BOX_TRESHOLD=0.25,
    )
    proposed_box = convert_boxes_from_absolute_to_qwen25_format(
        gdino_boxes, image.width, image.height
    )

    hint = json.dumps(
        {
            f"{cate_name}": proposed_box,
        }
    )
    question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {referring_expression} in the image."

    # compose input
    print(f"system_prompt: {system_prompt}")
    print(f"question: {question}")
    messages = [
        {
            "role": "system",
            "content": system_prompt,
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

    text = rexthinker_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = rexthinker_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    # Create placeholder visualization with GDINO results
    placeholder_gdino_vis = image.copy()
    try:
        import numpy as np

        from tools.inference_tools import visualize

        placeholder_gdino_vis = visualize(
            placeholder_gdino_vis,
            gdino_boxes,
            np.ones(len(gdino_boxes)),
            font_size=font_size,
            draw_width=draw_width,
        )
    except:
        pass

    # For now, let's use the standard generation approach
    # We can implement true streaming later with a more complex setup
    generated_ids = rexthinker_model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = rexthinker_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_text[0]

    # Now do post-processing with the complete text
    ref_vis_result, gdino_vis_result = postprocess_and_vis_inference_out(
        image,
        output_text,
        proposed_box,
        gdino_boxes,
        font_size=font_size,
        draw_width=draw_width,
        input_height=input_height,
        input_width=input_width,
    )

    return gdino_vis_result, ref_vis_result, output_text


def process_image_non_streaming(
    image,
    system_prompt,
    cate_name,
    referring_expression,
    draw_width,
    font_size,
    gdino_model,
    rexthinker_processor,
    rexthinker_model,
):
    """Non-streaming version for examples"""
    # Use the regular processing function
    return process_image_with_streaming(
        image,
        system_prompt,
        cate_name,
        referring_expression,
        draw_width,
        font_size,
        gdino_model,
        rexthinker_processor,
        rexthinker_model,
    )


def create_streaming_interface(models):
    """Create a streaming interface using a different approach"""
    (
        gdino_model,
        rexthinker_processor,
        rexthinker_model,
    ) = models

    def process_with_streaming(
        image,
        system_prompt,
        cate_name,
        referring_expression,
        draw_width,
        font_size,
    ):
        # Run GDINO inference
        gdino_boxes = inference_gdino(
            image,
            [cate_name],
            gdino_model,
            TEXT_TRESHOLD=0.25,
            BOX_TRESHOLD=0.25,
        )
        proposed_box = convert_boxes_from_absolute_to_qwen25_format(
            gdino_boxes, image.width, image.height
        )

        hint = json.dumps(
            {
                f"{cate_name}": proposed_box,
            }
        )
        question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {referring_expression} in the image."

        # compose input
        messages = [
            {
                "role": "system",
                "content": system_prompt,
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

        text = rexthinker_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = rexthinker_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        input_height = inputs["image_grid_thw"][0][1] * 14
        input_width = inputs["image_grid_thw"][0][2] * 14

        # Create GDINO visualization
        gdino_vis = image.copy()
        try:
            import numpy as np

            from tools.inference_tools import visualize

            gdino_vis = visualize(
                gdino_vis,
                gdino_boxes,
                np.ones(len(gdino_boxes)),
                font_size=font_size,
                draw_width=draw_width,
            )
        except:
            pass

        # Yield initial state with GDINO visualization
        yield gdino_vis, None, "Starting generation..."

        # Use streaming generation
        streamer = TextIteratorStreamer(
            rexthinker_processor.tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": 4096,
            "streamer": streamer,
            "do_sample": False,
        }

        # Start generation in a separate thread
        thread = threading.Thread(
            target=rexthinker_model.generate, kwargs=generation_kwargs
        )
        thread.start()

        # Stream text with reduced frequency to minimize flickering
        streaming_text = ""
        token_count = 0
        for new_text in streamer:
            streaming_text += new_text
            token_count += 1

            # Update every 5 tokens to reduce flickering further
            if token_count % 5 == 0:
                yield gdino_vis, None, streaming_text

        # Ensure final text is shown
        yield gdino_vis, None, streaming_text

        thread.join()

        # Now do post-processing with the complete text
        ref_vis_result, gdino_vis_result = postprocess_and_vis_inference_out(
            image,
            streaming_text,
            proposed_box,
            gdino_boxes,
            font_size=font_size,
            draw_width=draw_width,
            input_height=input_height,
            input_width=input_width,
        )

        # Final yield with complete visualizations
        yield gdino_vis_result, ref_vis_result, streaming_text

    return process_with_streaming


def create_demo(models):
    (
        gdino_model,
        rexthinker_processor,
        rexthinker_model,
    ) = models

    # Get the streaming function
    process_with_streaming = create_streaming_interface(models)

    with gr.Blocks() as demo:
        gr.Markdown("# Rex-Thinker Demo")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                gdino_prompt = gr.Textbox(
                    label="Object Category Name to get Candidate boxes",
                    placeholder="person",
                    value="person",
                )
                referring_prompt = gr.Textbox(
                    label="Referring Expression",
                    placeholder="person wearning red shirt and a black hat",
                    value="person wearning red shirt and a black hat",
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.",
                )
                with gr.Row():
                    draw_width = gr.Slider(
                        minimum=5.0,
                        maximum=100.0,
                        value=10.0,
                        step=1,
                        label="Draw Width for Visualization",
                    )
                    font_size = gr.Slider(
                        minimum=5.0,
                        maximum=100.0,
                        value=20.0,
                        step=1,
                        label="Font size for Visualization",
                    )
                run_button = gr.Button("Run with Streaming", variant="primary")

            with gr.Column():
                gdino_output = gr.Image(label="GroundingDINO Detection")
                final_output = gr.Image(label="Rex-Thinker Visualization")
            with gr.Column():
                llm_output = gr.Textbox(
                    label="LLM Raw Output", interactive=False, lines=50, max_lines=100
                )

        # Add examples section
        gr.Markdown("## Examples")
        examples = gr.Examples(
            examples=[
                [
                    "demo/example_images/demo_tomato.jpg",
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.",
                    "tomato",
                    "ripe tomato",
                    10,
                    20,
                ],
                [
                    "demo/example_images/demo_helmet.png",
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.",
                    "helmet",
                    "the forth helmet from left",
                    10,
                    20,
                ],
                [
                    "demo/example_images/demo_person.jpg",
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.",
                    "person",
                    "person in the red car but not driving",
                    10,
                    20,
                ],
                [
                    "demo/example_images/demo_letter.jpg",
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.",
                    "person",
                    "person wearing cloth that has two letters",
                    10,
                    20,
                ],
                [
                    "demo/example_images/demo_dog.jpg",
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.",
                    "dog",
                    "the dog sleep on the bed with a pot under its body",
                    10,
                    20,
                ],
            ],
            inputs=[
                input_image,
                system_prompt,
                gdino_prompt,
                referring_prompt,
                draw_width,
                font_size,
            ],
            outputs=[gdino_output, final_output, llm_output],
            fn=lambda img, sys, p1, p2, d, f: process_image_non_streaming(
                img,
                sys,
                p1,
                p2,
                d,
                f,
                gdino_model,
                rexthinker_processor,
                rexthinker_model,
            ),
            cache_examples=False,
        )

        # Run with streaming text and final visualizations
        run_button.click(
            fn=process_with_streaming,
            inputs=[
                input_image,
                system_prompt,
                gdino_prompt,
                referring_prompt,
                draw_width,
                font_size,
            ],
            outputs=[gdino_output, final_output, llm_output],
        )

    return demo


def main():
    args = parse_args()
    models = initialize_models(args)
    demo = create_demo(models)
    demo.launch(server_name=args.server_ip, server_port=args.server_port, share=True)


if __name__ == "__main__":
    main()
