#!/bin/bash

MODEL_PATH="IDEA-Research/Rex-Thinker-GRPO-7B"
ANNO_PATH="data/IDEA-Research/HumanRef/annotations.jsonl"
IMAGE_ROOT_DIR="data/IDEA-Research/HumanRef/images"
SAVE_PATH="IDEA-Research/Rex-Thinker-GRPO-7B/eval_humanref/"
MAX_NEW_TOKENS=2048
TOTAL_SAMPLES=6000
NUM_GPUS=8
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))
EXP_NAME="Rex-Thinker-GRPO-7B"

# Create tasks for each GPU
for i in $(seq 0 $((NUM_GPUS-1))); do
    START_IDX=$((i * SAMPLES_PER_GPU))
    END_IDX=$((START_IDX + SAMPLES_PER_GPU))
    
    # Create output directory
    mkdir -p $(dirname "${SAVE_PATH}")
    
    # Launch task
    CUDA_VISIBLE_DEVICES=$i python evaluation/eval.py \
        --model_path $MODEL_PATH \
        --anno_path $ANNO_PATH \
        --image_root_dir $IMAGE_ROOT_DIR \
        --save_path "${SAVE_PATH}eval_${START_IDX}_${END_IDX}.jsonl" \
        --max_new_tokens $MAX_NEW_TOKENS \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --system_prompt "THINK_PROMPT" &
done

# Wait for all tasks to complete
wait

echo "All evaluation tasks completed!"

# Merge all jsonl files
echo "Merging all jsonl files..."
cat ${SAVE_PATH}eval_*_*.jsonl > ${SAVE_PATH}eval_merged.jsonl

# Clean up temporary files
echo "Cleaning up temporary files..."
rm ${SAVE_PATH}eval_*_*.jsonl

# Run evaluation metrics
echo "Calculating evaluation metrics..."
python evaluation/metric.py \
    --gt_path $ANNO_PATH \
    --pred_path "${SAVE_PATH}eval_merged.jsonl" \
    --pred_names $EXP_NAME \
    --dump_path $SAVE_PATH

echo "All done! Final results saved to ${SAVE_PATH}eval_merged.jsonl"
echo "Evaluation metrics saved to ${SAVE_PATH}comparison.md and ${SAVE_PATH}comparison.json"
