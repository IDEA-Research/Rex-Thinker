import os
import re
from datetime import datetime
from typing import Dict, List


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"------------- {current_time} Format reward {bool(format_match)}-------------\n"
            )
            f.write(f"Content: {predict}\n")
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    def parse_json(json_output):
        pattern = r"\[([0-9\.]+(?:, ?[0-9\.]+)*)\]"
        matches = re.findall(pattern, json_output)
        coordinates = [
            [float(num) if "." in num else int(num) for num in match.split(",")]
            for match in matches
        ]
        return coordinates

    try:
        gt_box = parse_json(ground_truth)
        out_box = parse_json(predict)

        if len(gt_box) == 0 and len(out_box) == 0:
            reward = 1.0
            is_duplicate = False
        else:
            # if no duplicate boxes in out_box, we continue
            if len(out_box) == len(set(tuple(box) for box in out_box)):
                recall_list = [ob for ob in out_box if ob in gt_box]
                recall = len(recall_list) / len(gt_box) if len(gt_box) > 0 else 1.0
                precision = len(recall_list) / len(out_box) if len(out_box) > 0 else 1.0
                reward = (
                    2 * recall * precision / (recall + precision)
                    if (recall + precision) > 0
                    else 0.0
                )
                is_duplicate = False
            else:
                is_duplicate = True
                print("Duplicate boxes in out_box, skipping symbolic verification.")
                reward = 0.0
    except:
        reward = 0.0

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"------------- {current_time} Accuracy reward: {reward} is_duplicate {is_duplicate} -------------\n"
            )
            f.write(f"Content: {predict}\n")
            f.write(f"Solution: {ground_truth}\n")
    return reward


def compute_score(
    predicts: List[str], ground_truths: List[str], format_weight: float = 0.1
) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(
            r"\s*(<|>|/)\s*", r"\1", predict
        )  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
