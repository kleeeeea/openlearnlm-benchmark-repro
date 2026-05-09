"""
Content Knowledge Data Conversion Script
- cj_eval_hf_800.json + gpqa_main_120.json → questions_train.jsonl + questions_test.jsonl
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Path configuration
BASE_DIR = Path(__file__).parent.parent.parent / "benchmark_data" / "02_교과지식_content"
CJ_EVAL_PATH = BASE_DIR / "cj_eval_hf_800.json"
GPQA_PATH = BASE_DIR / "gpqa_main_120.json"
OUTPUT_TRAIN = BASE_DIR / "questions_train.jsonl"
OUTPUT_TEST = BASE_DIR / "questions_test.jsonl"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def convert_cj_eval(data: list) -> tuple:
    """Convert cj_eval data (exclude items with 0 options)"""
    converted = []
    skipped = []
    for idx, item in enumerate(data, start=1):
        # Create options array excluding empty choices
        options = []
        for key in ["A", "B", "C", "D", "E"]:
            if item.get(key, "").strip():
                options.append(item[key])

        item_id = f"cj_eval_{idx:04d}"

        # Exclude items with 0 options
        if len(options) == 0:
            skipped.append({
                "item_id": item_id,
                "question": item["question"][:50] + "...",
                "reason": "0 options (all A~E are empty)"
            })
            continue

        converted.append({
            "question": item["question"],
            "options": options,
            "answer": item["answer"],
            "item_id": item_id,
            "metadata": {
                "center": None,
                "role": None,
                "scenario": None,
                "subject": item["subject"],
                "difficulty": None,  # cj_eval has no difficulty field
                "domain": "cognitive",
                "question_type": "multiple_choice",
                "language": "en",
                "source": "cj_eval"
            }
        })
    return converted, skipped


def convert_gpqa(data: list) -> list:
    """Convert GPQA data"""
    converted = []
    for idx, item in enumerate(data, start=1):
        # Create options array excluding empty choices
        options = []
        for key in ["A", "B", "C", "D", "E"]:
            if item.get(key, "").strip():
                options.append(item[key])

        record = {
            "question": item["question"],
            "options": options,
            "answer": item["answer"],
            "item_id": f"gpqa_{idx:04d}",
            "metadata": {
                "center": None,
                "role": None,
                "scenario": None,
                "subject": item["subject"],
                "difficulty": item.get("difficulty"),
                "domain": "cognitive",
                "question_type": "multiple_choice",
                "language": "en",
                "source": "gpqa"
            }
        }

        # Add explanation as top-level field (if exists)
        if item.get("explanation"):
            record["explanation"] = item["explanation"]

        converted.append(record)
    return converted


def stratified_split(data: list, ratio: float, stratify_key: str) -> tuple:
    """Stratified train/test split"""
    random.seed(RANDOM_SEED)

    # Group by stratify_key
    groups = defaultdict(list)
    for item in data:
        key = item["metadata"].get(stratify_key, "unknown")
        groups[key].append(item)

    train_data = []
    test_data = []

    for key, items in groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * ratio)
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])

    # Final shuffle
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def save_jsonl(data: list, filepath: Path):
    """Save data to JSONL file"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("Content Knowledge Data Conversion")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data")
    with open(CJ_EVAL_PATH, "r", encoding="utf-8") as f:
        cj_eval_data = json.load(f)
    print(f"  - cj_eval: {len(cj_eval_data)} items")

    with open(GPQA_PATH, "r", encoding="utf-8") as f:
        gpqa_data = json.load(f)
    print(f"  - gpqa: {len(gpqa_data)} items")

    # Convert
    print("\n[2] Converting data")
    cj_converted, cj_skipped = convert_cj_eval(cj_eval_data)
    print(f"  - cj_eval converted: {len(cj_converted)} items")
    if cj_skipped:
        print(f"  - cj_eval skipped: {len(cj_skipped)} items (data quality issues)")
        for item in cj_skipped:
            print(f"    - {item['item_id']}: {item['reason']}")

    gpqa_converted = convert_gpqa(gpqa_data)
    print(f"  - gpqa converted: {len(gpqa_converted)} items")

    # Merge
    all_data = cj_converted + gpqa_converted
    print(f"  - Total: {len(all_data)} items")

    # Stratified split
    print(f"\n[3] Train/Test split (80:20, stratified by subject)")
    train_data, test_data = stratified_split(all_data, TRAIN_RATIO, "subject")
    print(f"  - Train: {len(train_data)} items")
    print(f"  - Test: {len(test_data)} items")

    # Save
    print("\n[4] Saving")
    save_jsonl(train_data, OUTPUT_TRAIN)
    print(f"  - {OUTPUT_TRAIN}")

    save_jsonl(test_data, OUTPUT_TEST)
    print(f"  - {OUTPUT_TEST}")

    # Verify
    print("\n[5] Verification")
    print(f"  - Train sample:")
    print(f"    {json.dumps(train_data[0], ensure_ascii=False, indent=2)[:500]}...")

    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
