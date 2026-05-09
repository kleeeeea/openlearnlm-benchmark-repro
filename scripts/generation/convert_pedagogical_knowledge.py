"""
Pedagogical Knowledge Data Conversion Script
- KICE.csv + Chile.csv → questions_train.jsonl + questions_test.jsonl
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# Path configuration
BASE_DIR = Path(__file__).parent.parent.parent / "benchmark_data" / "03_교육학지식_pedagogical"
KICE_PATH = BASE_DIR / "태깅 - KICE.csv"
CHILE_PATH = BASE_DIR / "태깅 - 칠레.csv"
OUTPUT_TRAIN = BASE_DIR / "questions_train.jsonl"
OUTPUT_TEST = BASE_DIR / "questions_test.jsonl"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def convert_kice(filepath: Path) -> list:
    """Convert KICE data"""
    converted = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create options array excluding empty choices
            options = []
            for key in ["a_eng", "b_eng", "c_eng", "d_eng"]:
                if row.get(key, "").strip():
                    options.append(row[key])

            # Convert answer to uppercase
            answer = row["answer"].upper()

            converted.append({
                "question": row["question_eng"],
                "options": options,
                "answer": answer,
                "item_id": f"kice_{row['id']}",  # use original id
                "metadata": {
                    "center": None,
                    "role": None,
                    "scenario": None,
                    "subject": "Education",
                    "difficulty": None,
                    "domain": "cognitive",
                    "question_type": "multiple_choice",
                    "language": "en",
                    "source": "kice",
                    "year": int(row["year"]) if row.get("year") else None,
                    "tag": row.get("Tag", "").strip() or None
                }
            })

    return converted


def convert_chile(filepath: Path) -> list:
    """Convert Chile data"""
    converted = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create options array excluding empty choices
            options = []
            for key in ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e", "answer_f", "answer_g"]:
                if row.get(key, "").strip():
                    options.append(row[key])

            converted.append({
                "question": row["question"],
                "options": options,
                "answer": row["correct_answer"],
                "item_id": f"chile_{row['question_id']}",  # use original question_id
                "metadata": {
                    "center": None,
                    "role": None,
                    "scenario": None,
                    "subject": "Education",
                    "difficulty": None,
                    "domain": "cognitive",
                    "question_type": "multiple_choice",
                    "language": "en",
                    "source": "chile",
                    "year": int(row["year"]) if row.get("year") else None,
                    "tag": row.get("태그", "").strip() or None,
                    "subdomain": row.get("pedagogical_subdomain", "").strip() or None,
                    "age_group": row.get("age_group", "").strip() or None,
                    "category": row.get("category", "").strip() or None,
                    "secondary_category": row.get("secondary_category", "").strip() or None
                }
            })

    return converted


def stratified_split(data: list, ratio: float, stratify_key: str) -> tuple:
    """Stratified train/test split"""
    random.seed(RANDOM_SEED)

    # Group by stratify_key
    groups = defaultdict(list)
    for item in data:
        key = item["metadata"].get(stratify_key, "unknown")
        if key is None:
            key = "unknown"
        groups[key].append(item)

    train_data = []
    test_data = []

    for key, items in groups.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * ratio))  # at least 1 in train
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
    print("Pedagogical Knowledge Data Conversion")
    print("=" * 60)

    # Load and convert data
    print("\n[1] Loading and converting data")

    kice_converted = convert_kice(KICE_PATH)
    print(f"  - KICE converted: {len(kice_converted)} items")

    chile_converted = convert_chile(CHILE_PATH)
    print(f"  - Chile converted: {len(chile_converted)} items")

    # Merge
    all_data = kice_converted + chile_converted
    print(f"  - Total: {len(all_data)} items")

    # Stratified split (by tag)
    print(f"\n[2] Train/Test split (80:20, stratified by tag)")
    train_data, test_data = stratified_split(all_data, TRAIN_RATIO, "tag")
    print(f"  - Train: {len(train_data)} items")
    print(f"  - Test: {len(test_data)} items")

    # Save
    print("\n[3] Saving")
    save_jsonl(train_data, OUTPUT_TRAIN)
    print(f"  - {OUTPUT_TRAIN}")

    save_jsonl(test_data, OUTPUT_TEST)
    print(f"  - {OUTPUT_TEST}")

    # Verify
    print("\n[4] Verification")

    # KICE sample
    kice_sample = next((d for d in train_data if d["metadata"]["source"] == "kice"), None)
    if kice_sample:
        print(f"  - KICE sample:")
        print(f"    {json.dumps(kice_sample, ensure_ascii=False, indent=2)[:600]}...")

    # Chile sample
    chile_sample = next((d for d in train_data if d["metadata"]["source"] == "chile"), None)
    if chile_sample:
        print(f"\n  - Chile sample:")
        print(f"    {json.dumps(chile_sample, ensure_ascii=False, indent=2)[:600]}...")

    print("\n" + "=" * 60)
    print("Conversion completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
