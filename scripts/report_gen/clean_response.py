"""Clean response JSONL files used by the Streamlit comparison report."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESPONSES_ROOT = PROJECT_ROOT / "outputs" / "responses"
DEFAULT_EXPERIMENT_GROUP_RESPONSE_FILE = (
    RESPONSES_ROOT / "01_기능_skills" / "Qwen3-4B-Instruct-2507.jsonl"
)
DEFAULT_BASELINE_GROUP_RESPONSE_FILE = (
    RESPONSES_ROOT / "01_기능_skills" / "Qwen3-4B-Instruct-2507-Official.jsonl"
)
DEFAULT_RESPONSE_FILES = [
    DEFAULT_EXPERIMENT_GROUP_RESPONSE_FILE,
    DEFAULT_BASELINE_GROUP_RESPONSE_FILE,
]

FALLBACK_MARKER = "fallback"

@dataclass
class CleanStats:
    path: Path
    original_rows: int = 0
    kept_rows: int = 0
    invalid_json_rows: int = 0
    missing_item_rows: int = 0
    missing_response_rows: int = 0
    nan_score_rows: int = 0
    fallback_reasoning_rows: int = 0
    duplicate_rows: int = 0
    backup_path: Path | None = None

    @property
    def removed_rows(self) -> int:
        return self.original_rows - self.kept_rows


def score_is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "nan", "none", "null"}:
            return True
        try:
            return math.isnan(float(stripped))
        except ValueError:
            return True
    return False


def has_model_response(row: dict[str, Any]) -> bool:
    for field in ("model_answer", "raw_content"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def has_fallback_reasoning(row: dict[str, Any]) -> bool:
    check_result = row.get("check_result") or {}
    fallback_fields = [
        check_result.get("reasoning", ""),
        check_result.get("rubric_source", ""),
        check_result.get("check_type", ""),
    ]
    return any(
        isinstance(value, str) and FALLBACK_MARKER in value.lower()
        for value in fallback_fields
    )


def row_is_valid(row: dict[str, Any], stats: CleanStats) -> bool:
    if row.get("item_id") is None:
        stats.missing_item_rows += 1
        return False

    if not row.get("success", False) or not has_model_response(row):
        stats.missing_response_rows += 1
        return False

    score = (row.get("check_result") or {}).get("score")
    if score_is_nan(score):
        stats.nan_score_rows += 1
        return False

    if has_fallback_reasoning(row):
        stats.fallback_reasoning_rows += 1
        return False

    return True


def load_valid_rows(path: Path, stats: CleanStats) -> list[dict[str, Any]]:
    valid_rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats.original_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats.invalid_json_rows += 1
                continue
            if row_is_valid(row, stats):
                valid_rows.append(row)
    return valid_rows


def dedupe_keep_latest(rows: list[dict[str, Any]], stats: CleanStats) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    deduped_reversed: list[dict[str, Any]] = []

    for row in reversed(rows):
        item_id = row.get("item_id")
        if item_id in seen:
            stats.duplicate_rows += 1
            continue
        seen.add(item_id)
        deduped_reversed.append(row)

    return list(reversed(deduped_reversed))


def write_rows(path: Path, rows: list[dict[str, Any]], stats: CleanStats) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{timestamp}")
    shutil.copy2(path, backup_path)
    stats.backup_path = backup_path

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_file(path: Path, dry_run: bool = False) -> CleanStats:
    stats = CleanStats(path=path)
    valid_rows = load_valid_rows(path, stats)
    cleaned_rows = dedupe_keep_latest(valid_rows, stats)
    stats.kept_rows = len(cleaned_rows)

    if not dry_run and stats.removed_rows:
        write_rows(path, cleaned_rows, stats)

    return stats


def print_stats(stats: CleanStats, dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "CLEANED"
    print(f"[{mode}] {stats.path}")
    print(f"  original_rows        : {stats.original_rows}")
    print(f"  kept_rows            : {stats.kept_rows}")
    print(f"  removed_rows         : {stats.removed_rows}")
    print(f"  duplicate_rows       : {stats.duplicate_rows}")
    print(f"  invalid_json_rows    : {stats.invalid_json_rows}")
    print(f"  missing_item_rows    : {stats.missing_item_rows}")
    print(f"  missing_response_rows: {stats.missing_response_rows}")
    print(f"  nan_score_rows       : {stats.nan_score_rows}")
    print(f"  fallback_reasoning_rows: {stats.fallback_reasoning_rows}")
    if stats.backup_path:
        print(f"  backup               : {stats.backup_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove invalid and duplicate rows from benchmark response JSONL files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Response JSONL paths. Defaults to the report baseline and experiment files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without modifying files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or DEFAULT_RESPONSE_FILES

    for path in paths:
        if not path.exists():
            print(f"[SKIP] Missing file: {path}")
            continue
        stats = clean_file(path, dry_run=args.dry_run)
        print_stats(stats, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
