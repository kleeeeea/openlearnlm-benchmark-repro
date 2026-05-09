"""Data Processor for CSV and JSONL handling"""

import csv
import json
from typing import Iterator, Dict, Any, Optional
from pathlib import Path

from prompt_builder import RowMetadata


class DataProcessor:
    """Read CSV and iterate over unprocessed rows"""

    def __init__(self, input_csv: Path):
        self.input_csv = input_csv

    def read_rows(self) -> Iterator[tuple]:
        """
        Iterate over CSV rows that need processing.
        Yields: (row_index, metadata, raw_row_dict)
        """
        with open(self.input_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for idx, row in enumerate(reader, start=1):
                # Skip incomplete rows
                if len(row) < 9:
                    continue

                # Skip rows that already have generated content (Solar-pro2 column)
                if len(row) >= 14 and row[13] and row[13].strip():
                    continue

                # Skip example rows or invalid data
                if not row[0].strip() or row[0].startswith("**") or row[0].startswith("(예시)"):
                    continue

                try:
                    metadata = RowMetadata(
                        center=row[0].strip(),
                        role=row[1].strip(),
                        scenario=row[2].strip(),
                        sub_scenario=row[3].strip() if len(row) > 3 else "",
                        subject=row[4].strip() if len(row) > 4 else "",
                        difficulty=row[5].strip() if len(row) > 5 else "Medium",
                        domain=row[6].strip() if len(row) > 6 else "cognitive",
                        bloom_level=row[7].strip() if len(row) > 7 else "Understanding",
                        question_type=row[8].strip() if len(row) > 8 else "long answer"
                    )

                    raw_dict = {
                        "center": row[0] if len(row) > 0 else "",
                        "role": row[1] if len(row) > 1 else "",
                        "scenario": row[2] if len(row) > 2 else "",
                        "sub_scenario": row[3] if len(row) > 3 else "",
                        "subject": row[4] if len(row) > 4 else "",
                        "difficulty": row[5] if len(row) > 5 else "",
                        "domain": row[6] if len(row) > 6 else "",
                        "bloom_level": row[7] if len(row) > 7 else "",
                        "question_type": row[8] if len(row) > 8 else "",
                    }

                    yield (idx, metadata, raw_dict)

                except Exception as e:
                    print(f"Warning: Skipping row {idx} due to error: {e}")
                    continue


class OutputWriter:
    """Write results to JSONL files with batch splitting"""

    def __init__(self, output_dir: Path, batch_size: int = 1000):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.current_batch = 1
        self.item_count = 0
        self.file = None

    def _get_batch_filename(self) -> Path:
        return self.output_dir / f"questions_batch_{self.current_batch:03d}.jsonl"

    def _count_lines(self, filepath: Path) -> int:
        """Count lines in a file"""
        if not filepath.exists():
            return 0
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _find_resume_position(self):
        """Find the correct batch and item count for resuming"""
        # Find existing batch files
        batch_num = 1
        while True:
            batch_file = self.output_dir / f"questions_batch_{batch_num:03d}.jsonl"
            if not batch_file.exists():
                break
            batch_num += 1

        # Go back to last existing file
        if batch_num > 1:
            batch_num -= 1
            last_file = self.output_dir / f"questions_batch_{batch_num:03d}.jsonl"
            line_count = self._count_lines(last_file)

            if line_count >= self.batch_size:
                # Last file is full, start new batch
                self.current_batch = batch_num + 1
                self.item_count = 0
            else:
                # Continue from last file
                self.current_batch = batch_num
                self.item_count = line_count

    def _open_new_batch(self):
        if self.file:
            self.file.close()
        self.file = open(self._get_batch_filename(), 'a', encoding='utf-8')

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._find_resume_position()
        self._open_new_batch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_item(self, item: Dict[str, Any]):
        """Write single item to JSONL, rotating files at batch_size"""
        self.file.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.file.flush()
        self.item_count += 1

        if self.item_count >= self.batch_size:
            self.current_batch += 1
            self.item_count = 0
            self._open_new_batch()

    def get_current_file(self) -> Path:
        """Return current batch file path"""
        return self._get_batch_filename()


class CSVUpdater:
    """Update original CSV with generated content"""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.updates: Dict[int, str] = {}  # row_idx -> generated content

    def add_update(self, row_idx: int, content: str):
        """Queue an update for later batch processing"""
        self.updates[row_idx] = content

    def apply_updates(self):
        """Apply all queued updates to CSV"""
        if not self.updates:
            return

        # Read all rows
        rows = []
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Apply updates (row_idx is 1-based, rows[0] is header)
        for row_idx, content in self.updates.items():
            if row_idx < len(rows):
                # Ensure row has enough columns
                while len(rows[row_idx]) < 14:
                    rows[row_idx].append("")
                rows[row_idx][13] = content

        # Write back
        with open(self.csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Updated {len(self.updates)} rows in CSV")
        self.updates.clear()
