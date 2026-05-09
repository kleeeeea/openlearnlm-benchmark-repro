"""Progress Tracker for resumable execution"""

import json
import os
from typing import Set, Dict, Any
from datetime import datetime
from pathlib import Path


class ProgressTracker:
    """Track processing progress for resume support"""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.processed_ids: Set[int] = set()
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "last_updated": None,
            "start_time": None
        }
        self._load_progress()

    def _load_progress(self):
        """Load existing progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get("processed_ids", []))
                    self.stats = data.get("stats", self.stats)
                    print(f"Loaded progress: {len(self.processed_ids)} items already processed")
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")

    def save_progress(self):
        """Save current progress to file"""
        self.stats["last_updated"] = datetime.now().isoformat()
        data = {
            "processed_ids": list(self.processed_ids),
            "stats": self.stats
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def mark_processed(self, item_id: int, success: bool):
        """Mark item as processed (only successful items are skipped on resume)"""
        self.stats["total_processed"] += 1
        if success:
            self.processed_ids.add(item_id)  # Only add successful items
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1
            # Failed items are NOT added to processed_ids, so they can be retried

    def is_processed(self, item_id: int) -> bool:
        """Check if item was already processed"""
        return item_id in self.processed_ids

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        success_rate = 0
        if self.stats["total_processed"] > 0:
            success_rate = self.stats["successful"] / self.stats["total_processed"] * 100
        return {
            **self.stats,
            "success_rate": success_rate
        }

    def sync_with_output_files(self, output_dir: Path):
        """Sync progress with existing output files to prevent duplicates on resume"""
        import json
        from pathlib import Path

        output_ids = set()
        batch_num = 1
        while True:
            batch_file = output_dir / f"questions_batch_{batch_num:03d}.jsonl"
            if not batch_file.exists():
                break
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if 'item_id' in item:
                            output_ids.add(item['item_id'])
            except Exception as e:
                print(f"Warning: Could not read {batch_file}: {e}")
            batch_num += 1

        # Find IDs in output but not in progress
        missing_ids = output_ids - self.processed_ids
        if missing_ids:
            print(f"[Sync] Found {len(missing_ids)} IDs in output files but not in progress. Adding them.")
            self.processed_ids.update(missing_ids)
            self.stats["successful"] += len(missing_ids)
            self.stats["total_processed"] += len(missing_ids)
            self.save_progress()
        else:
            print(f"[Sync] Progress and output files are already in sync.")
