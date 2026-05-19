"""Progress tracker for benchmark evaluation based on response files."""

import json
import threading
from pathlib import Path
from typing import Dict, List, Any


class EvalProgressTracker:
    """Track which (item_id, model_id) pairs have been evaluated."""

    def __init__(self, logs_dir: Path, responses_dir: Path):
        self.logs_dir = logs_dir
        self.responses_dir = responses_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = logs_dir / "progress.json"
        # {model_id: {str(item_id): is_correct}}
        self._completed: Dict[str, Dict[str, bool]] = {}
        self._lock = threading.Lock()
        self._load_response_files()

    def _load_response_files(self):
        completed: Dict[str, Dict[str, bool]] = {}
        if not self.responses_dir.exists():
            self._completed = completed
            return

        for path in sorted(self.responses_dir.glob("*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    item_id = result.get("item_id")
                    if item_id is None:
                        continue

                    model_id = result.get("model") or path.stem
                    if model_id not in completed:
                        completed[model_id] = {}
                    completed[model_id][str(item_id)] = bool(result.get("is_correct", False))

        self._completed = completed

    def sync_with_response_files(self):
        """Reload completed evaluations from response JSONL files."""
        with self._lock:
            self._load_response_files()

    def is_completed(self, item_id, model_id: str) -> bool:
        with self._lock:
            return str(item_id) in self._completed.get(model_id, {})

    def mark_completed(self, item_id, model_id: str, is_correct: bool):
        with self._lock:
            if model_id not in self._completed:
                self._completed[model_id] = {}
            self._completed[model_id][str(item_id)] = is_correct

    def get_pending_items(self, items: List[Dict[str, Any]], model_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            done = self._completed.get(model_id, {})
        return [item for item in items if str(item.get("item_id", 0)) not in done]

    def save_progress(self):
        """Write a best-effort snapshot for inspection; response files remain authoritative."""
        with self._lock:
            data = {model: dict(items) for model, items in self._completed.items()}
        with open(self._progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
