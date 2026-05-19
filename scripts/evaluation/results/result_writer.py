"""Result writer — appends evaluation results to per-model JSONL files."""

import json
import threading
from pathlib import Path
from typing import Dict, Any, IO


class EvalResultWriter:
    """Write evaluation results to JSONL files, one file per model."""

    def __init__(self, responses_dir: Path):
        self.responses_dir = responses_dir
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self._files: Dict[str, IO] = {}
        self._lock = threading.Lock()

    def _get_file(self, model_id: str) -> IO:
        if model_id not in self._files:
            safe_name = model_id.replace("/", "_").replace(":", "_")
            path = self.responses_dir / f"{safe_name}.jsonl"
            self._files[model_id] = open(path, "a", encoding="utf-8")
        return self._files[model_id]

    def write_result(self, result: Dict[str, Any]):
        if result is None:
            return
        model_id = result.get("model", "unknown")
        with self._lock:
            f = self._get_file(model_id)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    def close(self):
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files.clear()
