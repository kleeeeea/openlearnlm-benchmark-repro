#!/usr/bin/env python3
"""
OpenLearnLM Quality Checker - LLM-as-Judge Evaluation

Usage:
    python quality_checker.py [--limit COUNT] [--workers N] [--no-resume]

Examples:
    python quality_checker.py --limit 100 --workers 50   # Test with 100 items
    python quality_checker.py --workers 100              # Full quality check
"""

import argparse
import json
import time
import sys
import threading
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from api_client import OpenAIClient
from quality_rubric import build_judge_prompt, MCQ_PASS_THRESHOLD, MCQ_CRITICAL_THRESHOLDS


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"quality_check_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


@dataclass
class QualityProgress:
    """Track quality check progress"""
    processed_ids: set = field(default_factory=set)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "total_processed": 0,
        "passed": 0,
        "failed": 0,
    })
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_processed(self, item_id: int) -> bool:
        with self._lock:
            return item_id in self.processed_ids

    def mark_processed(self, item_id: int, passed: bool):
        with self._lock:
            self.processed_ids.add(item_id)
            self.stats["total_processed"] += 1
            if passed:
                self.stats["passed"] += 1
            else:
                self.stats["failed"] += 1

    def save(self, path: Path):
        with self._lock:
            data = {
                "processed_ids": list(self.processed_ids),
                "stats": self.stats,
                "last_updated": datetime.now().isoformat()
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)

    def load(self, path: Path):
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.processed_ids = set(data.get("processed_ids", []))
                self.stats = data.get("stats", self.stats)


class QualityResultWriter:
    """Write quality check results to JSONL files"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = output_dir / f"quality_results_{timestamp}.jsonl"
        self.failed_file = output_dir / f"quality_failed_{timestamp}.jsonl"

        self._results_handle = open(self.results_file, 'a', encoding='utf-8')
        self._failed_handle = open(self.failed_file, 'a', encoding='utf-8')
        self._lock = threading.Lock()

    def write_result(self, result: Dict[str, Any]):
        with self._lock:
            self._results_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            self._results_handle.flush()

            if not result.get("evaluation", {}).get("pass", True):
                self._failed_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                self._failed_handle.flush()

    def close(self):
        self._results_handle.close()
        self._failed_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_generated_items(output_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load all generated items from JSONL batch files"""
    items = []
    batch_files = sorted(output_dir.glob("questions_batch_*.jsonl"))

    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    items.append(item)
                    if limit and len(items) >= limit:
                        return items

    return items


def parse_judge_response(response_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parse LLM judge response into structured evaluation result"""
    try:
        # Clean response - remove markdown code blocks if present
        cleaned = re.sub(r'```json\s*', '', response_text)
        cleaned = re.sub(r'```\s*', '', cleaned)

        # Try to extract JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        result = json.loads(cleaned)

        # Validate required fields
        if "scores" not in result:
            return (False, None, "Missing 'scores' field in response")

        scores = result["scores"]
        if not isinstance(scores, dict):
            return (False, None, "'scores' must be a dictionary")

        # Calculate total if not provided
        if "total_score" not in result:
            result["total_score"] = sum(scores.values())

        # Determine pass/fail if not provided
        if "pass" not in result:
            total = result["total_score"]
            answer_accuracy = scores.get("answer_accuracy", 0)
            question_clarity = scores.get("question_clarity", 0)

            result["pass"] = (
                total >= MCQ_PASS_THRESHOLD and
                answer_accuracy >= MCQ_CRITICAL_THRESHOLDS.get("answer_accuracy", 3) and
                question_clarity >= MCQ_CRITICAL_THRESHOLDS.get("question_clarity", 2)
            )

        return (True, result, None)

    except json.JSONDecodeError as e:
        return (False, None, f"JSON parsing error: {str(e)}")
    except Exception as e:
        return (False, None, f"Parse error: {str(e)}")


class QualityChecker:
    """Parallel quality checker using LLM-as-Judge"""

    def __init__(self, config: Config, num_workers: int = 50):
        self.config = config
        self.num_workers = num_workers

        # Thread-safe counters
        self._processed_count = 0
        self._pass_count = 0
        self._fail_count = 0
        self._counter_lock = threading.Lock()

        # Client pool
        self._clients = [OpenAIClient(config) for _ in range(num_workers)]

        # Rate limiting
        self._delay = 0.5
        self._delay_lock = threading.Lock()

    def _get_client(self, worker_id: int) -> OpenAIClient:
        return self._clients[worker_id % len(self._clients)]

    def _check_single_item(
        self,
        worker_id: int,
        item: Dict[str, Any]
    ) -> Tuple[int, bool, Optional[Dict[str, Any]]]:
        """Check quality of a single item. Returns (item_id, passed, result)"""

        item_id = item.get("item_id", 0)
        client = self._get_client(worker_id)

        # Build judge prompt
        messages = build_judge_prompt(item)

        # Call API with retry
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            # Rate limiting
            with self._delay_lock:
                current_delay = self._delay
            time.sleep(current_delay)

            response = client.send_request(messages)

            if response.success:
                # Parse response
                is_valid, evaluation, error = parse_judge_response(response.content)

                if is_valid:
                    result = {
                        "item_id": item_id,
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "metadata": item.get("metadata", {}),
                        "evaluation": evaluation
                    }
                    return (item_id, evaluation.get("pass", False), result)
                else:
                    if attempt % 3 == 0:
                        print(f"Item {item_id}: Parse error at attempt {attempt} - {error}")
                    time.sleep(1)
                    continue
            else:
                error_str = str(response.error).lower()
                if any(x in error_str for x in ["429", "rate", "ssl", "connection"]):
                    with self._delay_lock:
                        self._delay = min(self._delay * 1.5, 10.0)

                if attempt % 3 == 0:
                    print(f"Item {item_id}: API error at attempt {attempt}")

                time.sleep(min(2 * attempt, 15))
                continue

        # Failed after max attempts
        result = {
            "item_id": item_id,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "metadata": item.get("metadata", {}),
            "evaluation": {"pass": False, "error": "Max attempts exceeded"}
        }
        return (item_id, False, result)

    def run(
        self,
        items: List[Dict[str, Any]],
        progress: QualityProgress,
        writer: QualityResultWriter,
        logger: logging.Logger
    ):
        # Filter already processed items
        items_to_check = [
            item for item in items
            if not progress.is_processed(item.get("item_id", 0))
        ]

        total_items = len(items_to_check)
        if total_items == 0:
            logger.info("No items to check.")
            return

        logger.info("=" * 60)
        logger.info("OpenLearnLM Quality Checker - LLM-as-Judge")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.MODEL_NAME}")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Items to check: {total_items}")
        logger.info(f"Output: {writer.results_file}")
        logger.info("-" * 60)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for i, item in enumerate(items_to_check):
                worker_id = i % self.num_workers
                future = executor.submit(self._check_single_item, worker_id, item)
                futures[future] = item.get("item_id", i)

            checkpoint_count = 0
            for future in as_completed(futures):
                item_id, passed, result = future.result()

                with self._counter_lock:
                    self._processed_count += 1
                    if passed:
                        self._pass_count += 1
                    else:
                        self._fail_count += 1

                    checkpoint_count += 1

                # Save result
                if result:
                    writer.write_result(result)
                progress.mark_processed(item_id, passed)

                # Progress display every 100 items
                if self._processed_count % 100 == 0 or self._processed_count == total_items:
                    elapsed = time.time() - start_time
                    rate = self._processed_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_items - self._processed_count) / rate if rate > 0 else 0
                    progress_pct = self._processed_count / total_items * 100
                    pass_rate = self._pass_count / self._processed_count * 100 if self._processed_count > 0 else 0

                    eta_hours = int(eta_seconds // 3600)
                    eta_mins = int((eta_seconds % 3600) // 60)
                    if eta_hours > 0:
                        eta_str = f"{eta_hours}h {eta_mins}m"
                    else:
                        eta_str = f"{eta_mins}m"

                    logger.info(
                        f"Progress: {self._processed_count:,}/{total_items:,} ({progress_pct:.1f}%) | "
                        f"Pass rate: {pass_rate:.1f}% | "
                        f"Speed: {rate:.1f}/s | "
                        f"ETA: {eta_str}"
                    )

                # Checkpoint save every 500 items
                if checkpoint_count >= 500:
                    progress.save(self.config.OUTPUT_DIR / "quality_progress.json")
                    checkpoint_count = 0

        # Final save
        progress.save(self.config.OUTPUT_DIR / "quality_progress.json")

        # Summary
        elapsed = time.time() - start_time
        elapsed_mins = int(elapsed // 60)
        elapsed_secs = int(elapsed % 60)

        logger.info("=" * 60)
        logger.info("Quality check completed!")
        logger.info("=" * 60)
        logger.info(f"Total checked: {self._processed_count:,} items")
        logger.info(f"Passed: {self._pass_count:,} | Failed: {self._fail_count:,}")
        if self._processed_count > 0:
            logger.info(f"Pass rate: {self._pass_count/self._processed_count*100:.1f}%")
        logger.info(f"Elapsed time: {elapsed_mins}m {elapsed_secs}s")
        logger.info(f"Results file: {writer.results_file}")
        logger.info(f"Failed file: {writer.failed_file}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="OpenLearnLM Quality Checker - LLM-as-Judge Evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of items to check"
    )
    parser.add_argument(
        "--workers", type=int, default=50,
        help="Number of parallel workers (default: 50)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, don't resume from previous progress"
    )

    args = parser.parse_args()

    config = Config()

    # Setup logging
    log_dir = config.PROJECT_ROOT / "3. Benchmark Prototype" / "generation_logs"
    logger = setup_logging(log_dir)

    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    # Load generated items
    logger.info("Loading generated items...")
    items = load_generated_items(config.OUTPUT_DIR, limit=args.limit)
    logger.info(f"Loaded {len(items):,} items")

    # Initialize progress tracker
    progress = QualityProgress()
    progress_file = config.OUTPUT_DIR / "quality_progress.json"
    if not args.no_resume and progress_file.exists():
        progress.load(progress_file)
        logger.info(f"Resuming from {len(progress.processed_ids):,} previously checked items")

    # Run quality check
    with QualityResultWriter(config.OUTPUT_DIR) as writer:
        checker = QualityChecker(config, num_workers=args.workers)
        checker.run(items, progress, writer, logger)


if __name__ == "__main__":
    main()
