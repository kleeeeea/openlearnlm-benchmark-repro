#!/usr/bin/env python3
"""
OpenLearnLM Benchmark Question Generator - Parallel Version

Usage:
    python generate_questions_parallel.py [--limit COUNT] [--workers N] [--no-resume]

Examples:
    python generate_questions_parallel.py --limit 100 --workers 5
    python generate_questions_parallel.py --workers 10  # Full run with 10 workers
"""

import argparse
import time
import sys
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging with file and console handlers"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generate_{timestamp}.log"

    # Clear existing handlers
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

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from api_client import OpenAIClient, APIResponse
from prompt_builder import PromptBuilder, RowMetadata
from validator import ResponseValidator
from progress_tracker import ProgressTracker
from data_processor import DataProcessor, OutputWriter
from korean_to_english_mapping import MappingNotFoundError, KoreanToEnglishTranslator


@dataclass
class UsageStats:
    """Track API usage and costs"""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    rate_limit_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_usage(self, usage: dict):
        with self._lock:
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)
            self.total_requests += 1

    def add_rate_limit_hit(self):
        with self._lock:
            self.rate_limit_hits += 1

    def get_summary(self) -> dict:
        with self._lock:
            total_tokens = self.total_prompt_tokens + self.total_completion_tokens
            # Upstage Solar Pro2 pricing (estimate): $0.003/1K tokens
            estimated_cost = total_tokens / 1000 * 0.003
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": total_tokens,
                "total_requests": self.total_requests,
                "rate_limit_hits": self.rate_limit_hits,
                "estimated_cost_usd": round(estimated_cost, 4)
            }


class RateLimiter:
    """Adaptive rate limiter with backoff"""

    def __init__(self, initial_delay: float = 0.5, max_delay: float = 30.0):
        self.delay = initial_delay
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            current_delay = self.delay
        time.sleep(current_delay)

    def increase_delay(self):
        with self._lock:
            self.delay = min(self.delay * 2, self.max_delay)
            print(f"[Rate Limit] Increased delay to {self.delay:.1f}s")

    def decrease_delay(self):
        with self._lock:
            self.delay = max(self.delay * 0.9, self.initial_delay)

    def reset_delay(self):
        with self._lock:
            self.delay = self.initial_delay


class AdaptiveWorkerManager:
    """Dynamically adjust worker count based on rate limit status"""

    def __init__(self, min_workers: int = 10, max_workers: int = 100, logger=None):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()

        # Rate limit tracking
        self.recent_rate_limits = 0
        self.recent_successes = 0
        self.check_interval = 50  # Check every N requests
        self.rate_limit_threshold = 10  # If > N rate limits in interval, reduce workers
        self.success_threshold = 45  # If > N successes in interval, try increasing

    def record_rate_limit(self):
        with self._lock:
            self.recent_rate_limits += 1
            self._maybe_adjust()

    def record_success(self):
        with self._lock:
            self.recent_successes += 1
            self._maybe_adjust()

    def _maybe_adjust(self):
        total = self.recent_rate_limits + self.recent_successes
        if total < self.check_interval:
            return

        # Analyze and adjust
        rate_limit_ratio = self.recent_rate_limits / total
        old_workers = self.current_workers

        if self.recent_rate_limits > self.rate_limit_threshold:
            # Too many rate limits - reduce workers
            self.current_workers = max(self.min_workers, int(self.current_workers * 0.7))
            if self.current_workers != old_workers:
                self.logger.info(f"[Adaptive] Too many rate limits ({self.recent_rate_limits}) -> reducing workers: {old_workers} -> {self.current_workers}")
        elif self.recent_successes > self.success_threshold and self.current_workers < self.max_workers:
            # Mostly successes - try increasing workers
            self.current_workers = min(self.max_workers, int(self.current_workers * 1.2))
            if self.current_workers != old_workers:
                self.logger.info(f"[Adaptive] Stable ({self.recent_successes} successes) -> increasing workers: {old_workers} -> {self.current_workers}")

        # Reset counters
        self.recent_rate_limits = 0
        self.recent_successes = 0

    def get_current_workers(self) -> int:
        with self._lock:
            return self.current_workers


class ParallelGenerator:
    """Parallel question generator with rate limiting and usage tracking"""

    def __init__(self, config: Config, num_workers: int = 5, adaptive: bool = False):
        self.config = config
        self.num_workers = num_workers
        self.adaptive = adaptive
        self.translator = KoreanToEnglishTranslator(strict_mode=False)
        self.prompt_builder = PromptBuilder()
        self.validator = ResponseValidator()
        self.usage_stats = UsageStats()
        self.rate_limiter = RateLimiter(initial_delay=0.5)
        self.worker_manager = None  # Initialized in run() with logger

        # Thread-safe counters
        self._processed_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._counter_lock = threading.Lock()

        # Shared client pool - create enough for max possible workers
        max_pool_size = 200 if adaptive else max(num_workers, 150)
        self._clients = [OpenAIClient(config) for _ in range(max_pool_size)]

    def _get_client(self, worker_id: int) -> OpenAIClient:
        return self._clients[worker_id % len(self._clients)]

    def _format_output(self, item_id: int, metadata: RowMetadata, generated: dict) -> dict:
        translation = self.translator.translate_metadata(
            center=metadata.center,
            scenario=metadata.scenario,
            sub_scenario=metadata.sub_scenario
        )
        return {
            "question": generated.get("question", ""),
            "answer": generated.get("answer", ""),
            "item_id": item_id,
            "metadata": {
                "center": translation.center_en,
                "role": metadata.role,
                "scenario": f"{translation.scenario_en} / {translation.sub_scenario_en}",
                "subject": metadata.subject,
                "difficulty": metadata.difficulty,
                "domain": metadata.domain,
                "question_type": metadata.question_type
            }
        }

    def _format_csv_content(self, generated: dict) -> str:
        question = generated.get("question", "")
        answer = generated.get("answer", "")
        return f"Question: {question}\n\nAnswer: {answer}"

    def _process_single_item(
        self,
        worker_id: int,
        row_idx: int,
        metadata: RowMetadata
    ) -> Tuple[int, bool, Optional[dict], Optional[str]]:
        """Process a single item. Returns (row_idx, success, output_item, csv_content)"""

        client = self._get_client(worker_id)

        # Build prompt
        try:
            messages = self.prompt_builder.build_prompt(metadata)
        except MappingNotFoundError as e:
            return (row_idx, False, None, None)

        # Call API with retry until success (max 50 attempts to avoid infinite loops)
        max_attempts = 50
        for attempt in range(1, max_attempts + 1):
            # Rate limiting
            self.rate_limiter.wait()

            response = client.send_request(messages)

            if response.success:
                # Track usage
                if response.usage:
                    self.usage_stats.add_usage(response.usage)

                # Validate response
                is_valid, parsed_data, error = self.validator.validate_response(
                    response.content,
                    question_type=metadata.question_type
                )

                if is_valid:
                    output_item = self._format_output(row_idx, metadata, parsed_data)
                    csv_content = self._format_csv_content(parsed_data)
                    self.rate_limiter.decrease_delay()
                    if self.worker_manager:
                        self.worker_manager.record_success()
                    return (row_idx, True, output_item, csv_content)
                else:
                    # Invalid response, retry
                    if attempt % 10 == 0:
                        print(f"Item {row_idx}: Invalid response at attempt {attempt} - {error}")
                    time.sleep(1)
                    continue
            else:
                error_str = str(response.error).lower()
                # Check for rate limit or connection errors (SSL, connection pool, etc.)
                is_overload_error = any(x in error_str for x in [
                    "429", "rate", "ssl", "connection", "timeout", "max retries"
                ])
                if is_overload_error:
                    self.usage_stats.add_rate_limit_hit()
                    self.rate_limiter.increase_delay()
                    if self.worker_manager:
                        self.worker_manager.record_rate_limit()

                # Log retry progress every 5 attempts
                if attempt % 5 == 0:
                    print(f"Item {row_idx}: Retry {attempt}/{max_attempts} - {str(response.error)[:50]}...")

                # Wait and retry with exponential backoff
                wait_time = min(self.config.RETRY_DELAY * (1 + attempt // 5), 30)
                time.sleep(wait_time)
                continue

        # If we reach here, max attempts exceeded
        print(f"Item {row_idx}: Failed after {max_attempts} attempts")
        return (row_idx, False, None, None)

    def run(
        self,
        limit: Optional[int] = None,
        resume: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        # Initialize
        if logger is None:
            logger = logging.getLogger(__name__)

        # Initialize adaptive worker manager if enabled
        if self.adaptive:
            self.worker_manager = AdaptiveWorkerManager(
                min_workers=10,
                max_workers=200,  # Limit to 200 to avoid SSL connection errors
                logger=logger
            )
            # Start with initial workers, will scale up if no rate limits
            self.worker_manager.current_workers = self.num_workers

        progress_file = self.config.OUTPUT_DIR / self.config.PROGRESS_FILE
        tracker = ProgressTracker(progress_file) if resume else ProgressTracker(Path("/dev/null"))

        # Sync progress with existing output files to prevent duplicates
        if resume:
            tracker.sync_with_output_files(self.config.OUTPUT_DIR)

        processor = DataProcessor(self.config.INPUT_CSV)

        # Collect items to process
        items_to_process: List[Tuple[int, RowMetadata]] = []
        for row_idx, metadata, _ in processor.read_rows():
            if resume and tracker.is_processed(row_idx):
                continue
            items_to_process.append((row_idx, metadata))
            if limit and len(items_to_process) >= limit:
                break

        total_items = len(items_to_process)
        if total_items == 0:
            logger.info("No items to process.")
            return

        mode_str = "ADAPTIVE MODE" if self.adaptive else "PARALLEL MODE"
        logger.info("=" * 60)
        logger.info(f"OpenLearnLM Question Generator - {mode_str}")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.MODEL_NAME}")
        logger.info(f"Workers: {self.num_workers}" + (" (adaptive)" if self.adaptive else ""))
        logger.info(f"Items to process: {total_items}")
        logger.info(f"Output dir: {self.config.OUTPUT_DIR}")
        logger.info(f"Batch size: {self.config.OUTPUT_BATCH_SIZE} items per file")
        logger.info(f"Resume mode: {resume}")
        logger.info("-" * 60)

        start_time = time.time()

        if self.adaptive:
            self._run_adaptive(items_to_process, total_items, tracker, logger, start_time)
        else:
            self._run_fixed(items_to_process, total_items, tracker, logger, start_time)

    def _run_fixed(self, items_to_process, total_items, tracker, logger, start_time):
        """Run with fixed number of workers"""
        with OutputWriter(self.config.OUTPUT_DIR, batch_size=self.config.OUTPUT_BATCH_SIZE) as writer:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                for i, (row_idx, metadata) in enumerate(items_to_process):
                    worker_id = i % self.num_workers
                    future = executor.submit(
                        self._process_single_item,
                        worker_id, row_idx, metadata
                    )
                    futures[future] = row_idx

                self._process_futures(futures, total_items, tracker, writer, logger, start_time, self.num_workers)

        self._print_summary(total_items, logger, start_time)

    def _run_adaptive(self, items_to_process, total_items, tracker, logger, start_time):
        """Run with adaptive worker count - use max workers like fixed mode for speed"""
        with OutputWriter(self.config.OUTPUT_DIR, batch_size=self.config.OUTPUT_BATCH_SIZE) as writer:
            # Use max_workers to maximize parallelism (same as fixed mode)
            max_workers = self.worker_manager.max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, (row_idx, metadata) in enumerate(items_to_process):
                    worker_id = i % max_workers
                    future = executor.submit(
                        self._process_single_item,
                        worker_id, row_idx, metadata
                    )
                    futures[future] = row_idx

                self._process_futures(futures, total_items, tracker, writer, logger, start_time, max_workers)

        self._print_summary(total_items, logger, start_time)

    def _process_futures(self, futures, total_items, tracker, writer, logger, start_time, display_workers):
        """Process completed futures and update progress"""
        checkpoint_count = 0
        for future in as_completed(futures):
            row_idx, success, output_item, csv_content = future.result()

            with self._counter_lock:
                self._processed_count += 1
                if success:
                    self._success_count += 1
                    if output_item:
                        writer.write_item(output_item)
                    tracker.mark_processed(row_idx, success=True)
                else:
                    self._failed_count += 1
                    tracker.mark_processed(row_idx, success=False)

                checkpoint_count += 1

                # Progress display every 100 items
                if self._processed_count % 100 == 0 or self._processed_count == total_items:
                    elapsed = time.time() - start_time
                    rate = self._processed_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_items - self._processed_count) / rate if rate > 0 else 0
                    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                    progress_pct = self._processed_count / total_items * 100

                    # Format ETA nicely
                    eta_hours = int(eta_seconds // 3600)
                    eta_mins = int((eta_seconds % 3600) // 60)
                    if eta_hours > 0:
                        eta_remain = f"{eta_hours}h {eta_mins}m remaining"
                    else:
                        eta_remain = f"{eta_mins}m remaining"

                    # Get current workers (adaptive or fixed)
                    current_workers = self.worker_manager.get_current_workers() if self.worker_manager else display_workers
                    pending = total_items - self._processed_count

                    logger.info(
                        f"Progress: {self._processed_count:,}/{total_items:,} ({progress_pct:.1f}%) | "
                        f"Speed: {rate:.1f}/s | "
                        f"Workers: {current_workers} | "
                        f"Pending: {pending:,} | "
                        f"ETA: {eta_time.strftime('%H:%M')} ({eta_remain})"
                    )

            # Checkpoint at configured interval
            if checkpoint_count >= self.config.CHECKPOINT_INTERVAL:
                tracker.save_progress()
                checkpoint_count = 0

        # Final save
        tracker.save_progress()

    def _print_summary(self, total_items, logger, start_time):
        """Print final summary"""
        elapsed = time.time() - start_time
        elapsed_hours = int(elapsed // 3600)
        elapsed_mins = int((elapsed % 3600) // 60)

        logger.info("=" * 60)
        logger.info("Generation completed!")
        logger.info("=" * 60)
        logger.info(f"Total processed: {self._processed_count:,} items")
        logger.info(f"Success: {self._success_count:,} | Failed: {self._failed_count:,}")
        if self._processed_count > 0:
            logger.info(f"Success rate: {self._success_count/self._processed_count*100:.1f}%")
        logger.info(f"Elapsed time: {elapsed_hours}h {elapsed_mins}m")
        if elapsed > 0:
            logger.info(f"Average speed: {self._processed_count/elapsed:.1f}/s")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="OpenLearnLM Benchmark Question Generator - Parallel Version"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of items to process"
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of parallel workers (default: 5, max for adaptive mode)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, don't resume from previous progress"
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Enable adaptive worker scaling (auto-adjust workers based on rate limits)"
    )

    args = parser.parse_args()

    config = Config()

    # Setup logging
    log_dir = config.PROJECT_ROOT / "3. Benchmark Prototype" / "generation_logs"
    logger = setup_logging(log_dir)

    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    generator = ParallelGenerator(config, num_workers=args.workers, adaptive=args.adaptive)
    generator.run(
        limit=args.limit,
        resume=not args.no_resume,
        logger=logger
    )


if __name__ == "__main__":
    main()
