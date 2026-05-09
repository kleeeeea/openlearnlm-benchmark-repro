"""Evaluation Orchestrator for Benchmark Evaluation"""

import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..config import EvalConfig
from ..results.progress_tracker import EvalProgressTracker
from ..results.result_writer import EvalResultWriter
from .evaluator import ModelEvaluator


class EvaluationOrchestrator:
    """Orchestrate evaluation across multiple models"""

    def __init__(self, config: EvalConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or self._setup_logger()

        # Initialize components
        self.progress_tracker = EvalProgressTracker(config.LOGS_DIR)
        self.result_writer = EvalResultWriter(config.RESPONSES_DIR)

        # Create evaluators for each model
        self.evaluators: Dict[str, ModelEvaluator] = {}
        for model_id in config.MODELS:
            self.evaluators[model_id] = ModelEvaluator(model_id, config)

        # Thread-safe counters
        self._processed_count = 0
        self._counter_lock = threading.Lock()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("evaluation")
        logger.setLevel(logging.INFO)

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console)

        # File handler
        log_file = self.config.LOGS_DIR / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

        return logger

    def load_test_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load test dataset"""
        items = []

        self.logger.info(f"Loading test data from {self.config.TEST_DATA_FILE}")

        with open(self.config.TEST_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                items.append(item)

                if limit and len(items) >= limit:
                    break

        self.logger.info(f"Loaded {len(items)} test items")
        return items

    def _evaluate_single(
        self,
        evaluator: ModelEvaluator,
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single item with a single model"""
        item_id = item.get("item_id", 0)
        model_id = evaluator.model_id

        # Skip if already completed
        if self.progress_tracker.is_completed(item_id, model_id):
            return None

        # Add delay for rate limiting
        time.sleep(self.config.REQUEST_DELAY)

        # Evaluate
        result = evaluator.evaluate_item(item)

        # Log failures immediately
        if not result.get("success", False):
            self.logger.warning(
                f"FAILED: {model_id} | item_id={item_id} | error={result.get('error', 'Unknown')}"
            )

        # Record progress
        is_correct = result.get("is_correct", False)
        self.progress_tracker.mark_completed(item_id, model_id, is_correct)

        # Write result
        self.result_writer.write_result(result)

        # Update counter
        with self._counter_lock:
            self._processed_count += 1

        return result

    def run(
        self,
        items: List[Dict[str, Any]],
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on all items with all models.

        Args:
            items: List of benchmark items
            resume: Whether to resume from previous progress

        Returns:
            Evaluation summary
        """
        start_time = time.time()
        total_evaluations = len(items) * len(self.evaluators)

        self.logger.info("=" * 60)
        self.logger.info("OpenLearnLM Benchmark Evaluation")
        self.logger.info("=" * 60)
        self.logger.info(f"Total items: {len(items)}")
        self.logger.info(f"Models: {', '.join(self.evaluators.keys())}")
        self.logger.info(f"Total evaluations: {total_evaluations}")
        self.logger.info(f"Resume mode: {resume}")
        self.logger.info("=" * 60)

        # Sync progress with existing files
        if resume:
            self.progress_tracker.sync_with_response_files()

        # Calculate pending evaluations
        pending_count = 0
        for model_id in self.evaluators:
            pending = self.progress_tracker.get_pending_items(items, model_id)
            pending_count += len(pending)
            self.logger.info(f"  {model_id}: {len(pending)} pending")

        self.logger.info(f"Total pending evaluations: {pending_count}")

        if pending_count == 0:
            self.logger.info("All evaluations already completed!")
            return self._generate_summary()

        # Run evaluations in parallel
        total_workers = len(self.evaluators) * self.config.WORKERS_PER_MODEL
        self.logger.info(f"Starting evaluation with {total_workers} workers...")

        self._processed_count = 0
        checkpoint_interval = self.config.CHECKPOINT_INTERVAL

        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = {}

            # Submit all evaluation tasks
            for item in items:
                for model_id, evaluator in self.evaluators.items():
                    item_id = item.get("item_id", 0)

                    # Skip if already completed
                    if resume and self.progress_tracker.is_completed(item_id, model_id):
                        continue

                    future = executor.submit(self._evaluate_single, evaluator, item)
                    futures[future] = (item_id, model_id)

            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                item_id, model_id = futures[future]

                try:
                    result = future.result()
                    completed += 1

                    # Progress logging
                    if completed % 100 == 0 or completed == len(futures):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(futures) - completed) / rate if rate > 0 else 0

                        self.logger.info(
                            f"Progress: {completed}/{len(futures)} "
                            f"({completed/len(futures)*100:.1f}%) | "
                            f"Rate: {rate:.1f}/s | "
                            f"ETA: {eta/60:.1f}min"
                        )

                    # Checkpoint
                    if completed % checkpoint_interval == 0:
                        self.progress_tracker.save_progress()

                except Exception as e:
                    self.logger.error(f"Error evaluating item {item_id} with {model_id}: {e}")

        # Final save
        self.progress_tracker.save_progress()
        self.result_writer.close()

        # Generate summary
        elapsed = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Evaluation completed in {elapsed/60:.1f} minutes")

        summary = self._generate_summary()
        self._log_summary(summary)

        return summary

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary"""
        summary = {
            "config": {
                "models": self.config.MODELS,
                "thinking_budget": self.config.THINKING_BUDGET_TOKENS,
                "temperature": self.config.TEMPERATURE,
            },
            "results": {},
            "by_dimension": {
                "by_difficulty": {},
                "by_domain": {},
                "by_question_type": {},
            }
        }

        # Model-level statistics
        for model_id, evaluator in self.evaluators.items():
            summary["results"][model_id] = evaluator.get_stats()

        return summary

    def _log_summary(self, summary: Dict[str, Any]):
        """Log evaluation summary"""
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)

        for model_id, stats in summary["results"].items():
            self.logger.info(f"\n{model_id}:")
            self.logger.info(f"  Processed: {stats.get('processed', 0)}")
            self.logger.info(f"  Correct: {stats.get('correct', 0)}")
            self.logger.info(f"  Accuracy: {stats.get('accuracy', 0)*100:.2f}%")
            self.logger.info(f"  Avg Latency: {stats.get('avg_latency_ms', 0):.0f}ms")

        self.logger.info("=" * 60)
