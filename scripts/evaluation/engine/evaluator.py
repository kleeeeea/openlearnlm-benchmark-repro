"""Model Evaluator for Benchmark Evaluation"""

import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..config import EvalConfig
from ..api.base_client import BaseModelClient, EvaluationRequest, EvaluationResponse
from ..api.openai_client import OpenAIDirectClient
from ..api.openrouter_client import OpenRouterClient
from .answer_checker import AnswerChecker


@dataclass
class EvaluatorStats:
    """Thread-safe statistics for a model evaluator"""
    processed: int = 0
    correct: int = 0
    failed: int = 0
    total_latency_ms: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, is_correct: bool, latency_ms: int, usage: Optional[Dict[str, int]] = None):
        """Record evaluation result (thread-safe)"""
        with self._lock:
            self.processed += 1
            if is_correct:
                self.correct += 1
            self.total_latency_ms += latency_ms

            if usage:
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)

    def record_failure(self):
        """Record failed evaluation (thread-safe)"""
        with self._lock:
            self.failed += 1

    @property
    def accuracy(self) -> float:
        """Calculate accuracy"""
        if self.processed == 0:
            return 0.0
        return self.correct / self.processed

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.processed == 0:
            return 0.0
        return self.total_latency_ms / self.processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "processed": self.processed,
            "correct": self.correct,
            "failed": self.failed,
            "accuracy": round(self.accuracy, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


class ModelEvaluator:
    """Evaluator for a single model"""

    def __init__(self, model_id: str, config: EvalConfig):
        self.model_id = model_id
        self.config = config
        self.client = self._create_client()
        self.stats = EvaluatorStats()
        self.checker = AnswerChecker(config)

    def _create_client(self) -> BaseModelClient:
        """Create the appropriate API client for the model"""
        client_type = self.config.get_model_client_type(self.model_id)

        if client_type == "openai":
            return OpenAIDirectClient(self.config, self.model_id)
        else:
            return OpenRouterClient(self.config, self.model_id)

    def evaluate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single benchmark item.

        Args:
            item: Benchmark item with question, answer, metadata

        Returns:
            Evaluation result dictionary
        """
        # Build evaluation request
        request = EvaluationRequest(
            item_id=item.get("item_id", 0),
            question=item.get("question", ""),
            expected_answer=item.get("answer", ""),
            question_type=item.get("metadata", {}).get("question_type", "multiple_choice"),
            metadata=item.get("metadata", {}),
            options=item.get("options")  # MCQ options (A, B, C, D)
        )

        # Call model API
        response: EvaluationResponse = self.client.evaluate(request)

        if not response.success:
            self.stats.record_failure()
            return {
                "item_id": request.item_id,
                "model": self.model_id,
                "success": False,
                "error": response.error,
                "latency_ms": response.latency_ms,
                "metadata": request.metadata,
            }

        # Check answer (LLM-as-Judge for long answers with scenario-specific rubric)
        scenario = request.metadata.get("scenario", "")
        check_result = self.checker.check_answer(
            model_answer=response.model_answer,
            expected_answer=request.expected_answer,
            question_type=request.question_type,
            question=request.question,
            scenario=scenario,
            metadata=request.metadata  # Pass metadata for attitude evaluation
        )

        # Record statistics
        self.stats.record(
            is_correct=check_result["is_correct"],
            latency_ms=response.latency_ms,
            usage=response.usage
        )

        return {
            "item_id": request.item_id,
            "model": self.model_id,
            "success": True,
            "question": request.question[:200] + "..." if len(request.question) > 200 else request.question,
            "expected_answer": request.expected_answer,
            "model_answer": response.model_answer,
            "raw_content": response.raw_content,
            "thinking_content": response.thinking_content,
            "is_correct": check_result["is_correct"],
            "check_result": check_result,
            "latency_ms": response.latency_ms,
            "usage": response.usage,
            "metadata": request.metadata,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current evaluation statistics"""
        return {
            "model": self.model_id,
            **self.stats.to_dict()
        }
