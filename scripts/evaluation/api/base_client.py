"""Base API Client for Model Evaluation"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from ..config import EvalConfig


@dataclass
class EvaluationRequest:
    """Evaluation request data"""
    item_id: int
    question: str
    expected_answer: str
    question_type: str  # "multiple choice" or "long answer"
    metadata: Dict[str, Any]
    options: Optional[List[str]] = None  # MCQ options (A, B, C, D)


@dataclass
class EvaluationResponse:
    """Model response result"""
    success: bool
    model_answer: Optional[str]
    raw_content: Optional[str]  # Full response content
    thinking_content: Optional[str]  # Reasoning process if available
    error: Optional[str]
    usage: Optional[Dict[str, int]]
    latency_ms: int


class BaseModelClient(ABC):
    """Abstract base class for model API clients"""

    def __init__(self, config: EvalConfig, model_id: str):
        self.config = config
        self.model_id = model_id

    @abstractmethod
    def evaluate(self, request: EvaluationRequest, retry_count: int = 0) -> EvaluationResponse:
        """
        Evaluate a single question and return the model's response.

        Args:
            request: The evaluation request containing question and metadata
            retry_count: Current retry attempt number

        Returns:
            EvaluationResponse with model's answer and metadata
        """
        pass

    @abstractmethod
    def get_thinking_params(self) -> Dict[str, Any]:
        """
        Get model-specific thinking budget parameters.

        Returns:
            Dictionary of parameters to include in API request
        """
        pass

    def build_messages(self, request: EvaluationRequest) -> List[Dict[str, str]]:
        """
        Build chat messages for the evaluation request.

        Args:
            request: The evaluation request

        Returns:
            List of message dictionaries for the chat API
        """
        if request.question_type == "multiple_choice":
            system_prompt = (
                "You are evaluating a multiple-choice question. "
                "Answer with ONLY the letter (A, B, C, or D) of the correct answer. "
                "Do not include any explanation or additional text."
            )
            # Build user message with options
            user_content = request.question
            if request.options:
                options_text = "\n".join(
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(request.options)
                )
                user_content = f"{request.question}\n\n{options_text}"
        else:
            system_prompt = (
                "You are answering an educational question. "
                "Provide a concise and accurate answer."
            )
            user_content = request.question

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def extract_answer(self, content: str, question_type: str) -> str:
        """
        Extract the answer from model response.

        Args:
            content: Raw response content from the model
            question_type: Type of question (multiple choice or long answer)

        Returns:
            Extracted answer string
        """
        if question_type == "multiple_choice":
            # Extract single letter answer (A, B, C, or D)
            content = content.strip().upper()

            # Try to find a single letter at the start
            if content and content[0] in "ABCD":
                return content[0]

            # Try to find patterns like "A.", "A)", "Answer: A", etc.
            import re
            patterns = [
                r'^([A-D])\.',
                r'^([A-D])\)',
                r'^([A-D]):',
                r'^([A-D])\s',
                r'answer\s*(?:is)?\s*[:=]?\s*([A-D])',
                r'\b([A-D])\b',
            ]

            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

            # Return first character if it looks like an answer
            return content[0] if content else ""
        else:
            # For long answer, return the full content
            return content.strip()
