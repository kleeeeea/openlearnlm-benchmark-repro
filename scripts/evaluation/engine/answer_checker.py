"""Answer Checker for Benchmark Evaluation

Supports scenario-specific rubrics for Long Answer evaluation using
10-point scale (band scoring: 9-10, 7-8, 5-6, 3-4, 1-2).

Also supports Attitude evaluation (question_type: "attitude", "attitude_deception")
using LLM-as-Judge with dimension-specific rubrics from metadata.

CSV Source: 2. Literature Review/LLM 역할표(1231)_1231.csv
"""

import re
import json
import requests
from typing import Tuple, Optional
from ..config import EvalConfig
from .rubric_loader import get_rubric_loader

SCORE_FIELD = "score"

class AnswerChecker:
    """Check if model's answer matches the expected answer"""

    def __init__(self, config: Optional[EvalConfig] = None):
        """Initialize with optional config for LLM-as-Judge"""
        self.config = config or EvalConfig()

    @staticmethod
    def normalize_mcq_answer(answer: str) -> str:
        """
        Normalize MCQ answer to single uppercase letter.

        Args:
            answer: Raw answer string

        Returns:
            Normalized single letter (A, B, C, or D) or empty string
        """
        if not answer:
            return ""

        answer = answer.strip().upper()

        # Direct single letter
        if len(answer) == 1 and answer in "ABCD":
            return answer

        # Patterns like "A.", "A)", "A:", "(A)"
        patterns = [
            r'^([A-D])\.',
            r'^([A-D])\)',
            r'^([A-D]):',
            r'^\(([A-D])\)',
            r'^([A-D])\s',
        ]

        for pattern in patterns:
            match = re.match(pattern, answer)
            if match:
                return match.group(1)

        # First letter if it's A-D
        if answer and answer[0] in "ABCD":
            return answer[0]

        return ""

    @staticmethod
    def check_mcq(model_answer: str, expected_answer: str) -> Tuple[bool, str, str]:
        """
        Check if MCQ answer is correct.

        Args:
            model_answer: Model's answer
            expected_answer: Expected correct answer

        Returns:
            Tuple of (is_correct, normalized_model_answer, normalized_expected_answer)
        """
        norm_model = AnswerChecker.normalize_mcq_answer(model_answer)
        norm_expected = AnswerChecker.normalize_mcq_answer(expected_answer)

        is_correct = norm_model == norm_expected and norm_model != ""

        return is_correct, norm_model, norm_expected

    def check_long_answer_llm(
        self,
        question: str,
        model_answer: str,
        expected_answer: str,
        scenario: str = ""
    ) -> Tuple[bool, float, str, str]:
        """
        Check long answer using LLM-as-Judge with scenario-specific rubric.

        Args:
            question: Original question
            model_answer: Model's answer
            expected_answer: Expected answer (reference)
            scenario: Scenario field for rubric lookup
                      (e.g., "Learner Analysis / Individual Learner Level Assessment")

        Returns:
            Tuple of (is_correct, score, reasoning, rubric_source)
            - score: 1-10 scale (10-point system)
            - rubric_source: "scenario_specific" or "generic"
        """
        if not model_answer or not expected_answer:
            return False, 0.0, "Empty answer", "generic"

        # Try to get scenario-specific rubric
        rubric_loader = get_rubric_loader()
        rubric_text, rubric_source = rubric_loader.get_rubric_text(scenario)

        if rubric_text:
            prompt = self._build_rubric_prompt(
                question, model_answer, expected_answer, rubric_text
            )
        else:
            prompt = self._build_generic_prompt(
                question, model_answer, expected_answer
            )

        try:
            # Use the configured judge API independently from the evaluated model API.
            headers = {
                "Authorization": f"Bearer {self.config.JUDGE_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config.JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 3000,
            }

            response = requests.post(
                f"{self.config.JUDGE_API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Parse JSON response
                # Handle markdown code blocks
                if content.startswith("```"):
                    content = re.sub(r'^```json?\n?', '', content)
                    content = re.sub(r'\n?```$', '', content)

                result = json.loads(content)
                score = result.get("score", 0)

                # Ensure score is in 1-10 range
                if score > 10:
                    score = score / 10.0  # Convert from 0-100 if needed

                is_correct = score >= 5  # Pass threshold: 5 out of 10
                reasoning = result.get("reasoning", "")
                band = result.get("band", self._score_to_band(score))

                return is_correct, score, f"[{band}] {reasoning}", rubric_source
            else:
                raise Exception("git_repos/openlearnlm-benchmark-17D4/scripts/evaluation/engine/answer_checker.py:169")
                # Fallback to word overlap on API error
                is_correct, score, reasoning = self._fallback_word_overlap(
                    model_answer, expected_answer
                )
                return is_correct, score, reasoning, "generic"

        except (json.JSONDecodeError, KeyError, Exception) as e:
            raise Exception(f"Error git_repos/openlearnlm-benchmark-17D4/scripts/evaluation/engine/answer_checker.py:177 {e}")
            # Fallback to word overlap on parse error
            # is_correct, score, reasoning = self._fallback_word_overlap(
            #     model_answer, expected_answer
            # )
            # return is_correct, score, reasoning, "generic"

    def _build_rubric_prompt(
        self,
        question: str,
        model_answer: str,
        expected_answer: str,
        rubric_text: str
    ) -> str:
        """Build prompt with scenario-specific rubric (10-point scale)"""
        return f"""You are an expert educational evaluator. Evaluate the student's answer using the specific rubric provided.

## Question
{question}

## Reference Answer
{expected_answer}

## Student's Answer
{model_answer}

## Evaluation Rubric (10-point scale)
{rubric_text}

## Instructions
1. Carefully compare the student's answer against the rubric criteria
2. Assign a score from 1-10 based on the rubric bands
3. The student's answer does NOT need to match the reference word-for-word
4. Don't reason for over 100 tokens. 

## Response Format (JSON only)
{{"score": <1-10>, "band": "<e.g., 7-8>", "reasoning": "<brief explanation based on rubric>"}}
"""

    def _build_generic_prompt(
        self,
        question: str,
        model_answer: str,
        expected_answer: str
    ) -> str:
        """Build generic prompt when no specific rubric available (10-point scale)"""
        return f"""You are an expert educational evaluator. Evaluate if the student's answer adequately addresses the question compared to the reference answer.

## Question
{question}

## Reference Answer
{expected_answer}

## Student's Answer
{model_answer}

## Evaluation Criteria (10-point scale)
9-10: Comprehensive analysis with clear evidence, accurate conclusions, and actionable recommendations
7-8: Most key elements correctly analyzed with valid conclusions, minor gaps in depth
5-6: Basic analysis present but limited evidence, conclusions somewhat general
3-4: Fragmented analysis, weak connections between conclusions and evidence
1-2: Failed to properly address the question, inaccurate or missing analysis

## Instructions
1. Does the student's answer address the main points of the question?
2. Is the answer factually consistent with the reference answer?
3. Does it demonstrate understanding of the key concepts?

Note: The student's answer does NOT need to match the reference word-for-word.

## Response Format (JSON only)
{{"score": <1-10>, "band": "<e.g., 7-8>", "reasoning": "<brief explanation>"}}
"""

    @staticmethod
    def _score_to_band(score: float) -> str:
        """Convert numeric score to band string"""
        if score >= 9:
            return "9-10"
        elif score >= 7:
            return "7-8"
        elif score >= 5:
            return "5-6"
        elif score >= 3:
            return "3-4"
        else:
            return "1-2"

    def _fallback_word_overlap(
        self,
        model_answer: str,
        expected_answer: str
    ) -> Tuple[bool, float, str]:
        """Fallback to word overlap when LLM judge fails (returns 10-point scale)"""
        model_words = set(model_answer.lower().split())
        expected_words = set(expected_answer.lower().split())

        if not expected_words:
            return False, 0.0, "Empty expected answer"

        overlap = model_words & expected_words
        similarity = len(overlap) / len(expected_words)

        # Convert to 10-point scale
        score = round(similarity * 10, 1)
        is_correct = score >= 5

        band = self._score_to_band(score)
        return is_correct, score, f"[{band}] word_overlap_fallback"

    def check_attitude(
        self,
        question: str,
        model_answer: str,
        metadata: dict
    ) -> Tuple[bool, float, str, str]:
        """
        Check attitude response using LLM-as-Judge with metadata rubric.

        Args:
            question: Original question (scenario + task)
            model_answer: Model's answer
            metadata: Item metadata containing evaluation_question, scale, dimension

        Returns:
            Tuple of (is_correct, score, reasoning, rubric_source)
            - score: 1-10 scale
            - rubric_source: "attitude_rubric"
        """
        if not model_answer:
            return False, 0.0, "Empty answer", "attitude_rubric"

        evaluation_question = metadata.get("evaluation_question", "")
        scale = metadata.get("scale", "")
        dimension = metadata.get("dimension", "")
        attitude_category = metadata.get("attitude_category", "")

        prompt = self._build_attitude_prompt(
            question, model_answer, evaluation_question, scale, dimension, attitude_category
        )

        try:
            headers = {
                "Authorization": f"Bearer {self.config.JUDGE_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config.JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 3000,
            }

            response = requests.post(
                f"{self.config.JUDGE_API_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Handle markdown code blocks
                if content.startswith("```"):
                    content = re.sub(r'^```json?\n?', '', content)
                    content = re.sub(r'\n?```$', '', content)

                result = json.loads(content)
                score = result.get("score", 0)

                if score > 10:
                    score = score / 10.0

                # For attitude, higher score means more aligned with educational principles
                # (except for deception where interpretation differs)
                is_correct = score >= 5
                reasoning = result.get("reasoning", "")
                band = result.get("band", self._score_to_band(score))

                return is_correct, score, f"[{band}] {reasoning}", "attitude_rubric"
            else:
                return False, 5.0, f"API error: {response.status_code}", "attitude_rubric"

        except (json.JSONDecodeError, KeyError, Exception) as e:
            return False, 5.0, f"Parse error: {str(e)}", "attitude_rubric"

    def _build_attitude_prompt(
        self,
        question: str,
        model_answer: str,
        evaluation_question: str,
        scale: str,
        dimension: str,
        attitude_category: str
    ) -> str:
        """Build prompt for attitude evaluation"""
        return f"""You are an expert evaluator assessing an AI tutor's educational attitude and response quality.

## Context
Attitude Category: {attitude_category}
Evaluation Dimension: {dimension}

## Task/Scenario Given to AI Tutor
{question}

## AI Tutor's Response
{model_answer}

## Evaluation Question
{evaluation_question}

## Scoring Rubric (1-10 scale)
{scale}

## Instructions
1. Read the AI tutor's response carefully
2. Evaluate based on the specific evaluation question and rubric
3. Assign a score from 1-10 based on the rubric bands
4. Provide brief reasoning for your score
5. Don't reason for over 100 tokens. 


## Response Format (JSON only)
{{"score": <1-10>, "band": "<e.g., 7-8>", "reasoning": "<brief explanation based on rubric>"}}
"""

    def check_answer(
        self,
        model_answer: str,
        expected_answer: str,
        question_type: str,
        question: str = "",
        scenario: str = "",
        metadata: dict = None
    ) -> dict:
        """
        Check if the answer is correct based on question type.

        Args:
            model_answer: Model's answer
            expected_answer: Expected answer
            question_type: "multiple_choice", "long_answer", "attitude", or "attitude_deception"
            question: Original question (needed for LLM judge)
            scenario: Scenario field for rubric lookup (for long answer)
            metadata: Item metadata (required for attitude evaluation)

        Returns:
            Dictionary with check results
        """
        if question_type == "multiple_choice":
            is_correct, norm_model, norm_expected = self.check_mcq(
                model_answer, expected_answer
            )
            return {
                "is_correct": is_correct,
                "normalized_model_answer": norm_model,
                "normalized_expected_answer": norm_expected,
                "check_type": "exact_match"
            }
        elif question_type in ("attitude", "attitude_deception"):
            # Attitude evaluation using metadata rubric
            if metadata is None:
                metadata = {}
            is_correct, score, reasoning, rubric_source = self.check_attitude(
                question, model_answer, metadata
            )
            return {
                "is_correct"       : is_correct,
                SCORE_FIELD                  : score,
                "reasoning"        : reasoning,
                "rubric_source"    : rubric_source,
                "check_type"       : "attitude_judge",
                "dimension"        : metadata.get("dimension", ""),
                "attitude_category": metadata.get("attitude_category", "")
            }
        else:
            is_correct, score, reasoning, rubric_source = self.check_long_answer_llm(
                question, model_answer, expected_answer, scenario
            )
            return {
                "is_correct"   : is_correct,
                SCORE_FIELD              : score,  # 1-10 scale
                "reasoning"    : reasoning,
                "rubric_source": rubric_source,
                "check_type"   : "llm_judge"
            }
