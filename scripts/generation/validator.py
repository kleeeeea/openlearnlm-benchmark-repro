"""Response Validator for API outputs"""

import json
import re
from typing import Tuple, Optional, Dict, Any


class ResponseValidator:
    """Validate and parse API responses"""

    REQUIRED_FIELDS = ["question", "answer"]
    VALID_MCQ_ANSWERS = {"A", "B", "C", "D"}

    # Phrases that should NOT appear in MCQ questions
    MCQ_FORBIDDEN_PHRASES = [
        "and why",
        "explain why",
        "justify your choice",
        "describe why",
        "tell why",
        "explain your",
        "justify your",
    ]

    def clean_json_response(self, raw_response: str) -> str:
        """Remove markdown code blocks and clean response"""
        # Remove ```json ... ``` pattern
        cleaned = re.sub(r'```json\s*', '', raw_response)
        cleaned = re.sub(r'```\s*', '', cleaned)

        # Try to extract JSON object if there's extra text
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        return cleaned.strip()

    def normalize_mcq_answer(self, answer: str) -> Tuple[bool, str]:
        """
        Normalize MCQ answer to single letter format.
        Returns: (success, normalized_answer)
        """
        answer = str(answer).strip()

        # Already valid single letter
        if answer.upper() in self.VALID_MCQ_ANSWERS:
            return (True, answer.upper())

        # Try to extract letter from patterns like "B) Blue square" or "B."
        match = re.match(r'^([A-Da-d])\s*[\)\.\:\-]', answer)
        if match:
            return (True, match.group(1).upper())

        # Try to find single letter at the start
        if len(answer) >= 1 and answer[0].upper() in self.VALID_MCQ_ANSWERS:
            # Only accept if it's clearly a letter answer (not a word starting with A-D)
            if len(answer) == 1 or not answer[1].isalpha():
                return (True, answer[0].upper())

        return (False, answer)

    def validate_mcq_answer(self, answer: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate and normalize MCQ answer format.
        Returns: (is_valid, normalized_answer, error_message)
        """
        success, normalized = self.normalize_mcq_answer(answer)

        if success:
            return (True, normalized, None)

        return (
            False,
            None,
            f"Invalid MCQ answer format: '{answer}'. Expected single letter (A, B, C, or D)"
        )

    def validate_mcq_question_format(self, question: str) -> Tuple[bool, Optional[str]]:
        """
        Validate MCQ question does not contain forbidden phrases.
        Returns: (is_valid, error_message)
        """
        question_lower = question.lower()
        for phrase in self.MCQ_FORBIDDEN_PHRASES:
            if phrase in question_lower:
                return (False, f"MCQ contains forbidden phrase: '{phrase}'")
        return (True, None)

    def validate_response(
        self,
        raw_response: str,
        question_type: str = "multiple choice"
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate and parse response.
        Returns: (is_valid, parsed_data, error_message)
        """
        try:
            cleaned = self.clean_json_response(raw_response)
            data = json.loads(cleaned)

            # Check required fields
            for field in self.REQUIRED_FIELDS:
                if field not in data:
                    return (False, None, f"Missing required field: {field}")
                if not data[field] or not str(data[field]).strip():
                    return (False, None, f"Empty required field: {field}")

            # Validate MCQ format
            if question_type.lower().strip() == "multiple choice":
                # Validate answer format
                is_valid, normalized, error = self.validate_mcq_answer(data["answer"])
                if not is_valid:
                    return (False, None, error)
                # Update answer to normalized format
                data["answer"] = normalized

                # Validate question format (no forbidden phrases)
                is_valid, error = self.validate_mcq_question_format(data["question"])
                if not is_valid:
                    return (False, None, error)

            return (True, data, None)

        except json.JSONDecodeError as e:
            return (False, None, f"JSON parsing error: {str(e)}")
        except Exception as e:
            return (False, None, f"Validation error: {str(e)}")
