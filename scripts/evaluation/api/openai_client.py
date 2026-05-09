"""OpenAI Direct API Client for GPT-5.2"""

import time
import requests
from typing import Dict, Any

from .base_client import BaseModelClient, EvaluationRequest, EvaluationResponse
from ..config import EvalConfig


class OpenAIDirectClient(BaseModelClient):
    """OpenAI Direct API Client for GPT-5.2 evaluation"""

    def __init__(self, config: EvalConfig, model_id: str = "gpt-5.2"):
        super().__init__(config, model_id)
        self.headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

    def get_thinking_params(self) -> Dict[str, Any]:
        """Get GPT-5.2 specific thinking parameters

        Note: The 'reasoning' parameter was removed as it's not supported
        by the standard OpenAI chat completions API.
        """
        return {}

    def evaluate(self, request: EvaluationRequest, retry_count: int = 0) -> EvaluationResponse:
        """
        Evaluate a question using GPT-5.2.

        Args:
            request: The evaluation request
            retry_count: Current retry attempt

        Returns:
            EvaluationResponse with model's answer
        """
        start_time = time.time()

        try:
            messages = self.build_messages(request)

            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_completion_tokens": self.config.MAX_ANSWER_TOKENS,
                "temperature": self.config.TEMPERATURE,
                **self.get_thinking_params()
            }

            response = requests.post(
                f"{self.config.OPENAI_BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.REQUEST_TIMEOUT
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content")

                # For reasoning models, content might be in 'reasoning' or 'reasoning_content'
                thinking_content = None
                if "reasoning_content" in message and message["reasoning_content"]:
                    thinking_content = message["reasoning_content"]
                    if not content or not content.strip():
                        content = thinking_content
                elif "reasoning" in message and message["reasoning"]:
                    thinking_content = message["reasoning"]
                    if not content or not content.strip():
                        content = thinking_content

                if not content or not content.strip():
                    # Include raw message data for debugging
                    error_detail = f"Empty response from model. Message keys: {list(message.keys())}"
                    if "refusal" in message and message["refusal"]:
                        error_detail = f"Model refused: {message['refusal']}"
                    return EvaluationResponse(
                        success=False,
                        model_answer=None,
                        raw_content=str(message),
                        thinking_content=thinking_content,
                        error=error_detail,
                        usage=data.get("usage"),
                        latency_ms=latency_ms
                    )

                # Extract answer based on question type
                model_answer = self.extract_answer(content, request.question_type)

                return EvaluationResponse(
                    success=True,
                    model_answer=model_answer,
                    raw_content=content,
                    thinking_content=thinking_content,
                    error=None,
                    usage=data.get("usage"),
                    latency_ms=latency_ms
                )

            elif response.status_code == 429:  # Rate limit
                if retry_count < self.config.MAX_RETRIES:
                    wait_time = self.config.RETRY_DELAY * (retry_count + 1)
                    time.sleep(wait_time)
                    return self.evaluate(request, retry_count + 1)

                return EvaluationResponse(
                    success=False,
                    model_answer=None,
                    raw_content=None,
                    thinking_content=None,
                    error="Rate limit exceeded",
                    usage=None,
                    latency_ms=latency_ms
                )

            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"

                if retry_count < self.config.MAX_RETRIES:
                    time.sleep(self.config.RETRY_DELAY)
                    return self.evaluate(request, retry_count + 1)

                return EvaluationResponse(
                    success=False,
                    model_answer=None,
                    raw_content=None,
                    thinking_content=None,
                    error=error_msg,
                    usage=None,
                    latency_ms=latency_ms
                )

        except requests.exceptions.Timeout:
            latency_ms = int((time.time() - start_time) * 1000)

            if retry_count < self.config.MAX_RETRIES:
                time.sleep(self.config.RETRY_DELAY)
                return self.evaluate(request, retry_count + 1)

            return EvaluationResponse(
                success=False,
                model_answer=None,
                raw_content=None,
                thinking_content=None,
                error="Request timeout",
                usage=None,
                latency_ms=latency_ms
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            if retry_count < self.config.MAX_RETRIES:
                time.sleep(self.config.RETRY_DELAY)
                return self.evaluate(request, retry_count + 1)

            return EvaluationResponse(
                success=False,
                model_answer=None,
                raw_content=None,
                thinking_content=None,
                error=str(e),
                usage=None,
                latency_ms=latency_ms
            )
