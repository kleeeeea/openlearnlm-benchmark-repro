"""OpenRouter API Client for Gemini, Claude, and DeepSeek"""

import time
import requests
from typing import Dict, Any

from .base_client import BaseModelClient, EvaluationRequest, EvaluationResponse
from ..config import EvalConfig


class OpenRouterClient(BaseModelClient):
    """OpenRouter API Client for multiple model providers"""

    # Model-specific thinking budget configurations
    # Note: The 'reasoning' parameter was removed as it's not supported
    # by OpenRouter for most models. Models use their built-in reasoning
    # capabilities without explicit configuration.
    THINKING_CONFIG = {
        "google/gemini-3-pro-preview": {},
        "anthropic/claude-opus-4.5": {},
        "deepseek/deepseek-v3.2": {},
        "x-ai/grok-4.1-fast": {},
        "qwen/qwen3-max": {},
        "z-ai/glm-4.7": {},
        "moonshotai/kimi-k2-thinking": {},
    }

    def __init__(self, config: EvalConfig, model_id: str):
        super().__init__(config, model_id)
        self.headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": config.SITE_URL,
            "X-Title": config.SITE_NAME,
            "Content-Type": "application/json",
        }

    def get_thinking_params(self) -> Dict[str, Any]:
        """Get model-specific thinking parameters"""
        return self.THINKING_CONFIG.get(self.model_id, {})

    def evaluate(self, request: EvaluationRequest, retry_count: int = 0) -> EvaluationResponse:
        """
        Evaluate a question using OpenRouter.

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
                "max_tokens": self.config.MAX_ANSWER_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                **self.get_thinking_params()
            }

            response = requests.post(
                f"{self.config.OPENROUTER_BASE_URL}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.REQUEST_TIMEOUT
            )

            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()

                # Check for error in response body (e.g., Overloaded from provider)
                if "error" in data:
                    import json
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    error_code = data.get("error", {}).get("code", 0)

                    # Retry on transient errors (500, 502 Overloaded, etc.)
                    if error_code in [500, 502, 503, 529] or "overload" in error_msg.lower():
                        if retry_count < self.config.MAX_RETRIES:
                            wait_time = self.config.RETRY_DELAY * (retry_count + 1)
                            time.sleep(wait_time)
                            return self.evaluate(request, retry_count + 1)

                    error_detail = f"Provider error: {error_msg} (code: {error_code})"
                    return EvaluationResponse(
                        success=False,
                        model_answer=None,
                        raw_content=str(data),
                        thinking_content=None,
                        error=error_detail,
                        usage=data.get("usage"),
                        latency_ms=latency_ms
                    )

                # Handle OpenRouter response format
                if "choices" not in data or len(data["choices"]) == 0:
                    # Log full response for debugging
                    import json
                    error_detail = f"No choices in response. Full response: {json.dumps(data, ensure_ascii=False)[:500]}"
                    return EvaluationResponse(
                        success=False,
                        model_answer=None,
                        raw_content=str(data),
                        thinking_content=None,
                        error=error_detail,
                        usage=data.get("usage"),
                        latency_ms=latency_ms
                    )

                message = data["choices"][0]["message"]
                content = message.get("content")

                # For reasoning models, content might be in 'reasoning' field
                thinking_content_extracted = None
                if "reasoning" in message and message["reasoning"]:
                    thinking_content_extracted = message["reasoning"]
                    # If content is empty but reasoning exists, use reasoning as content
                    if not content or not content.strip():
                        content = thinking_content_extracted

                if not content or not content.strip():
                    # Include raw message data for debugging
                    error_detail = f"Empty response from model. Message keys: {list(message.keys())}"
                    if "refusal" in message and message["refusal"]:
                        error_detail = f"Model refused: {message['refusal']}"
                    return EvaluationResponse(
                        success=False,
                        model_answer=None,
                        raw_content=str(message),
                        thinking_content=thinking_content_extracted,
                        error=error_detail,
                        usage=data.get("usage"),
                        latency_ms=latency_ms
                    )

                # Extract thinking content if available (varies by model)
                thinking_content = thinking_content_extracted  # Use already extracted
                if not thinking_content:
                    if "reasoning_content" in message:
                        thinking_content = message["reasoning_content"]
                    elif "thinking" in message:
                        thinking_content = message["thinking"]

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
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after else self.config.RETRY_DELAY * (retry_count + 1)
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

            elif response.status_code == 502 or response.status_code == 503:
                # Service temporarily unavailable
                if retry_count < self.config.MAX_RETRIES:
                    wait_time = self.config.RETRY_DELAY * (retry_count + 1)
                    time.sleep(wait_time)
                    return self.evaluate(request, retry_count + 1)

                return EvaluationResponse(
                    success=False,
                    model_answer=None,
                    raw_content=None,
                    thinking_content=None,
                    error=f"Service unavailable: HTTP {response.status_code}",
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
