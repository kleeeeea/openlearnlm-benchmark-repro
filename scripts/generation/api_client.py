"""OpenAI API Client"""

import time
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import Config


@dataclass
class APIResponse:
    success: bool
    content: Optional[str]
    error: Optional[str]
    usage: Optional[Dict[str, int]]


class OpenAIClient:
    """OpenAI API Client (renamed for compatibility)"""

    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

    def send_request(
        self,
        messages: List[Dict[str, str]],
        retry_count: int = 0
    ) -> APIResponse:
        """Send request to OpenAI API with retry logic"""
        try:
            response = requests.post(
                f"{self.config.OPENAI_BASE_URL}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.config.MODEL_NAME,
                    "messages": messages,
                    "max_completion_tokens": self.config.MAX_TOKENS,
                    "reasoning_effort": "minimal",
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # Check for empty or None content (model may refuse to respond)
                if not content or not content.strip():
                    return APIResponse(
                        success=False,
                        content=None,
                        error="Empty response from model (possible content filtering)",
                        usage=data.get("usage")
                    )
                return APIResponse(
                    success=True,
                    content=content,
                    error=None,
                    usage=data.get("usage")
                )
            elif response.status_code == 429:  # Rate limit
                if retry_count < self.config.MAX_RETRIES:
                    wait_time = self.config.RETRY_DELAY * (retry_count + 1)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    return self.send_request(messages, retry_count + 1)
                return APIResponse(
                    success=False,
                    content=None,
                    error="Rate limit exceeded",
                    usage=None
                )
            else:
                return APIResponse(
                    success=False,
                    content=None,
                    error=f"HTTP {response.status_code}: {response.text}",
                    usage=None
                )

        except requests.exceptions.Timeout:
            if retry_count < self.config.MAX_RETRIES:
                print(f"Timeout. Retrying ({retry_count + 1}/{self.config.MAX_RETRIES})...")
                time.sleep(self.config.RETRY_DELAY)
                return self.send_request(messages, retry_count + 1)
            return APIResponse(
                success=False,
                content=None,
                error="Request timeout",
                usage=None
            )

        except Exception as e:
            if retry_count < self.config.MAX_RETRIES:
                print(f"Error: {e}. Retrying ({retry_count + 1}/{self.config.MAX_RETRIES})...")
                time.sleep(self.config.RETRY_DELAY)
                return self.send_request(messages, retry_count + 1)
            return APIResponse(
                success=False,
                content=None,
                error=str(e),
                usage=None
            )
