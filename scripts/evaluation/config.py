"""Configuration for OpenLearnLM Benchmark Evaluation"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _load_dotenv(env_path: Path):
    """Simple .env file loader (no external dependency)"""
    if not env_path.exists():
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue

            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value


# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
_load_dotenv(_project_root / ".env")


@dataclass
class EvalConfig:
    """Evaluation configuration"""

    # API Keys
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENROUTER_API_KEY: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))

    # API URLs (overridable via env vars)
    OPENAI_BASE_URL: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    OPENROUTER_BASE_URL: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))

    # LLM-as-Judge API (overridable via env vars)
    JUDGE_API_URL: str = field(default_factory=lambda: os.getenv("JUDGE_API_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")))
    JUDGE_API_KEY: str = field(default_factory=lambda: os.getenv("JUDGE_API_KEY", os.getenv("OPENROUTER_API_KEY", "")))
    JUDGE_MODEL: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini"))

    # Models to evaluate
    MODELS: List[str] = field(default_factory=lambda: [
        "gpt-5.2",                        # OpenAI Direct
        "google/gemini-3-pro-preview",    # OpenRouter (Gemini 3 Pro Preview)
        "anthropic/claude-opus-4.5",      # OpenRouter
        "deepseek/deepseek-v3.2",         # OpenRouter
        "x-ai/grok-4.1-fast",             # OpenRouter (xAI Grok 4.1 Fast)
        # "qwen/qwen3-max",               # OpenRouter - DISABLED: free tier exhausted
        "z-ai/glm-4.7",                   # OpenRouter (Z.AI GLM 4.7)
        "moonshotai/kimi-k2-thinking",    # OpenRouter (Moonshot Kimi K2 Thinking)
    ])

    # Thinking Budget (unified at 2048 tokens)
    THINKING_BUDGET_TOKENS: int = 2048

    # Generation Settings (for reproducibility)
    TEMPERATURE: float = 0.0
    TOP_P: float = 1.0
    MAX_ANSWER_TOKENS: int = 4096  # For model's answer (reasoning models need extra for thinking)

    # Parallelism
    WORKERS_PER_MODEL: int = 10
    REQUEST_DELAY: float = 0.2  # seconds
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0  # seconds
    REQUEST_TIMEOUT: int = 120  # seconds

    # Progress tracking
    CHECKPOINT_INTERVAL: int = 100  # items per checkpoint

    # File Paths
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # Benchmark category (for folder structure)
    BENCHMARK_CATEGORY: str = "01_기능_skills"

    # Map CLI category names to actual data folder names
    _CATEGORY_FOLDER: dict = field(default_factory=lambda: {
        "01_기능_skills": "skills",
        "02_교과지식_content": "content_knowledge",
        "03_교육학지식_pedagogical": "pedagogical_knowledge",
        "04_태도_attitude": "attitude",
    })

    @property
    def CATEGORY_FOLDER(self) -> str:
        return self._CATEGORY_FOLDER.get(self.BENCHMARK_CATEGORY, self.BENCHMARK_CATEGORY)

    @property
    def BENCHMARK_DATA_DIR(self) -> Path:
        """Benchmark data directory"""
        return self.PROJECT_ROOT / "data"

    @property
    def TEST_DATA_FILE(self) -> Path:
        """Path to the test dataset"""
        return self.BENCHMARK_DATA_DIR / self.CATEGORY_FOLDER / "questions_test.jsonl"

    @property
    def OUTPUTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "outputs"

    @property
    def LOGS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "logs"

    @property
    def RESPONSES_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "responses" / self.BENCHMARK_CATEGORY

    @property
    def REPORTS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "reports"

    # OpenRouter specific
    SITE_URL: str = "XXXX-1"
    SITE_NAME: str = "OpenLearnLM Benchmark"

    def __post_init__(self):
        # Ensure output directories exist
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Validate API keys
        if not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set in environment variables")
        if not self.OPENROUTER_API_KEY:
            print("Warning: OPENROUTER_API_KEY not set in environment variables")
        if not self.JUDGE_API_KEY:
            print("Warning: JUDGE_API_KEY not set in environment variables")

    def get_model_client_type(self, model_id: str) -> str:
        """Determine which API client to use for a model"""
        if model_id.startswith("gpt"):
            return "openai"
        else:
            return "openrouter"

    def get_thinking_params(self, model_id: str) -> dict:
        """Get model-specific thinking budget parameters"""
        if model_id.startswith("gpt"):
            # GPT-5.2 uses reasoning.effort
            return {"reasoning": {"effort": "medium"}}
        elif "deepseek" in model_id:
            # DeepSeek uses reasoning.effort
            return {"reasoning": {"effort": "medium"}}
        else:
            # Gemini, Claude use reasoning.max_tokens
            return {"reasoning": {"max_tokens": self.THINKING_BUDGET_TOKENS}}


# Default configuration instance
config = EvalConfig()
