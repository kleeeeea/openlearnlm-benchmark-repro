"""Configuration for OpenLearnLM Question Generator"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    # API Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    MODEL_NAME: str = "gpt-5-mini-2025-08-07"
    # Generation Settings
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # Batch Settings
    BATCH_SIZE: int = 100
    REQUEST_DELAY: float = 1.0  # seconds
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0  # seconds
    CHECKPOINT_INTERVAL: int = 50  # items per checkpoint (for resume)
    OUTPUT_BATCH_SIZE: int = 1000  # items per output file

    # File Paths (relative to project root)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    INPUT_CSV: Path = PROJECT_ROOT / "2. Literature Review" / "벤치마크 조합_문제 찍어내기.csv"
    OUTPUT_DIR: Path = PROJECT_ROOT / "3. Benchmark Prototype" / "generation_output"
    PROGRESS_FILE: str = "progress.json"

    # Output Settings
    OUTPUT_LANGUAGE: str = "en"

    def __post_init__(self):
        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Validate API key
        if not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set in environment variables")
