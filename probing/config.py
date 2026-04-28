"""Central config — loads .env, exposes Gemini settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env")


@dataclass
class Config:
    api_key: str
    chat_model: str = "gemini-3.1-flash-lite-preview"

    n_samples_for_entropy: int = 5
    sample_temperature: float = 0.7
    answer_temperature: float = 0.2

    rag_top_k: int = 3
    seed_questions_limit: int = 12

    outputs_dir: Path = REPO_ROOT / "outputs" / "probing"


def load_config() -> Config:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY missing. Put it in .env as GEMINI_API_KEY=..."
        )
    cfg = Config(
        api_key=api_key,
        chat_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    )
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
