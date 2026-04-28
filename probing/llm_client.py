"""Thin Gemini wrapper used everywhere a chat call is made.

Centralising it lets every module call `client.chat(system, user, temperature)`
without knowing which provider is behind it. If the project ever switches back
to OpenAI or supports a second provider, only this file changes.
"""
from __future__ import annotations

import json
import re
import time
from typing import List, Optional

from google import genai
from google.genai import types

from .config import Config


class GeminiClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.api_key)
        self.model = cfg.chat_model

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.5,
        max_retries: int = 4,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                    ),
                )
                text = (resp.text or "").strip()
                if text:
                    return text
                last_err = RuntimeError("empty response")
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "quota" in msg or "rate" in msg or "429" in msg or "503" in msg:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"chat failed after {max_retries} attempts: {last_err}")

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> dict:
        """Force JSON-shaped output for judge / clustering calls."""
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise RuntimeError(f"could not parse JSON from model: {text!r}")
