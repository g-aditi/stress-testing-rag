"""Target QA system. Three answer modes:

  closed_book(q)                  -> no retrieval, no aggregate stats
  rag(q)                          -> retrieve from dataset, build context, answer
  rag_with_probe_context(seed, probe)
                                  -> retrieve using probe terms, answer SEED

This separation lets the orchestrator implement the research design:
  Phase A = closed_book(seed)      (what does the model know on its own?)
  Phase B = rag_with_probe_context (after probing surfaces the gap)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .datasets import CrimeDataset
from .llm_client import GeminiClient


@dataclass
class QAResult:
    answer: str
    context: str
    retrieved_count: int


class CrimeQAAgent:
    def __init__(self, cfg: Config, dataset: CrimeDataset, llm: Optional[GeminiClient] = None):
        self.cfg = cfg
        self.dataset = dataset
        self.llm = llm or GeminiClient(cfg)

    # --- closed-book ------------------------------------------------------

    def closed_book(self, question: str, temperature: Optional[float] = None) -> QAResult:
        """No retrieval, no dataset access — pure LLM knowledge."""
        system = (
            "You are a crime-statistics QA assistant. Answer the user's question "
            "using only your own knowledge. Do not invent dataset numbers; if you "
            "do not know, say so plainly. Keep answers concise (1-3 sentences)."
        )
        text = self.llm.chat(
            system=system,
            user=question,
            temperature=temperature if temperature is not None else self.cfg.answer_temperature,
        )
        return QAResult(answer=text, context="", retrieved_count=0)

    def sample_closed_book(self, question: str, n: int) -> List[str]:
        return [
            self.closed_book(question, temperature=self.cfg.sample_temperature).answer
            for _ in range(n)
        ]

    # --- RAG --------------------------------------------------------------

    def _build_context(self, retrieval_q: str) -> str:
        rows = self.dataset.retrieve(retrieval_q, self.cfg.rag_top_k)
        ctx = self.dataset.format_context(rows)
        agg = self.dataset.aggregate_stats(retrieval_q)
        if agg:
            ctx = agg + "\n\n" + ctx
        return ctx

    def rag(self, question: str, temperature: Optional[float] = None) -> QAResult:
        ctx = self._build_context(question)
        system = (
            f"You are a crime-statistics QA assistant grounded in the {self.dataset.name} "
            f"dataset ({self.dataset.description}). Answer using only the provided context. "
            "If the context does not support an answer, say what is missing rather than "
            "inventing facts. Keep answers concise (1-3 sentences)."
        )
        user = f"Question: {question}\n\nDataset context:\n{ctx}\n\nAnswer using only the context above."
        text = self.llm.chat(
            system=system,
            user=user,
            temperature=temperature if temperature is not None else self.cfg.answer_temperature,
        )
        return QAResult(answer=text, context=ctx, retrieved_count=self.cfg.rag_top_k)

    def rag_with_probe_context(
        self,
        seed: str,
        probe: str,
        temperature: Optional[float] = None,
    ) -> QAResult:
        """Retrieve using BOTH seed + probe terms, then answer the SEED.

        The probe is used ONLY to widen / sharpen retrieval. It is **not**
        shown to the answering LLM, because the Challenger agent's probes are
        adversarial-by-design ("addresses are truncated, cannot be uniquely
        identified, etc.") and the LLM otherwise inherits that framing and
        refuses to commit even when the correct record is right there in the
        retrieved context.
        """
        joined = f"{seed} {probe}"
        ctx = self._build_context(joined)
        system = (
            f"You are a crime-statistics QA assistant grounded in the {self.dataset.name} "
            f"dataset ({self.dataset.description}). Answer the user's question using the "
            "provided dataset context. If a record matches the asked address/date/category, "
            "report what it says — do NOT add caveats about anonymisation or aggregation. "
            "If the context truly does not contain a matching record, say so plainly. "
            "Keep answers concise (1-3 sentences)."
        )
        user = (
            f"Question: {seed}\n\n"
            f"Dataset context:\n{ctx}\n\n"
            "Answer the question using only the context above."
        )
        text = self.llm.chat(
            system=system,
            user=user,
            temperature=temperature if temperature is not None else self.cfg.answer_temperature,
        )
        return QAResult(answer=text, context=ctx, retrieved_count=self.cfg.rag_top_k)

    def sample_rag_with_probe_context(self, seed: str, probe: str, n: int) -> List[str]:
        return [
            self.rag_with_probe_context(seed, probe, temperature=self.cfg.sample_temperature).answer
            for _ in range(n)
        ]
