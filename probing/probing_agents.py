"""Probing agents — multi-agent question generation pipeline.

Two probe-construction strategies:

  1. SingleShotBaseline: ONE LLM call producing a generic follow-up. Sees only
     seed question and initial answer; no uncertainty info, no critic, no
     challenger. Fair zero-shot probing baseline.

  2. GeneratorCriticChallenger: three sequential LLM calls — Generator -> Critic
     -> Challenger. Implements "Option 1" from the project working doc.

Architectural reference: this is a from-scratch re-implementation of the
principled-probing idea from ProbeLLM (github.com/HowieHwong/ProbeLLM).
We do not import that library; we follow the agentic decomposition pattern in
three explicit stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .llm_client import GeminiClient


@dataclass
class ProbeTrace:
    seed_question: str
    initial_answer: str
    sampled_answers: List[str]
    initial_uncertainty: float
    generator_q: str
    critic_q: str
    challenger_q: str
    final_probe: str
    method: str


class GeneratorCriticChallenger:
    """Three-stage probe builder. Each stage strengthens the probing question."""

    NAME = "agentic_gcc"

    def __init__(self, cfg: Config, llm: Optional[GeminiClient] = None):
        self.cfg = cfg
        self.llm = llm or GeminiClient(cfg)

    def _generator(self, seed_q: str, initial_answer: str, sampled: List[str]) -> str:
        system = (
            "You are the GENERATOR agent in a stress-testing pipeline for an LLM "
            "answering Phoenix, AZ crime questions. Given the original question, "
            "the initial answer, and several sampled answers showing the model's "
            "variability, propose ONE follow-up probing question that targets "
            "where the answers disagree or look weakest. Output only the question."
        )
        sample_block = "\n".join(f"- {a}" for a in sampled)
        user = (
            f"Original question: {seed_q}\n"
            f"Initial answer: {initial_answer}\n"
            f"Sampled answers:\n{sample_block}\n\n"
            "Write one probing question."
        )
        return self.llm.chat(system, user, temperature=0.5)

    def _critic(self, seed_q: str, initial_answer: str, generator_q: str) -> str:
        system = (
            "You are the CRITIC agent. Improve the GENERATOR's probing question "
            "for specificity, clarity, and diagnostic value (a good probe forces "
            "the target to commit to a falsifiable answer). Keep it grounded in "
            "the Phoenix crime domain. Output only the improved question."
        )
        user = (
            f"Original question: {seed_q}\n"
            f"Initial answer: {initial_answer}\n"
            f"Generator question: {generator_q}\n\n"
            "Rewrite as one improved probing question."
        )
        return self.llm.chat(system, user, temperature=0.5)

    def _challenger(self, seed_q: str, initial_answer: str, critic_q: str) -> str:
        system = (
            "You are the CHALLENGER agent. Make the question adversarial: add a "
            "contradiction, edge case, or assumption-flip that exposes hidden "
            "fragility in the initial answer (e.g. constrain timeframe, exclude a "
            "category, force a numerical estimate). Stay in the Phoenix crime "
            "domain. Output only the final adversarial question."
        )
        user = (
            f"Original question: {seed_q}\n"
            f"Initial answer: {initial_answer}\n"
            f"Critic-improved question: {critic_q}\n\n"
            "Write one adversarial probing question."
        )
        return self.llm.chat(system, user, temperature=0.5)

    def build_probe(
        self,
        seed_q: str,
        initial_answer: str,
        sampled: List[str],
        initial_uncertainty: float,
    ) -> ProbeTrace:
        gen = self._generator(seed_q, initial_answer, sampled)
        crit = self._critic(seed_q, initial_answer, gen)
        chal = self._challenger(seed_q, initial_answer, crit)
        return ProbeTrace(
            seed_question=seed_q,
            initial_answer=initial_answer,
            sampled_answers=sampled,
            initial_uncertainty=initial_uncertainty,
            generator_q=gen,
            critic_q=crit,
            challenger_q=chal,
            final_probe=chal,
            method=self.NAME,
        )


class SingleShotBaseline:
    """API-generated zero-shot probe — fair comparison vs the multi-stage pipeline."""

    NAME = "baseline_singleshot"

    def __init__(self, cfg: Config, llm: Optional[GeminiClient] = None):
        self.cfg = cfg
        self.llm = llm or GeminiClient(cfg)

    def build_probe(
        self,
        seed_q: str,
        initial_answer: str,
        sampled: List[str],
        initial_uncertainty: float,
    ) -> ProbeTrace:
        system = (
            "You are a follow-up question writer. Given a question and the "
            "initial answer, write ONE follow-up question for the same QA "
            "system. Keep it short. Output only the question."
        )
        user = (
            f"Original question: {seed_q}\n"
            f"Initial answer: {initial_answer}\n\n"
            "Write one follow-up question."
        )
        probe = self.llm.chat(system, user, temperature=0.7)
        return ProbeTrace(
            seed_question=seed_q,
            initial_answer=initial_answer,
            sampled_answers=sampled,
            initial_uncertainty=initial_uncertainty,
            generator_q=probe,
            critic_q="",
            challenger_q="",
            final_probe=probe,
            method=self.NAME,
        )
