"""Evaluation metrics from the project doc — fully LLM-judge based, no embeddings.

Per-probe signals collected:
  - uncertainty_delta = post_uncertainty - pre_uncertainty
  - answer_changed (semantic) — LLM judge: are pre and post the same claim?
  - contradiction_found — LLM judge: are pre and post contradictory?

Aggregate metrics:
  1. Mean uncertainty increase
  2. Contradiction / inconsistency rate
  3. Diagnostic question effectiveness (probe is "effective" if any signal triggers)
  4. Gaps found per question
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .llm_client import GeminiClient


@dataclass
class ProbeOutcome:
    seed_question: str
    probe_question: str
    pre_answer: str
    post_answer: str
    pre_uncertainty: float
    post_uncertainty: float
    answer_changed: bool
    contradiction: bool

    @property
    def uncertainty_delta(self) -> float:
        return self.post_uncertainty - self.pre_uncertainty

    @property
    def is_effective(self) -> bool:
        return (
            self.uncertainty_delta > 0.05
            or self.answer_changed
            or self.contradiction
        )


@dataclass
class AggregateMetrics:
    n_probes: int
    mean_uncertainty_increase: float
    contradiction_rate: float
    diagnostic_effectiveness: float
    gaps_per_question: float
    per_probe: List[ProbeOutcome] = field(default_factory=list)


def detect_answer_change(
    cfg: Config,
    pre: str,
    post: str,
    llm: Optional[GeminiClient] = None,
) -> bool:
    """LLM-judge: do these two answers commit to the same factual claim?"""
    if llm is None:
        llm = GeminiClient(cfg)
    system = (
        "You are a strict judge. Decide whether two answers from the same QA "
        "system commit to the SAME factual claim. They are the SAME claim if "
        "the core factual content matches, even if wording differs. They are "
        "DIFFERENT claims if they disagree on a fact, scope, sign, or commit "
        "to different specifics. Reply with exactly one word: SAME or DIFFERENT."
    )
    user = f"Answer A:\n{pre}\n\nAnswer B:\n{post}\n\nSAME or DIFFERENT?"
    verdict = llm.chat(system, user, temperature=0.0).strip().upper()
    return verdict.startswith("DIFF")


def judge_correctness(
    cfg: Config,
    question: str,
    candidate_answer: str,
    gold_answer: str,
    gold_kind: str,
    llm: Optional[GeminiClient] = None,
) -> str:
    """LLM-judge correctness. Returns one of:
       'correct' | 'partial' | 'incorrect' | 'hedged_appropriate' | 'overconfident_wrong'.

    For gold_kind == 'hedged' the gold answer is a description of appropriate
    hedging; we judge whether the candidate is appropriately uncertain rather
    than claiming a single fact.
    """
    if llm is None:
        llm = GeminiClient(cfg)

    if gold_kind == "hedged":
        system = (
            "You are a strict judge. The question is intentionally ambiguous; the "
            "GOLD describes what an appropriately hedged answer should acknowledge. "
            "Reply with one of these labels (exactly): "
            "HEDGED_APPROPRIATE — answer hedges or clarifies the ambiguity sensibly; "
            "OVERCONFIDENT_WRONG — answer commits to a single claim without hedging; "
            "PARTIAL — partly hedges but misses a key dimension."
        )
        user = (
            f"Question: {question}\n\n"
            f"Gold (description of appropriate hedging): {gold_answer}\n\n"
            f"Candidate answer: {candidate_answer}\n\n"
            "Reply with exactly one label."
        )
    else:
        system = (
            "You are a strict judge of factual QA answers. Reply with exactly one of "
            "these labels: CORRECT (matches the gold fact), PARTIAL (contains the gold "
            "fact mixed with extra wrong claims, or close but slightly off), "
            "INCORRECT (does not match the gold), HEDGED_APPROPRIATE (the answer "
            "honestly says it does not know — only when the answer makes no factual "
            "claim that contradicts the gold)."
        )
        user = (
            f"Question: {question}\n\n"
            f"Gold answer (truth): {gold_answer}\n"
            f"Gold kind: {gold_kind}\n\n"
            f"Candidate answer: {candidate_answer}\n\n"
            "Reply with exactly one label."
        )
    verdict = llm.chat(system, user, temperature=0.0).strip().upper()
    for label in ("CORRECT", "PARTIAL", "INCORRECT", "HEDGED_APPROPRIATE", "OVERCONFIDENT_WRONG"):
        if verdict.startswith(label):
            return label.lower()
    return "incorrect"


def detect_contradiction(
    cfg: Config,
    pre: str,
    post: str,
    llm: Optional[GeminiClient] = None,
) -> bool:
    """LLM-judge contradiction detection."""
    if llm is None:
        llm = GeminiClient(cfg)
    system = (
        "You are a strict judge. Decide whether two answers from the same QA "
        "system are CONTRADICTORY (one asserts something the other denies, or "
        "they give incompatible facts). Reply with exactly one word: YES or NO."
    )
    user = f"Answer A:\n{pre}\n\nAnswer B:\n{post}\n\nAre A and B contradictory?"
    verdict = llm.chat(system, user, temperature=0.0).strip().upper()
    return verdict.startswith("YES")


def aggregate(outcomes: List[ProbeOutcome]) -> AggregateMetrics:
    n = len(outcomes)
    if n == 0:
        return AggregateMetrics(0, 0.0, 0.0, 0.0, 0.0, [])

    mean_unc = sum(o.uncertainty_delta for o in outcomes) / n
    contradiction_rate = sum(1 for o in outcomes if o.contradiction) / n
    effective = sum(1 for o in outcomes if o.is_effective)
    diagnostic = effective / n

    return AggregateMetrics(
        n_probes=n,
        mean_uncertainty_increase=mean_unc,
        contradiction_rate=contradiction_rate,
        diagnostic_effectiveness=diagnostic,
        gaps_per_question=diagnostic,
        per_probe=outcomes,
    )
