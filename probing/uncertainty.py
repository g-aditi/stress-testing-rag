"""Uncertainty estimation via semantic entropy — LLM-judge clustering variant.

Re-implements the SemanticEntropy estimator from LM-Polygraph (Fadeeva et al.,
EMNLP 2023; github.com/IINemo/lm-polygraph) — specifically the black-box
variant of Kuhn et al. (ICLR 2023, "Semantic Uncertainty"). The procedure:

  1. Sample N answers from the QA system at non-zero temperature.
  2. Group samples by *semantic equivalence* (paraphrases share a cluster).
  3. Compute Shannon entropy over the cluster-size distribution.

Difference from the upstream LM-Polygraph implementation:

  LM-Polygraph uses an NLI model (DeBERTa-MNLI by default) for bidirectional
  entailment. We replace that with a single Gemini call that returns a JSON
  cluster assignment over the N samples. This is closer in spirit to the NLI
  approach than embedding cosine — we ask "are these the same claim?" rather
  than "are these vectors close?" — and it avoids running embeddings or any
  HF model. Conceptually identical metric; one model call per uncertainty
  estimate.

Higher entropy = more disagreement across samples = more uncertainty. We also
return a normalised entropy in [0, 1] (divided by log2(N)) so values are
comparable across different sample sizes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .llm_client import GeminiClient


@dataclass
class UncertaintyReport:
    answers: List[str]
    cluster_assignments: List[int]
    n_clusters: int
    semantic_entropy: float
    normalized_entropy: float


def _entropy(assignments: List[int]) -> float:
    n = len(assignments)
    if n == 0:
        return 0.0
    counts: dict[int, int] = {}
    for a in assignments:
        counts[a] = counts.get(a, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def _judge_cluster(llm: GeminiClient, answers: List[str]) -> List[int]:
    """One LLM call -> cluster assignment over N answers."""
    numbered = "\n".join(f"[{i+1}] {a}" for i, a in enumerate(answers))
    system = (
        "You are a semantic clustering judge. You will be given N answers from "
        "the same QA system. Group them into equivalence classes by meaning: "
        "two answers belong to the same cluster if they make the same factual "
        "claim, even if worded differently. Two answers belong to different "
        "clusters if they disagree on a factual point or commit to different "
        "claims. Output JSON of the form "
        '{"clusters": [c_1, c_2, ..., c_N]} where c_i is the 0-indexed cluster '
        "ID for answer i. Use 0, 1, 2, ... in order of first appearance."
    )
    user = (
        f"Cluster these {len(answers)} answers. "
        "Respond ONLY with the JSON object.\n\n" + numbered
    )
    data = llm.chat_json(system, user, temperature=0.0)
    raw = data.get("clusters", [])
    if not isinstance(raw, list) or len(raw) != len(answers):
        return list(range(len(answers)))
    return [int(c) for c in raw]


def estimate_uncertainty(
    cfg: Config,
    answers: List[str],
    llm: Optional[GeminiClient] = None,
) -> UncertaintyReport:
    """Compute semantic entropy from a set of sampled answers."""
    if len(answers) < 2:
        return UncertaintyReport(
            answers=answers,
            cluster_assignments=[0] * len(answers),
            n_clusters=1 if answers else 0,
            semantic_entropy=0.0,
            normalized_entropy=0.0,
        )

    if llm is None:
        llm = GeminiClient(cfg)
    assignments = _judge_cluster(llm, answers)
    h = _entropy(assignments)
    max_h = math.log2(len(answers)) if len(answers) > 1 else 1.0
    normalized = h / max_h if max_h > 0 else 0.0

    return UncertaintyReport(
        answers=answers,
        cluster_assignments=assignments,
        n_clusters=len(set(assignments)),
        semantic_entropy=h,
        normalized_entropy=normalized,
    )
