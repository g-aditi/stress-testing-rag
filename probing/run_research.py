"""Research-grade orchestrator with gated probing.

Pipeline per seed:

  PHASE A — CLOSED-BOOK QA (no RAG, no probing)
    closed_answer  = QA.closed_book(seed, T=0.2)
    closed_samples = QA.closed_book(seed, T=0.7) x N
    closed_uncertainty = semantic_entropy(closed_samples)
    closed_correctness = judge(closed_answer vs gold)

  PHASE B — POLICY GATE
    if closed_correctness in {correct, hedged_appropriate}:
        skip probing (system passed)
    else:
        proceed to Phase C

  PHASE C — MULTI-AGENT PROBING + RECOVERY (only if A failed)
    probe_trace   = GCC.build_probe(seed, closed_answer, closed_samples, ...)
    probe         = probe_trace.final_probe
    rag_answer    = QA.rag_with_probe_context(seed, probe, T=0.2)
    rag_samples   = QA.rag_with_probe_context(seed, probe, T=0.7) x N
    rag_uncertainty = semantic_entropy(rag_samples)
    rag_correctness = judge(rag_answer vs gold)

  PHASE D — SCORING per trial
    recovered = (closed_correctness in {incorrect, partial, overconfident_wrong})
                AND (rag_correctness in {correct, hedged_appropriate})
    Δuncertainty = closed_uncertainty - rag_uncertainty (for probed trials)

Aggregate metrics (per dataset, per category, overall):
  - closed_book accuracy
  - probing-augmented accuracy (counts hedged_appropriate as correct for ambiguous)
  - recovery rate (rescued / probed)
  - mean Δuncertainty on probed trials
  - effective probes (any signal triggered, vs Δuncertainty>0.05 OR factual recovery)

All four output formats are checkpointed after each seed.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, load_config
from .datasets import CrimeDataset, PhoenixCrime, USHomicide
from .evaluation import detect_contradiction, judge_correctness
from .llm_client import GeminiClient
from .probing_agents import GeneratorCriticChallenger, ProbeTrace
from .qa_agent import CrimeQAAgent
from .seed_generator import Seed, generate_phoenix_seeds, generate_ushom_seeds
from .uncertainty import UncertaintyReport, estimate_uncertainty

import random


CORRECTNESS_PASS_AMBIGUOUS = {"correct", "hedged_appropriate"}
CORRECTNESS_PASS_FACTUAL = {"correct"}
CORRECTNESS_FAIL_FACTUAL = {"incorrect", "partial", "overconfident_wrong", "hedged_appropriate"}
CORRECTNESS_FAIL_AMBIGUOUS = {"incorrect", "partial", "overconfident_wrong"}


def is_pass(correctness: str, gold_kind: str) -> bool:
    """Gold-kind-aware pass rule.

    For ambiguous (hedged-gold) questions, an honestly hedged answer IS a
    pass — there is no single correct fact. For factual / statistical /
    comparative questions, only a verified-correct answer is a pass; hedging
    ("I do not have access to that data") is a *recoverable gap* that the
    probing+RAG path is designed to fix, so it must trigger Phase C.
    """
    if gold_kind == "hedged":
        return correctness in CORRECTNESS_PASS_AMBIGUOUS
    return correctness in CORRECTNESS_PASS_FACTUAL


def is_fail(correctness: str, gold_kind: str) -> bool:
    return not is_pass(correctness, gold_kind)


# Legacy aliases retained so older external imports do not break.
CORRECTNESS_PASS = CORRECTNESS_PASS_AMBIGUOUS
CORRECTNESS_FAIL = CORRECTNESS_FAIL_AMBIGUOUS


def _print(msg: str) -> None:
    print(f"[research] {msg}", flush=True)


@dataclass
class TrialRecord:
    seed_text: str
    category: str
    dataset: str
    gold_answer: str
    gold_kind: str

    closed_answer: str = ""
    closed_samples: List[str] = field(default_factory=list)
    closed_cluster_assignments: List[int] = field(default_factory=list)
    closed_n_clusters: int = 0
    closed_semantic_entropy: float = 0.0
    closed_normalized_entropy: float = 0.0
    closed_correctness: str = ""

    probed: bool = False

    probe_method: str = ""
    probe_generator_q: str = ""
    probe_critic_q: str = ""
    probe_challenger_q: str = ""
    probe_final: str = ""

    rag_answer: str = ""
    rag_samples: List[str] = field(default_factory=list)
    rag_cluster_assignments: List[int] = field(default_factory=list)
    rag_n_clusters: int = 0
    rag_semantic_entropy: float = 0.0
    rag_normalized_entropy: float = 0.0
    rag_correctness: str = ""
    rag_context: str = ""

    contradiction_pre_vs_post: bool = False
    delta_uncertainty: float = 0.0
    recovered: bool = False


def _build_trial_record_from_seed(seed: Seed) -> TrialRecord:
    return TrialRecord(
        seed_text=seed.text,
        category=seed.category,
        dataset=seed.dataset,
        gold_answer=seed.gold_answer,
        gold_kind=seed.gold_kind,
    )


def run_phase_a(
    cfg: Config,
    qa: CrimeQAAgent,
    llm: GeminiClient,
    seed: Seed,
    record: TrialRecord,
) -> None:
    closed = qa.closed_book(seed.text)
    samples = qa.sample_closed_book(seed.text, cfg.n_samples_for_entropy)
    unc = estimate_uncertainty(cfg, samples, llm=llm)
    correctness = judge_correctness(
        cfg, seed.text, closed.answer, seed.gold_answer, seed.gold_kind, llm=llm
    )

    record.closed_answer = closed.answer
    record.closed_samples = unc.answers
    record.closed_cluster_assignments = unc.cluster_assignments
    record.closed_n_clusters = unc.n_clusters
    record.closed_semantic_entropy = unc.semantic_entropy
    record.closed_normalized_entropy = unc.normalized_entropy
    record.closed_correctness = correctness


def run_phase_c(
    cfg: Config,
    qa: CrimeQAAgent,
    llm: GeminiClient,
    gcc: GeneratorCriticChallenger,
    seed: Seed,
    record: TrialRecord,
) -> None:
    trace = gcc.build_probe(
        seed_q=seed.text,
        initial_answer=record.closed_answer,
        sampled=record.closed_samples,
        initial_uncertainty=record.closed_normalized_entropy,
    )
    record.probe_method = trace.method
    record.probe_generator_q = trace.generator_q
    record.probe_critic_q = trace.critic_q
    record.probe_challenger_q = trace.challenger_q
    record.probe_final = trace.final_probe

    rag = qa.rag_with_probe_context(seed.text, trace.final_probe)
    samples = qa.sample_rag_with_probe_context(seed.text, trace.final_probe, cfg.n_samples_for_entropy)
    unc = estimate_uncertainty(cfg, samples, llm=llm)
    correctness = judge_correctness(
        cfg, seed.text, rag.answer, seed.gold_answer, seed.gold_kind, llm=llm
    )
    contradicted = detect_contradiction(cfg, record.closed_answer, rag.answer, llm=llm)

    record.rag_answer = rag.answer
    record.rag_context = rag.context
    record.rag_samples = unc.answers
    record.rag_cluster_assignments = unc.cluster_assignments
    record.rag_n_clusters = unc.n_clusters
    record.rag_semantic_entropy = unc.semantic_entropy
    record.rag_normalized_entropy = unc.normalized_entropy
    record.rag_correctness = correctness
    record.contradiction_pre_vs_post = contradicted
    record.delta_uncertainty = unc.normalized_entropy - record.closed_normalized_entropy
    record.recovered = (
        is_fail(record.closed_correctness, seed.gold_kind)
        and is_pass(record.rag_correctness, seed.gold_kind)
    )
    record.probed = True


# ------------------------- Aggregate metrics -----------------------------

def _accuracy(records: List[TrialRecord], judgement: str) -> float:
    """Fraction of records whose judgement field passes the gold-kind-aware rule.

    judgement: 'closed_correctness' or 'augmented_correctness'.
    For non-probed trials, the augmented score inherits closed_correctness
    (because the system did not need probing — closed-book was already correct).
    """
    if not records:
        return 0.0
    n_pass = 0
    for r in records:
        if judgement == "closed_correctness":
            if is_pass(r.closed_correctness, r.gold_kind):
                n_pass += 1
        elif judgement == "augmented_correctness":
            final = r.rag_correctness if r.probed else r.closed_correctness
            if is_pass(final, r.gold_kind):
                n_pass += 1
    return n_pass / len(records)


def aggregate_metrics(records: List[TrialRecord]) -> Dict:
    n = len(records)
    probed = [r for r in records if r.probed]
    recovered = [r for r in probed if r.recovered]

    by_cat: Dict[str, Dict] = {}
    for cat in sorted({r.category for r in records}):
        sub = [r for r in records if r.category == cat]
        sub_probed = [r for r in sub if r.probed]
        by_cat[cat] = {
            "n": len(sub),
            "n_probed": len(sub_probed),
            "closed_book_accuracy": round(_accuracy(sub, "closed_correctness"), 4),
            "augmented_accuracy": round(_accuracy(sub, "augmented_correctness"), 4),
            "recovery_rate": round(
                sum(1 for r in sub_probed if r.recovered) / len(sub_probed), 4
            ) if sub_probed else 0.0,
        }

    by_ds: Dict[str, Dict] = {}
    for ds in sorted({r.dataset for r in records}):
        sub = [r for r in records if r.dataset == ds]
        sub_probed = [r for r in sub if r.probed]
        by_ds[ds] = {
            "n": len(sub),
            "n_probed": len(sub_probed),
            "closed_book_accuracy": round(_accuracy(sub, "closed_correctness"), 4),
            "augmented_accuracy": round(_accuracy(sub, "augmented_correctness"), 4),
            "recovery_rate": round(
                sum(1 for r in sub_probed if r.recovered) / len(sub_probed), 4
            ) if sub_probed else 0.0,
        }

    return {
        "n_trials": n,
        "n_probed": len(probed),
        "n_recovered": len(recovered),
        "closed_book_accuracy_overall": round(_accuracy(records, "closed_correctness"), 4),
        "augmented_accuracy_overall": round(_accuracy(records, "augmented_correctness"), 4),
        "accuracy_lift": round(
            _accuracy(records, "augmented_accuracy" if False else "augmented_correctness")
            - _accuracy(records, "closed_correctness"),
            4,
        ),
        "recovery_rate_overall": round(
            len(recovered) / len(probed), 4
        ) if probed else 0.0,
        "mean_delta_uncertainty_on_probed": round(
            sum(r.delta_uncertainty for r in probed) / len(probed), 4
        ) if probed else 0.0,
        "contradiction_rate_on_probed": round(
            sum(1 for r in probed if r.contradiction_pre_vs_post) / len(probed), 4
        ) if probed else 0.0,
        "by_category": by_cat,
        "by_dataset": by_ds,
    }


# ------------------------- Output writers --------------------------------

def render_audit(records: List[TrialRecord]) -> str:
    parts: List[str] = []
    for i, r in enumerate(records, start=1):
        parts.append("=" * 100)
        parts.append(f"TRIAL #{i}/{len(records)} | dataset={r.dataset} | category={r.category}")
        parts.append("=" * 100)
        parts.append(f"SEED:      {r.seed_text}")
        parts.append(f"GOLD ({r.gold_kind}): {r.gold_answer}")
        parts.append("")
        parts.append("PHASE A — CLOSED-BOOK QA (no RAG, no probing)")
        parts.append(f"  closed_answer:  {r.closed_answer}")
        parts.append(f"  closed_samples ({len(r.closed_samples)} @ T=0.7):")
        for k, (a, c) in enumerate(zip(r.closed_samples, r.closed_cluster_assignments)):
            parts.append(f"    [{k+1}] (cluster {c}) {a}")
        parts.append(f"  closed_uncertainty: H={r.closed_semantic_entropy:.3f} | "
                     f"H_norm={r.closed_normalized_entropy:.3f} | n_clusters={r.closed_n_clusters}")
        parts.append(f"  CORRECTNESS A: {r.closed_correctness}")
        parts.append("")
        if r.probed:
            parts.append("PHASE C — PROBING + RECOVERY")
            parts.append(f"  generator_q:  {r.probe_generator_q}")
            parts.append(f"  critic_q:     {r.probe_critic_q}")
            parts.append(f"  challenger_q: {r.probe_challenger_q}")
            parts.append(f"  final probe:  {r.probe_final}")
            parts.append("")
            parts.append(f"  rag_answer (RAG with probe-driven context):")
            parts.append(f"    {r.rag_answer}")
            parts.append(f"  rag_samples ({len(r.rag_samples)} @ T=0.7):")
            for k, (a, c) in enumerate(zip(r.rag_samples, r.rag_cluster_assignments)):
                parts.append(f"    [{k+1}] (cluster {c}) {a}")
            parts.append(f"  rag_uncertainty: H={r.rag_semantic_entropy:.3f} | "
                         f"H_norm={r.rag_normalized_entropy:.3f} | n_clusters={r.rag_n_clusters}")
            parts.append(f"  CORRECTNESS C:  {r.rag_correctness}")
            parts.append(f"  Δuncertainty:   {r.delta_uncertainty:+.3f}")
            parts.append(f"  contradiction:  {r.contradiction_pre_vs_post}")
            parts.append(f"  RECOVERED:      {r.recovered}")
        else:
            parts.append("PHASE C — SKIPPED (closed-book already passed; policy gate did not fire)")
        parts.append("")
    return "\n".join(parts)


def render_markdown(records: List[TrialRecord], metrics: Dict, figures: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Stress-Testing Research Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append("**Phase A — Closed-book QA**: ask the seed, no retrieval. Score against gold.")
    lines.append("")
    lines.append("**Policy gate** (gold-kind aware): probe whenever Phase A is a *recoverable gap*.")
    lines.append("- ambiguous (hedged-gold): `correct` or `hedged_appropriate` → pass; everything else → probe")
    lines.append("- factual / statistical / comparative: only `correct` is a pass; "
                 "`hedged_appropriate` is treated as a recoverable gap (the model said *I don't know* — "
                 "exactly what the RAG path is meant to fix) and triggers Phase C.")
    lines.append("")
    lines.append("**Phase C — Multi-agent probing**: Generator → Critic → Challenger build a probe.")
    lines.append("The probe drives retrieval; QA re-answers the seed using the retrieved context.")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append(f"- trials: **{metrics['n_trials']}**")
    lines.append(f"- probed (Phase A failed): **{metrics['n_probed']}**")
    lines.append(f"- recovered (probing rescued the answer): **{metrics['n_recovered']}**")
    lines.append("")
    lines.append(f"- closed-book accuracy: **{metrics['closed_book_accuracy_overall']:.0%}**")
    lines.append(f"- probing-augmented accuracy: **{metrics['augmented_accuracy_overall']:.0%}**")
    lines.append(f"- accuracy lift: **{metrics['accuracy_lift']:+.0%}**")
    lines.append(f"- recovery rate (rescued / probed): **{metrics['recovery_rate_overall']:.0%}**")
    lines.append(f"- mean Δuncertainty on probed trials: **{metrics['mean_delta_uncertainty_on_probed']:+.3f}**")
    lines.append(f"- contradiction rate on probed trials: **{metrics['contradiction_rate_on_probed']:.0%}**")
    lines.append("")
    lines.append("### By dataset")
    lines.append("")
    lines.append("| dataset | n | n_probed | closed_book_acc | augmented_acc | recovery |")
    lines.append("|---------|---|----------|-----------------|---------------|----------|")
    for ds, m in metrics["by_dataset"].items():
        lines.append(f"| {ds} | {m['n']} | {m['n_probed']} | {m['closed_book_accuracy']:.0%} | {m['augmented_accuracy']:.0%} | {m['recovery_rate']:.0%} |")
    lines.append("")
    lines.append("### By category")
    lines.append("")
    lines.append("| category | n | n_probed | closed_book_acc | augmented_acc | recovery |")
    lines.append("|----------|---|----------|-----------------|---------------|----------|")
    for cat, m in metrics["by_category"].items():
        lines.append(f"| {cat} | {m['n']} | {m['n_probed']} | {m['closed_book_accuracy']:.0%} | {m['augmented_accuracy']:.0%} | {m['recovery_rate']:.0%} |")
    lines.append("")

    if figures:
        lines.append("## Figures")
        lines.append("")
        for fig in figures:
            lines.append(f"![{Path(fig).stem}]({fig})")
            lines.append("")

    lines.append("## Per-trial details (compact)")
    lines.append("")
    for i, r in enumerate(records, start=1):
        lines.append(f"### {i}. [{r.dataset}/{r.category}] {r.seed_text}")
        lines.append(f"- gold ({r.gold_kind}): {r.gold_answer}")
        lines.append(f"- closed-book: _{r.closed_answer}_  → **{r.closed_correctness}**")
        if r.probed:
            lines.append(f"- generator: _{r.probe_generator_q}_")
            lines.append(f"- critic:    _{r.probe_critic_q}_")
            lines.append(f"- challenger: _{r.probe_challenger_q}_")
            lines.append(f"- rag answer: _{r.rag_answer}_  → **{r.rag_correctness}**  | recovered={r.recovered}")
        else:
            lines.append("- (Phase C skipped — closed-book already passed)")
        lines.append("")
    return "\n".join(lines)


def write_tsv(path: Path, records: List[TrialRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "trial_idx", "dataset", "category", "gold_kind", "seed_text", "gold_answer",
            "closed_answer", "closed_correctness", "closed_unc",
            "probed",
            "generator_q", "critic_q", "challenger_q", "final_probe",
            "rag_answer", "rag_correctness", "rag_unc",
            "delta_unc", "contradiction", "recovered",
        ])
        for i, r in enumerate(records, start=1):
            w.writerow([
                i, r.dataset, r.category, r.gold_kind, r.seed_text, r.gold_answer,
                r.closed_answer, r.closed_correctness, f"{r.closed_normalized_entropy:.3f}",
                r.probed,
                r.probe_generator_q, r.probe_critic_q, r.probe_challenger_q, r.probe_final,
                r.rag_answer, r.rag_correctness, f"{r.rag_normalized_entropy:.3f}",
                f"{r.delta_uncertainty:+.3f}", r.contradiction_pre_vs_post, r.recovered,
            ])


def write_json(
    path: Path,
    cfg: Config,
    records: List[TrialRecord],
    metrics: Dict,
    n_attempted: int,
    fatal_error: Optional[Exception],
) -> None:
    trials = []
    for i, r in enumerate(records, start=1):
        trials.append({
            "trial_id": i,
            "dataset": r.dataset,
            "category": r.category,
            "seed_question": r.seed_text,
            "gold": {"answer": r.gold_answer, "kind": r.gold_kind},
            "phase_a_closed_book": {
                "answer": r.closed_answer,
                "samples": r.closed_samples,
                "cluster_assignments": r.closed_cluster_assignments,
                "n_clusters": r.closed_n_clusters,
                "semantic_entropy": r.closed_semantic_entropy,
                "normalized_entropy": r.closed_normalized_entropy,
                "correctness": r.closed_correctness,
            },
            "policy_gate": {
                "probed": r.probed,
                "rule": (
                    "ambiguous question: hedging counts as pass; "
                    "factual / statistical / comparative: only verified-correct passes "
                    "(hedging treated as recoverable gap)."
                ),
                "reason": (
                    f"closed-book judged {r.closed_correctness} "
                    f"(gold_kind={r.gold_kind}) → "
                    + ("PASS, no probe" if not r.probed else "FAIL, probe fired")
                ),
            },
            "phase_c_probe_and_recover": (
                None if not r.probed else {
                    "probe_method": r.probe_method,
                    "generator_question": r.probe_generator_q,
                    "critic_question": r.probe_critic_q,
                    "challenger_question": r.probe_challenger_q,
                    "final_probe": r.probe_final,
                    "rag_answer": r.rag_answer,
                    "rag_samples": r.rag_samples,
                    "rag_cluster_assignments": r.rag_cluster_assignments,
                    "rag_n_clusters": r.rag_n_clusters,
                    "rag_semantic_entropy": r.rag_semantic_entropy,
                    "rag_normalized_entropy": r.rag_normalized_entropy,
                    "rag_correctness": r.rag_correctness,
                    "delta_uncertainty": r.delta_uncertainty,
                    "contradiction_pre_vs_post": r.contradiction_pre_vs_post,
                    "recovered": r.recovered,
                }
            ),
        })

    payload = {
        "experiment_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": cfg.chat_model,
            "n_samples_for_entropy": cfg.n_samples_for_entropy,
            "rag_top_k": cfg.rag_top_k,
            "n_seeds_attempted": n_attempted,
            "n_seeds_completed": len(records),
            "fatal_error": str(fatal_error) if fatal_error else None,
            "metric_definitions": {
                "closed_book_accuracy": "fraction of trials where Phase A judged correct/hedged",
                "augmented_accuracy":   "fraction of trials where final answer (after probing if any) judged correct/hedged",
                "recovery_rate":        "fraction of probed trials where probing recovered a correct answer from a wrong closed-book one",
                "delta_uncertainty":    "rag_normalized_entropy − closed_normalized_entropy on probed trials",
            },
        },
        "trials": trials,
        "evaluation_metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --------------------------- Main ----------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Research-grade stress-testing runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds")
    parser.add_argument("--samples", type=int, default=None, help="Override n_samples_for_entropy")
    parser.add_argument("--phoenix-only", action="store_true")
    parser.add_argument("--ushom-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    if args.samples:
        cfg.n_samples_for_entropy = args.samples

    rng = random.Random(args.seed)
    seeds: List[Seed] = []
    datasets: Dict[str, CrimeDataset] = {}
    if not args.ushom_only:
        ph = PhoenixCrime.load()
        datasets["phoenix"] = ph
        seeds += generate_phoenix_seeds(ph, rng)
    if not args.phoenix_only:
        us = USHomicide.load()
        datasets["us_homicide"] = us
        seeds += generate_ushom_seeds(us, rng)

    if args.limit:
        seeds = seeds[: args.limit]

    _print(f"loaded {len(seeds)} seeds across datasets {list(datasets.keys())}; "
           f"samples_per_uncertainty={cfg.n_samples_for_entropy}; model={cfg.chat_model}")

    llm = GeminiClient(cfg)
    qa_by_ds = {ds_name: CrimeQAAgent(cfg, ds, llm=llm) for ds_name, ds in datasets.items()}
    gcc = GeneratorCriticChallenger(cfg, llm=llm)

    records: List[TrialRecord] = []
    fatal_error: Optional[Exception] = None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.outputs_dir.parent / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / f"figures_{ts}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"research_results_{ts}.json"
    md_path = out_dir / f"research_report_{ts}.md"
    audit_path = out_dir / f"research_audit_{ts}.txt"
    tsv_path = out_dir / f"research_results_{ts}.tsv"

    def checkpoint(figures: Optional[List[str]] = None) -> None:
        metrics = aggregate_metrics(records)
        json_path.write_text(
            "", encoding="utf-8"
        )
        write_json(json_path, cfg, records, metrics, n_attempted=len(seeds), fatal_error=fatal_error)
        md_path.write_text(render_markdown(records, metrics, figures or []), encoding="utf-8")
        audit_path.write_text(render_audit(records), encoding="utf-8")
        write_tsv(tsv_path, records)

    for i, seed in enumerate(seeds, start=1):
        _print(f"  trial {i}/{len(seeds)} [{seed.dataset}/{seed.category}]: {seed.text!r}")
        record = _build_trial_record_from_seed(seed)
        try:
            qa = qa_by_ds[seed.dataset]
            run_phase_a(cfg, qa, llm, seed, record)
            _print(f"    A: {record.closed_correctness} (gold_kind={seed.gold_kind})")
            if is_fail(record.closed_correctness, seed.gold_kind):
                run_phase_c(cfg, qa, llm, gcc, seed, record)
                _print(
                    f"    C: {record.rag_correctness} | "
                    f"recovered={record.recovered} | "
                    f"Δunc={record.delta_uncertainty:+.3f} | "
                    f"contradiction={record.contradiction_pre_vs_post}"
                )
            else:
                _print("    C: skipped (passed)")
            records.append(record)
            checkpoint()
        except KeyboardInterrupt:
            _print("!! interrupted")
            break
        except Exception as e:
            _print(f"!! error on trial {i}: {type(e).__name__}: {e}")
            fatal_error = e
            checkpoint()
            break

    # Generate figures from final state
    try:
        from .graphs import render_all_figures
        figs = render_all_figures(records, fig_dir)
        rel_figs = [str(f.relative_to(out_dir)) for f in figs]
        _print(f"figures written: {len(rel_figs)}")
        checkpoint(figures=rel_figs)
    except Exception as e:
        _print(f"!! figure rendering failed: {type(e).__name__}: {e}")

    _print(f"json:   {json_path}")
    _print(f"md:     {md_path}")
    _print(f"audit:  {audit_path}")
    _print(f"tsv:    {tsv_path}")
    _print(f"figs:   {fig_dir}")


if __name__ == "__main__":
    main()
