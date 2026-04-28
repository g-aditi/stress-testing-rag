"""Main orchestrator: run probing experiments and produce a comparison report.

For every seed question we run TWO probing strategies in sequence:

  - SingleShotBaseline           one LLM call, generic follow-up (API-generated)
  - GeneratorCriticChallenger    three LLM calls: Generator -> Critic -> Challenger

Both strategies are scored with identical evaluation logic.

Pipeline per (seed, method):
  STAGE 1 - INITIAL QA:    answer the seed at low temp -> sample N at high temp
                           -> semantic-entropy clustering (Gemini judge)
                           -> pre_uncertainty
  STAGE 2 - PROBE BUILD:   single-shot OR generator/critic/challenger
  STAGE 3 - POST QA:       answer the probe at low temp -> sample N at high temp
                           -> semantic-entropy clustering -> post_uncertainty
  STAGE 4 - SCORING:       Δuncertainty, answer_changed (LLM judge), contradiction
                           (LLM judge), is_effective

After all trials we aggregate across probes per method.

Outputs (in outputs/probing/):
  - probing_results_<ts>.json   STRUCTURED: experiment_metadata, trials[],
                                evaluation_metrics  (metrics block at end)
  - probing_audit_<ts>.txt      One readable block per probe with everything
  - probing_results_<ts>.tsv    Flat row-per-probe for spreadsheet review
  - probing_report_<ts>.md      Human-readable summary

Both the JSON and the audit log checkpoint after every seed so partial runs
are recoverable.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .config import Config, load_config
from .evaluation import (
    AggregateMetrics,
    ProbeOutcome,
    aggregate,
    detect_answer_change,
    detect_contradiction,
)
from .llm_client import GeminiClient
from .probing_agents import (
    GeneratorCriticChallenger,
    ProbeTrace,
    SingleShotBaseline,
)
from .qa_agent import CrimeQAAgent
from .seed_questions import SEED_QUESTIONS, SeedQuestion
from .uncertainty import UncertaintyReport, estimate_uncertainty


def _print(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def run_one(
    cfg: Config,
    qa: CrimeQAAgent,
    llm: GeminiClient,
    seed: SeedQuestion,
    probe_builder,
) -> Tuple[ProbeTrace, ProbeOutcome, UncertaintyReport, UncertaintyReport, str]:
    pre_result = qa.answer(seed.text)
    pre_samples = qa.sample_answers(seed.text, cfg.n_samples_for_entropy)
    pre_unc = estimate_uncertainty(cfg, pre_samples, llm=llm)

    trace = probe_builder.build_probe(
        seed_q=seed.text,
        initial_answer=pre_result.answer,
        sampled=pre_samples,
        initial_uncertainty=pre_unc.normalized_entropy,
    )

    post_result = qa.answer(trace.final_probe)
    post_samples = qa.sample_answers(trace.final_probe, cfg.n_samples_for_entropy)
    post_unc = estimate_uncertainty(cfg, post_samples, llm=llm)

    answer_changed = detect_answer_change(cfg, pre_result.answer, post_result.answer, llm=llm)
    contradiction = detect_contradiction(cfg, pre_result.answer, post_result.answer, llm=llm)

    outcome = ProbeOutcome(
        seed_question=seed.text,
        probe_question=trace.final_probe,
        pre_answer=pre_result.answer,
        post_answer=post_result.answer,
        pre_uncertainty=pre_unc.normalized_entropy,
        post_uncertainty=post_unc.normalized_entropy,
        answer_changed=answer_changed,
        contradiction=contradiction,
    )
    _print(
        f"    [{trace.method}] Δunc={outcome.uncertainty_delta:+.3f} "
        f"changed={answer_changed} contradicted={contradiction} "
        f"effective={outcome.is_effective}"
    )
    return trace, outcome, pre_unc, post_unc, pre_result.context


def _build_method_block(
    trace: ProbeTrace,
    outcome: ProbeOutcome,
    pre_unc: UncertaintyReport,
    post_unc: UncertaintyReport,
) -> dict:
    """Per-trial-per-method JSON block in stage form: question -> probe -> score."""
    if trace.method == "agentic_gcc":
        probe_stage = {
            "method": "Generator -> Critic -> Challenger",
            "generator_question": trace.generator_q,
            "critic_question": trace.critic_q,
            "challenger_question": trace.challenger_q,
            "final_probe": trace.final_probe,
        }
    else:
        probe_stage = {
            "method": "single-shot LLM follow-up",
            "final_probe": trace.final_probe,
        }
    return {
        "stage_1_initial_qa": {
            "pre_answer": outcome.pre_answer,
            "pre_answer_samples": pre_unc.answers,
            "pre_cluster_assignments": pre_unc.cluster_assignments,
            "pre_n_clusters": pre_unc.n_clusters,
            "pre_semantic_entropy": pre_unc.semantic_entropy,
            "pre_normalized_entropy": pre_unc.normalized_entropy,
        },
        "stage_2_probe": probe_stage,
        "stage_3_post_qa": {
            "post_answer": outcome.post_answer,
            "post_answer_samples": post_unc.answers,
            "post_cluster_assignments": post_unc.cluster_assignments,
            "post_n_clusters": post_unc.n_clusters,
            "post_semantic_entropy": post_unc.semantic_entropy,
            "post_normalized_entropy": post_unc.normalized_entropy,
        },
        "stage_4_scoring": {
            "delta_uncertainty": outcome.uncertainty_delta,
            "answer_changed": outcome.answer_changed,
            "contradiction": outcome.contradiction,
            "is_effective": outcome.is_effective,
        },
    }


def render_audit_log(seeds: List[SeedQuestion], method_results: Dict[str, Dict]) -> str:
    parts: List[str] = []
    for i, seed in enumerate(seeds):
        for method, payload in method_results.items():
            t: ProbeTrace = payload["traces"][i]
            o: ProbeOutcome = payload["outcomes"][i]
            pre_unc: UncertaintyReport = payload["pre_unc"][i]
            post_unc: UncertaintyReport = payload["post_unc"][i]

            parts.append("=" * 100)
            parts.append(f"TRIAL #{i+1}/{len(seeds)} [{seed.category}] | METHOD: {method}")
            parts.append("=" * 100)
            parts.append(f"SEED QUESTION: {seed.text}")
            parts.append("")
            parts.append("STAGE 1 - INITIAL QA")
            parts.append(f"  pre_answer (T=0.2, RAG):")
            parts.append(f"    {o.pre_answer}")
            parts.append(f"  pre samples ({len(pre_unc.answers)} @ T=0.7):")
            for k, (ans, cid) in enumerate(zip(pre_unc.answers, pre_unc.cluster_assignments)):
                parts.append(f"    [{k+1}] (cluster {cid}) {ans}")
            parts.append(
                f"  pre_uncertainty: n_clusters={pre_unc.n_clusters} | "
                f"H={pre_unc.semantic_entropy:.3f} | H_norm={pre_unc.normalized_entropy:.3f}"
            )
            parts.append("")
            parts.append("STAGE 2 - PROBE")
            if method == "agentic_gcc":
                parts.append(f"  generator_q:  {t.generator_q}")
                parts.append(f"  critic_q:     {t.critic_q}")
                parts.append(f"  challenger_q: {t.challenger_q}")
                parts.append(f"  final probe:  {t.final_probe}")
            else:
                parts.append(f"  single-shot probe: {t.final_probe}")
            parts.append("")
            parts.append("STAGE 3 - POST QA")
            parts.append(f"  post_answer (T=0.2, RAG):")
            parts.append(f"    {o.post_answer}")
            parts.append(f"  post samples ({len(post_unc.answers)} @ T=0.7):")
            for k, (ans, cid) in enumerate(zip(post_unc.answers, post_unc.cluster_assignments)):
                parts.append(f"    [{k+1}] (cluster {cid}) {ans}")
            parts.append(
                f"  post_uncertainty: n_clusters={post_unc.n_clusters} | "
                f"H={post_unc.semantic_entropy:.3f} | H_norm={post_unc.normalized_entropy:.3f}"
            )
            parts.append("")
            parts.append("STAGE 4 - SCORING")
            parts.append(f"  Δuncertainty   = {o.uncertainty_delta:+.3f}")
            parts.append(f"  answer_changed = {o.answer_changed}    (Gemini judge SAME/DIFFERENT)")
            parts.append(f"  contradiction  = {o.contradiction}    (Gemini judge YES/NO)")
            parts.append(f"  is_effective   = {o.is_effective}    (any of: Δunc>0.05, change, contradict)")
            parts.append("")
    return "\n".join(parts)


def render_markdown(seeds: List[SeedQuestion], method_results: Dict[str, Dict]) -> str:
    lines: List[str] = []
    lines.append("# Phoenix Crime QA — Stress-Testing Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Target QA: `CrimeQAAgent` — RAG over Phoenix open crime CSV")
    lines.append("- Backend: Google Gemini (no embeddings; LLM-judge clustering for semantic entropy)")
    lines.append("- Probe methods: **baseline_singleshot** vs **agentic_gcc** (Generator -> Critic -> Challenger)")
    lines.append(f"- Seed questions completed: {len(seeds)}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    for method, payload in method_results.items():
        m: AggregateMetrics = payload["metrics"]
        lines.append(f"### {method}")
        lines.append(f"- n_probes: **{m.n_probes}**")
        lines.append(f"- mean uncertainty increase: **{m.mean_uncertainty_increase:+.3f}**")
        lines.append(f"- contradiction rate: **{m.contradiction_rate:.0%}**")
        lines.append(f"- diagnostic effectiveness: **{m.diagnostic_effectiveness:.0%}**")
        lines.append(f"- gaps per question: **{m.gaps_per_question:.2f}**")
        lines.append("")

    if "agentic_gcc" in method_results and "baseline_singleshot" in method_results:
        a = method_results["agentic_gcc"]["metrics"]
        b = method_results["baseline_singleshot"]["metrics"]
        lines.append("### Δ (agentic_gcc − baseline_singleshot)")
        lines.append(f"- Δ mean uncertainty increase: **{a.mean_uncertainty_increase - b.mean_uncertainty_increase:+.3f}**")
        lines.append(f"- Δ contradiction rate:        **{(a.contradiction_rate - b.contradiction_rate):+.0%}**")
        lines.append(f"- Δ diagnostic effectiveness:  **{(a.diagnostic_effectiveness - b.diagnostic_effectiveness):+.0%}**")
        lines.append(f"- Δ gaps per question:         **{a.gaps_per_question - b.gaps_per_question:+.2f}**")
        lines.append("")

    lines.append("## Per-trial summaries")
    lines.append("")
    for i, seed in enumerate(seeds):
        lines.append(f"### Trial {i+1} [{seed.category}]: {seed.text}")
        lines.append("")
        for method, payload in method_results.items():
            t: ProbeTrace = payload["traces"][i]
            o: ProbeOutcome = payload["outcomes"][i]
            lines.append(f"#### {method}")
            lines.append(f"- pre_answer (unc={o.pre_uncertainty:.3f}): {o.pre_answer}")
            if t.method == "agentic_gcc":
                lines.append(f"- generator:  _{t.generator_q}_")
                lines.append(f"- critic:     _{t.critic_q}_")
                lines.append(f"- challenger: _{t.challenger_q}_")
            lines.append(f"- final probe: **{t.final_probe}**")
            lines.append(f"- post_answer (unc={o.post_uncertainty:.3f}): {o.post_answer}")
            lines.append(
                f"- Δunc={o.uncertainty_delta:+.3f} | answer_changed={o.answer_changed} | "
                f"contradiction={o.contradiction} | effective={o.is_effective}"
            )
            lines.append("")
    return "\n".join(lines)


def write_tsv(path: Path, seeds: List[SeedQuestion], method_results: Dict[str, Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "trial_idx", "category", "method", "seed_question",
            "pre_answer", "pre_unc",
            "generator_q", "critic_q", "challenger_q", "final_probe",
            "post_answer", "post_unc",
            "delta_unc", "answer_changed", "contradiction", "is_effective",
        ])
        for i, seed in enumerate(seeds):
            for method, payload in method_results.items():
                t: ProbeTrace = payload["traces"][i]
                o: ProbeOutcome = payload["outcomes"][i]
                w.writerow([
                    i + 1, seed.category, method, seed.text,
                    o.pre_answer, f"{o.pre_uncertainty:.3f}",
                    t.generator_q, t.critic_q, t.challenger_q, t.final_probe,
                    o.post_answer, f"{o.post_uncertainty:.3f}",
                    f"{o.uncertainty_delta:+.3f}",
                    o.answer_changed, o.contradiction, o.is_effective,
                ])


def write_structured_json(
    path: Path,
    cfg: Config,
    seeds: List[SeedQuestion],
    method_results: Dict[str, Dict],
    n_attempted: int,
    fatal_error: object | None,
) -> None:
    """Layout: experiment_metadata, trials[], evaluation_metrics (at end)."""
    trials = []
    for i, seed in enumerate(seeds):
        trial = {
            "trial_id": i + 1,
            "category": seed.category,
            "seed_question": seed.text,
        }
        for method, payload in method_results.items():
            trial[method] = _build_method_block(
                payload["traces"][i],
                payload["outcomes"][i],
                payload["pre_unc"][i],
                payload["post_unc"][i],
            )
        trials.append(trial)

    metrics_block = {}
    for method, payload in method_results.items():
        m: AggregateMetrics = payload["metrics"]
        metrics_block[method] = {
            "n_probes": m.n_probes,
            "mean_uncertainty_increase": round(m.mean_uncertainty_increase, 4),
            "contradiction_rate": round(m.contradiction_rate, 4),
            "diagnostic_effectiveness": round(m.diagnostic_effectiveness, 4),
            "gaps_per_question": round(m.gaps_per_question, 4),
        }
    if "agentic_gcc" in metrics_block and "baseline_singleshot" in metrics_block:
        a = metrics_block["agentic_gcc"]
        b = metrics_block["baseline_singleshot"]
        metrics_block["delta_agentic_minus_baseline"] = {
            "mean_uncertainty_increase": round(a["mean_uncertainty_increase"] - b["mean_uncertainty_increase"], 4),
            "contradiction_rate": round(a["contradiction_rate"] - b["contradiction_rate"], 4),
            "diagnostic_effectiveness": round(a["diagnostic_effectiveness"] - b["diagnostic_effectiveness"], 4),
            "gaps_per_question": round(a["gaps_per_question"] - b["gaps_per_question"], 4),
        }

    payload = {
        "experiment_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": cfg.chat_model,
            "n_samples_for_entropy": cfg.n_samples_for_entropy,
            "rag_top_k": cfg.rag_top_k,
            "n_seeds_attempted": n_attempted,
            "n_seeds_completed": len(seeds),
            "fatal_error": str(fatal_error) if fatal_error else None,
            "methods_compared": list(method_results.keys()),
            "metric_definitions": {
                "mean_uncertainty_increase": "mean(post_unc - pre_unc) — semantic entropy delta",
                "contradiction_rate": "fraction of probes where Gemini judge said pre/post are contradictory",
                "diagnostic_effectiveness": "fraction of probes that triggered any-of {Δunc>0.05, answer_changed, contradiction}",
                "gaps_per_question": "same as diagnostic_effectiveness, framed per-question",
            },
        },
        "trials": trials,
        "evaluation_metrics": metrics_block,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crime QA stress-test runner")
    parser.add_argument("--limit", type=int, default=None, help="Number of seed questions to run")
    parser.add_argument("--samples", type=int, default=None, help="Override n_samples_for_entropy")
    args = parser.parse_args()

    cfg = load_config()
    if args.samples:
        cfg.n_samples_for_entropy = args.samples

    seeds_all = SEED_QUESTIONS[: args.limit] if args.limit else SEED_QUESTIONS[: cfg.seed_questions_limit]
    _print(f"running {len(seeds_all)} seed questions x 2 methods (singleshot, gcc); "
           f"samples_per_uncertainty={cfg.n_samples_for_entropy}; model={cfg.chat_model}")

    llm = GeminiClient(cfg)
    qa = CrimeQAAgent(cfg, llm=llm)
    methods = [
        ("baseline_singleshot", SingleShotBaseline(cfg, llm=llm)),
        ("agentic_gcc", GeneratorCriticChallenger(cfg, llm=llm)),
    ]

    method_results: Dict[str, Dict] = {
        name: {"traces": [], "outcomes": [], "pre_unc": [], "post_unc": []}
        for name, _ in methods
    }
    completed_seeds: List[SeedQuestion] = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = cfg.outputs_dir / f"probing_report_{ts}.md"
    audit_path = cfg.outputs_dir / f"probing_audit_{ts}.txt"
    tsv_path = cfg.outputs_dir / f"probing_results_{ts}.tsv"
    json_path = cfg.outputs_dir / f"probing_results_{ts}.json"

    fatal_error: Exception | None = None

    def checkpoint(reason: str) -> None:
        if not completed_seeds:
            _print(f"checkpoint skipped ({reason}): no seeds completed yet")
            return
        for name in method_results:
            method_results[name]["metrics"] = aggregate(method_results[name]["outcomes"])
        md_path.write_text(render_markdown(completed_seeds, method_results), encoding="utf-8")
        audit_path.write_text(render_audit_log(completed_seeds, method_results), encoding="utf-8")
        write_tsv(tsv_path, completed_seeds, method_results)
        write_structured_json(json_path, cfg, completed_seeds, method_results,
                              n_attempted=len(seeds_all), fatal_error=fatal_error)
        _print(f"checkpoint written ({reason}): {len(completed_seeds)}/{len(seeds_all)} seeds")

    for i, seed in enumerate(seeds_all):
        _print(f"  seed {i+1}/{len(seeds_all)} [{seed.category}]: {seed.text!r}")
        try:
            seed_results = []
            for method_name, builder in methods:
                t, o, pre_u, post_u, _ctx = run_one(cfg, qa, llm, seed, builder)
                seed_results.append((method_name, t, o, pre_u, post_u))
            for method_name, t, o, pre_u, post_u in seed_results:
                method_results[method_name]["traces"].append(t)
                method_results[method_name]["outcomes"].append(o)
                method_results[method_name]["pre_unc"].append(pre_u)
                method_results[method_name]["post_unc"].append(post_u)
            completed_seeds.append(seed)
            checkpoint(f"after seed {i+1}")
        except KeyboardInterrupt:
            _print("!! interrupted by user")
            break
        except Exception as e:
            _print(f"!! error on seed {i+1}: {type(e).__name__}: {e}")
            fatal_error = e
            checkpoint("after error")
            break

    _print(f"markdown report: {md_path}")
    _print(f"audit log:       {audit_path}")
    _print(f"tsv results:     {tsv_path}")
    _print(f"json results:    {json_path}")
    if fatal_error is not None:
        _print(
            f"NOTE: run halted early after {len(completed_seeds)}/{len(seeds_all)} seeds "
            f"due to: {type(fatal_error).__name__}"
        )


if __name__ == "__main__":
    main()
