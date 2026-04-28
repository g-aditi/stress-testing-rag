"""Build the consolidated, paper-grade research report.

Reads the latest `research_results_<ts>.json` plus its accompanying figures
folder and renders a single Markdown file (`RESEARCH_REPORT_<ts>.md`) that
combines:

  - System overview & architecture diagram
  - Datasets and seed-generation methodology
  - Aggregate evaluation tables (overall, by dataset, by category)
  - All six figures embedded as images
  - One illustrative trial per question category showing the full
    closed-book → probe → recover trace
  - Pointers to the raw artefacts (audit txt, JSON, TSV)

Run:

    python -m probing.build_final_report                # uses latest results
    python -m probing.build_final_report --json <path>  # specific results file
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "outputs" / "research"


def _latest_results_json() -> Path:
    candidates = sorted(RESEARCH_DIR.glob("research_results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no research_results_*.json in {RESEARCH_DIR}")
    return candidates[-1]


def _figures_dir_for(json_path: Path) -> Optional[Path]:
    ts = json_path.stem.replace("research_results_", "")
    fig_dir = json_path.parent / f"figures_{ts}"
    return fig_dir if fig_dir.exists() else None


def _audit_for(json_path: Path) -> Optional[Path]:
    ts = json_path.stem.replace("research_results_", "")
    audit = json_path.parent / f"research_audit_{ts}.txt"
    return audit if audit.exists() else None


def _tsv_for(json_path: Path) -> Optional[Path]:
    ts = json_path.stem.replace("research_results_", "")
    tsv = json_path.parent / f"research_results_{ts}.tsv"
    return tsv if tsv.exists() else None


def _markdown_table(rows: List[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _signed_pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def _pick_exemplars(trials: List[Dict]) -> Dict[str, Dict]:
    """One illustrative trial per category, preferring probed+recovered ones."""
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for t in trials:
        by_cat[t["category"]].append(t)

    chosen: Dict[str, Dict] = {}
    for cat, sub in by_cat.items():
        recovered = [t for t in sub if t.get("phase_c_probe_and_recover") and
                     t["phase_c_probe_and_recover"].get("recovered")]
        probed = [t for t in sub if t.get("phase_c_probe_and_recover")]
        chosen[cat] = (recovered[0] if recovered
                       else probed[0] if probed
                       else sub[0])
    return chosen


def _exemplar_block(cat: str, t: Dict) -> str:
    a = t["phase_a_closed_book"]
    pc = t.get("phase_c_probe_and_recover")
    parts = [f"### Exemplar — `{cat}` (`{t['dataset']}`)",
             "",
             f"**Seed:** {t['seed_question']}",
             "",
             f"**Gold ({t['gold']['kind']}):** {t['gold']['answer']}",
             "",
             "**Phase A — Closed-book QA (no retrieval)**",
             "",
             f"> {a['answer']}",
             "",
             f"- semantic entropy (normalised): `{a['normalized_entropy']:.3f}` "
             f"across `{a['n_clusters']}` cluster(s) of {len(a['samples'])} samples",
             f"- judge correctness: **{a['correctness']}**",
             "",
             ]
    pg = t.get("policy_gate", {})
    parts.append(f"**Policy gate:** {pg.get('reason', '')}")
    parts.append("")
    if pc:
        parts.append("**Phase C — Multi-agent probe → RAG recovery**")
        parts.append("")
        parts.append(f"- Generator question: _{pc['generator_question']}_")
        parts.append(f"- Critic refinement:  _{pc['critic_question']}_")
        parts.append(f"- Challenger probe:   _{pc['challenger_question']}_")
        parts.append("")
        parts.append("**RAG-augmented answer (retrieval driven by probe terms)**")
        parts.append("")
        parts.append(f"> {pc['rag_answer']}")
        parts.append("")
        parts.append(f"- judge correctness: **{pc['rag_correctness']}**")
        parts.append(f"- Δuncertainty (post − pre): `{pc['delta_uncertainty']:+.3f}`")
        parts.append(f"- pre/post contradiction: `{pc['contradiction_pre_vs_post']}`")
        parts.append(f"- **Recovered:** `{pc['recovered']}`")
    else:
        parts.append("_Policy gate did not fire — closed-book answer was already a pass under the gold-kind-aware rule._")
    parts.append("")
    return "\n".join(parts)


def build_report(json_path: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    meta = payload["experiment_metadata"]
    metrics = payload["evaluation_metrics"]
    trials = payload["trials"]

    fig_dir = _figures_dir_for(json_path)
    audit_path = _audit_for(json_path)
    tsv_path = _tsv_for(json_path)
    ts = json_path.stem.replace("research_results_", "")

    out_path = json_path.parent / f"RESEARCH_REPORT_{ts}.md"
    rel_fig = lambda name: (
        f"figures_{ts}/{name}" if fig_dir and (fig_dir / name).exists() else None
    )

    lines: List[str] = []
    lines.append("# Stress-Testing Crime QA — Research Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_Model: `{meta['model']}` · entropy samples per estimate: {meta['n_samples_for_entropy']} · RAG top-k: {meta['rag_top_k']}_")
    lines.append("")

    # ------------ 1. abstract ------------
    lines.append("## 1. Abstract")
    lines.append("")
    lines.append(
        "We stress-test a closed-book LLM on US crime-statistics QA across two "
        "real-world datasets (Phoenix open-incident records 2015-2025; FBI "
        "homicide records 1980-2014) by routing every wrong-or-hedged single-shot "
        "answer through a **Generator → Critic → Challenger** probing pipeline "
        "whose final probe drives a retrieval-augmented re-answer. We measure "
        "(a) the accuracy lift between closed-book and probing-augmented answers, "
        "(b) the recovery rate on initially-failed trials, (c) the change in "
        "semantic entropy across the two phases, and (d) per-trial pre/post "
        "contradictions. The pipeline writes per-trial QA / probe / recovery "
        "traces and a fully reproducible audit log to `outputs/research/`. "
        "Below: aggregate metrics over "
        f"**{metrics['n_trials']}** seeds (with **{metrics['n_probed']}** "
        "passing through Phase C) plus six figures."
    )
    lines.append("")

    # ------------ 2. system architecture ------------
    lines.append("## 2. System architecture")
    lines.append("")
    lines.append("```")
    lines.append("                       ┌───────────────────────────┐")
    lines.append("                       │   Phase A: closed-book QA │")
    lines.append("seed question  ───────►│   answer + N samples      │")
    lines.append("                       │   semantic entropy        │")
    lines.append("                       │   judge vs gold           │")
    lines.append("                       └─────────────┬─────────────┘")
    lines.append("                                     │")
    lines.append("                                     ▼")
    lines.append("                       ┌───────────────────────────┐")
    lines.append("                       │  Phase B: policy gate     │")
    lines.append("                       │  ambiguous: hedge = pass  │")
    lines.append("                       │  factual:  hedge = gap    │")
    lines.append("                       └─┬───────────────────┬─────┘")
    lines.append("                  pass ◄─┘                   └─► fail")
    lines.append("                                                     │")
    lines.append("                                                     ▼")
    lines.append("                            ┌────────────────────────────────┐")
    lines.append("                            │  Phase C: multi-agent probe    │")
    lines.append("                            │  Generator → Critic → Challenger│")
    lines.append("                            │  → drives RAG retrieval        │")
    lines.append("                            │  → re-answer + N samples + judge│")
    lines.append("                            └────────────────┬───────────────┘")
    lines.append("                                             │")
    lines.append("                                             ▼")
    lines.append("                            ┌────────────────────────────────┐")
    lines.append("                            │  Phase D: scoring              │")
    lines.append("                            │  recovered, Δentropy,          │")
    lines.append("                            │  contradiction-pre-vs-post     │")
    lines.append("                            └────────────────────────────────┘")
    lines.append("```")
    lines.append("")
    lines.append("**Why a gold-kind-aware policy gate?** A closed-book LLM that says "
                 "*\"I do not have access to that record\"* on a single-incident factual "
                 "question is honest but useless: it is exactly the recoverable gap that "
                 "the RAG path is meant to fix. We therefore treat `hedged_appropriate` "
                 "as a **pass** for ambiguous questions (where hedging is the correct "
                 "answer) but as a **fail** for factual / statistical / comparative "
                 "questions (where hedging means the model declined to retrieve).")
    lines.append("")

    # ------------ 3. uncertainty estimator ------------
    lines.append("## 3. Uncertainty estimator")
    lines.append("")
    lines.append("Semantic entropy follows the LM-Polygraph `SemanticEntropy` recipe (Fadeeva "
                 "et al. EMNLP 2023; Kuhn et al. ICLR 2023): sample N answers at T=0.7, "
                 "cluster by semantic equivalence, take Shannon entropy over cluster sizes, "
                 "normalise by `log2(N)`. We replace the upstream NLI cluster-judge with a "
                 "single Gemini call returning a JSON cluster assignment — cheaper but "
                 "conceptually identical (`{\"are these the same claim?\"}` instead of "
                 "embedding cosine).")
    lines.append("")

    # ------------ 4. probing agents ------------
    lines.append("## 4. Probing agents")
    lines.append("")
    lines.append("Re-implements the agentic decomposition from ProbeLLM (Hwong et al.) "
                 "without importing the registry plumbing:")
    lines.append("")
    lines.append("- **Generator** — proposes one follow-up that targets where the sampled "
                 "answers disagree the most.")
    lines.append("- **Critic** — rewrites for specificity, falsifiability, and "
                 "domain-grounding.")
    lines.append("- **Challenger** — adds an adversarial twist (timeframe constraint, "
                 "category exclusion, forced-numerical estimate) that exposes hidden "
                 "fragility.")
    lines.append("")
    lines.append("The Challenger output is the *final probe*. It is used as the **retrieval "
                 "query** that drives the RAG-augmented re-answer; its text is **not** "
                 "shown to the answering LLM (the adversarial framing biases the LLM into "
                 "premature refusal).")
    lines.append("")

    # ------------ 5. headline metrics ------------
    lines.append("## 5. Headline results")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| trials (total) | **{metrics['n_trials']}** |")
    lines.append(f"| trials probed (gate fired) | **{metrics['n_probed']}** |")
    lines.append(f"| trials recovered by probing | **{metrics['n_recovered']}** |")
    lines.append(f"| closed-book accuracy | **{_pct(metrics['closed_book_accuracy_overall'])}** |")
    lines.append(f"| probing-augmented accuracy | **{_pct(metrics['augmented_accuracy_overall'])}** |")
    lines.append(f"| accuracy lift | **{_signed_pct(metrics['accuracy_lift'])}** |")
    lines.append(f"| recovery rate (rescued / probed) | **{_pct(metrics['recovery_rate_overall'])}** |")
    lines.append(f"| mean Δentropy on probed trials | **{metrics['mean_delta_uncertainty_on_probed']:+.3f}** |")
    lines.append(f"| pre/post contradiction rate on probed trials | **{_pct(metrics['contradiction_rate_on_probed'])}** |")
    lines.append("")

    by_ds = metrics["by_dataset"]
    lines.append("### By dataset")
    lines.append("")
    rows = []
    for ds, m in by_ds.items():
        rows.append([
            ds, m["n"], m["n_probed"],
            _pct(m["closed_book_accuracy"]),
            _pct(m["augmented_accuracy"]),
            _pct(m["recovery_rate"]),
        ])
    lines.append(_markdown_table(
        rows, ["dataset", "n", "n_probed", "closed-book", "probing-augmented", "recovery"]
    ))
    lines.append("")

    by_cat = metrics["by_category"]
    lines.append("### By question category")
    lines.append("")
    rows = []
    for cat, m in by_cat.items():
        rows.append([
            cat, m["n"], m["n_probed"],
            _pct(m["closed_book_accuracy"]),
            _pct(m["augmented_accuracy"]),
            _pct(m["recovery_rate"]),
        ])
    lines.append(_markdown_table(
        rows, ["category", "n", "n_probed", "closed-book", "probing-augmented", "recovery"]
    ))
    lines.append("")

    # ------------ 6. figures ------------
    lines.append("## 6. Figures")
    lines.append("")
    figs = [
        ("fig01_accuracy_lift.png", "Closed-book vs probing-augmented accuracy (overall + by dataset)."),
        ("fig02_accuracy_by_category.png", "Accuracy split by question category."),
        ("fig03_recovery_rate.png", "Recovery rate per category — fraction of probed trials whose answer was rescued."),
        ("fig04_uncertainty_dist.png", "Distribution of normalised semantic entropy pre vs post probing."),
        ("fig05_uncertainty_delta.png", "Per-trial Δentropy after probing (sorted)."),
        ("fig06_correctness_flow.png", "Trial-outcome flow: passed closed-book / recovered by probing / still wrong."),
    ]
    for fname, caption in figs:
        rel = rel_fig(fname)
        if rel:
            lines.append(f"![{Path(fname).stem}]({rel})")
            lines.append("")
            lines.append(f"*Figure: {caption}*")
            lines.append("")

    # ------------ 7. exemplars ------------
    lines.append("## 7. Per-category exemplar traces")
    lines.append("")
    lines.append("One trial per category, preferring trials where the policy gate fired "
                 "and probing actually recovered the correct answer (so the full closed-book "
                 "→ probe → RAG recovery trajectory is visible).")
    lines.append("")
    chosen = _pick_exemplars(trials)
    for cat in sorted(chosen.keys()):
        lines.append(_exemplar_block(cat, chosen[cat]))

    # ------------ 8. raw artefacts ------------
    lines.append("## 8. Raw artefacts")
    lines.append("")
    lines.append(f"- structured JSON results: `{json_path.name}`")
    if audit_path:
        lines.append(f"- per-trial audit log:    `{audit_path.name}` (every closed-book → probe → RAG trace)")
    if tsv_path:
        lines.append(f"- spreadsheet TSV:        `{tsv_path.name}`")
    if fig_dir:
        lines.append(f"- raw figures:            `{fig_dir.name}/`")
    lines.append("")
    lines.append("All artefacts are checkpointed after every trial, so the run is "
                 "recoverable from a partial completion.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated research report")
    parser.add_argument("--json", type=Path, default=None,
                        help="Path to research_results_<ts>.json (default: latest)")
    args = parser.parse_args()

    json_path = args.json or _latest_results_json()
    out = build_report(json_path)
    print(f"[report] consolidated report written to: {out}")


if __name__ == "__main__":
    main()
