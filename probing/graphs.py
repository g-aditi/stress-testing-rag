"""Matplotlib figures for the research report.

Each function takes the list of TrialRecord and a target directory, writes
one PNG, and returns the path. `render_all_figures` runs every figure and
returns the list of paths it produced.

Figures:
  fig01_accuracy_lift           closed-book vs probing-augmented accuracy (overall + by dataset)
  fig02_accuracy_by_category    same but split by question category
  fig03_recovery_rate           recovery rate per category and per dataset
  fig04_uncertainty_dist        entropy histogram pre vs post (probed trials only)
  fig05_uncertainty_delta       Δuncertainty per probed trial (sorted)
  fig06_correctness_flow        sankey-ish bar: closed_correct / probed / recovered / unrecovered
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from .run_research import is_pass


def _accuracy(records, judgement: str) -> float:
    if not records:
        return 0.0
    n_pass = 0
    for r in records:
        if judgement == "closed":
            if is_pass(r.closed_correctness, r.gold_kind):
                n_pass += 1
        elif judgement == "augmented":
            final = r.rag_correctness if r.probed else r.closed_correctness
            if is_pass(final, r.gold_kind):
                n_pass += 1
    return n_pass / len(records)


def fig_accuracy_lift(records, out_dir: Path) -> Path:
    datasets = sorted({r.dataset for r in records})
    labels = ["overall"] + datasets
    closed_vals = [_accuracy(records, "closed")] + [
        _accuracy([r for r in records if r.dataset == d], "closed") for d in datasets
    ]
    aug_vals = [_accuracy(records, "augmented")] + [
        _accuracy([r for r in records if r.dataset == d], "augmented") for d in datasets
    ]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [v * 100 for v in closed_vals], w, label="closed-book", color="#9aa0a6")
    ax.bar(x + w / 2, [v * 100 for v in aug_vals], w, label="probing-augmented", color="#1a73e8")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Closed-book vs probing-augmented accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    for i, (cv, av) in enumerate(zip(closed_vals, aug_vals)):
        ax.text(i - w / 2, cv * 100 + 1.5, f"{cv*100:.0f}%", ha="center", fontsize=9)
        ax.text(i + w / 2, av * 100 + 1.5, f"{av*100:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    out = out_dir / "fig01_accuracy_lift.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_accuracy_by_category(records, out_dir: Path) -> Path:
    cats = sorted({r.category for r in records})
    closed_vals = [_accuracy([r for r in records if r.category == c], "closed") for c in cats]
    aug_vals = [_accuracy([r for r in records if r.category == c], "augmented") for c in cats]
    x = np.arange(len(cats))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [v * 100 for v in closed_vals], w, label="closed-book", color="#9aa0a6")
    ax.bar(x + w / 2, [v * 100 for v in aug_vals], w, label="probing-augmented", color="#1a73e8")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy by question category")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "fig02_accuracy_by_category.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_recovery_rate(records, out_dir: Path) -> Path:
    cats = sorted({r.category for r in records})
    rates = []
    counts = []
    for c in cats:
        sub = [r for r in records if r.category == c and r.probed]
        if sub:
            rates.append(sum(1 for r in sub if r.recovered) / len(sub) * 100)
        else:
            rates.append(0.0)
        counts.append(len(sub))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cats, rates, color="#0f9d58")
    ax.set_ylabel("recovery rate (%)")
    ax.set_title("Recovery rate by category (rescued / probed)")
    ax.set_ylim(0, 100)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.0f}% (n={n})", ha="center", fontsize=9)
    fig.tight_layout()
    out = out_dir / "fig03_recovery_rate.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_uncertainty_dist(records, out_dir: Path) -> Path:
    probed = [r for r in records if r.probed]
    if not probed:
        return out_dir / "fig04_uncertainty_dist.png"
    pre = [r.closed_normalized_entropy for r in probed]
    post = [r.rag_normalized_entropy for r in probed]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 11)
    ax.hist(pre, bins=bins, alpha=0.6, label="pre (closed-book)", color="#9aa0a6")
    ax.hist(post, bins=bins, alpha=0.6, label="post (probing-augmented)", color="#1a73e8")
    ax.set_xlabel("normalised semantic entropy")
    ax.set_ylabel("count of probed trials")
    ax.set_title("Distribution of uncertainty pre vs post probing")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "fig04_uncertainty_dist.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_uncertainty_delta(records, out_dir: Path) -> Path:
    probed = sorted([r for r in records if r.probed], key=lambda r: r.delta_uncertainty)
    deltas = [r.delta_uncertainty for r in probed]
    colors = ["#0f9d58" if d <= 0 else "#d93025" for d in deltas]
    x = np.arange(len(probed))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Δ normalised entropy (post − pre)")
    ax.set_xlabel("probed trial (sorted)")
    ax.set_title("Per-trial Δuncertainty after probing")
    fig.tight_layout()
    out = out_dir / "fig05_uncertainty_delta.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fig_correctness_flow(records, out_dir: Path) -> Path:
    """Stacked bar showing the journey of all trials."""
    n = len(records)
    closed_pass = [r for r in records if is_pass(r.closed_correctness, r.gold_kind)]
    closed_fail = [r for r in records if not is_pass(r.closed_correctness, r.gold_kind)]
    probed = [r for r in closed_fail if r.probed]
    recovered = [r for r in probed if r.recovered]
    unrecovered = [r for r in probed if not r.recovered]

    cats = ["passed closed-book", "recovered by probing", "still wrong after probing"]
    vals = [len(closed_pass), len(recovered), len(unrecovered)]
    colors = ["#9aa0a6", "#0f9d58", "#d93025"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats, vals, color=colors)
    ax.set_ylabel("number of trials")
    ax.set_title(f"Trial outcomes (n = {n})")
    for bar, v in zip(bars, vals):
        pct = (v / n * 100) if n else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v} ({pct:.0f}%)", ha="center", fontsize=10)
    fig.tight_layout()
    out = out_dir / "fig06_correctness_flow.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def render_all_figures(records, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if not records:
        return paths
    paths.append(fig_accuracy_lift(records, out_dir))
    paths.append(fig_accuracy_by_category(records, out_dir))
    paths.append(fig_recovery_rate(records, out_dir))
    paths.append(fig_uncertainty_dist(records, out_dir))
    paths.append(fig_uncertainty_delta(records, out_dir))
    paths.append(fig_correctness_flow(records, out_dir))
    return paths
