# stress-testing-rag

A multi-agent stress-testing pipeline that measures whether **uncertainty-triggered
probing** can recover a closed-book LLM's wrong / hedged answers on US crime QA.
Two real datasets: Phoenix open incidents (2015-2025) and FBI homicide records
(1980-2014). Re-implements the architecture from the project working doc:
closed-book QA → gold-kind-aware policy gate → Generator → Critic → Challenger
probe → RAG-augmented re-answer → scoring (Δentropy, recovery, contradiction).

Backend: **Google Gemini** via the `google-genai` SDK.
Default model: **`gemini-3.1-flash-lite-preview`** (override with `GEMINI_MODEL=...`).
No embeddings. Semantic equivalence is decided by an LLM-judge clustering call,
which is closer in spirit to LM-Polygraph's NLI-based equivalence relation than
embedding cosine would be.

## Layout

```
stress-testing-rag/
├── data/                            # Crime CSVs (Phoenix + US homicide)
├── outputs/research/                # Research-grade run outputs (canonical)
├── outputs/probing/                 # Legacy single-method comparison runs
├── phoenix_crime_rag_demo.py        # Original single-incident no-RAG vs RAG demo
└── probing/
    ├── config.py                    # Loads .env, central settings
    ├── llm_client.py                # Gemini client wrapper (chat + JSON)
    ├── datasets.py                  # Phoenix + US-homicide loaders, retrieval, aggregate stats
    ├── qa_agent.py                  # Target QA system (closed-book + RAG modes)
    ├── uncertainty.py               # Semantic entropy via LLM-judge clustering
    ├── probing_agents.py            # Generator / Critic / Challenger + single-shot baseline
    ├── evaluation.py                # Correctness / contradiction / change LLM judges
    ├── seed_generator.py            # Programmatic seeds w/ ground-truth from each CSV
    ├── seed_questions.py            # Original hand-curated 12 seeds (legacy comparison runs)
    ├── graphs.py                    # 6 matplotlib figures
    ├── run_research.py              # Research orchestrator (gated multi-phase, checkpointed)
    ├── run_experiment.py            # Legacy single-shot vs GCC comparison runner
    └── build_final_report.py        # Builds the consolidated RESEARCH_REPORT_<ts>.md
```

## Setup

Put your Gemini key in `../.env` (already done):

```
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-3.1-flash-lite-preview
```

Install deps:

```
pip install -r requirements.txt
```

The Phoenix crime CSV is already in `data/`. If missing, run
`python3 phoenix_crime_rag_demo.py` once to download it.

## Run the research experiment (canonical)

```
# full run: ~58 programmatically generated seeds across both datasets
python3 -m probing.run_research --samples 5

# smoke test: 3 seeds, 3 samples per uncertainty estimate
python3 -m probing.run_research --limit 3 --samples 3

# only one dataset
python3 -m probing.run_research --phoenix-only
python3 -m probing.run_research --ushom-only

# After the run, build the consolidated report
python3 -m probing.build_final_report
```

Outputs (in `outputs/research/`, one set per timestamp):

| File | Purpose |
|------|---------|
| `research_results_<ts>.json` | Structured per-trial JSON with phase A / policy gate / phase C / scoring + `evaluation_metrics` block |
| `research_audit_<ts>.txt`    | Human-readable per-trial trace (closed-book → samples → entropy → probe stages → RAG answer → scoring) |
| `research_results_<ts>.tsv`  | Flat one-row-per-trial spreadsheet |
| `figures_<ts>/fig0X_*.png`   | 6 matplotlib figures (accuracy lift, by category, recovery rate, entropy distribution, Δentropy, outcome flow) |
| `RESEARCH_REPORT_<ts>.md`    | Built by `build_final_report` — single polished Markdown with abstract, architecture diagram, headline tables, embedded figures, and per-category exemplar traces |

Outputs are **checkpointed after every trial** so a failure midway still leaves usable partial results.

## Pipeline (per seed)

```
PHASE A — CLOSED-BOOK QA
  closed_answer    ← QA.closed_book(seed, T=0.2)
  closed_samples   ← QA.closed_book(seed, T=0.7) × N
  closed_unc       ← semantic entropy over LLM-judge clusters
  closed_judge     ← LLM-judge correctness vs gold

PHASE B — POLICY GATE  (gold-kind aware)
  ambiguous (hedged-gold):  pass = correct ∪ hedged_appropriate
  factual / statistical / comparative:  pass = correct only
                                       (hedge → recoverable gap → probe)

PHASE C — MULTI-AGENT PROBE → RECOVERY  (only if Phase A fails)
  generator_q  ← targets where samples disagree
  critic_q     ← improves specificity / falsifiability
  challenger_q ← adversarial twist (final probe)
  ctx          ← retrieve(seed + probe terms)
  rag_answer   ← QA.rag(seed | ctx, T=0.2)        # probe used for retrieval only
  rag_samples  ← QA.rag(seed | ctx, T=0.7) × N
  rag_unc      ← semantic entropy
  rag_judge    ← LLM-judge correctness vs gold

PHASE D — SCORING per trial
  recovered    = (closed failed) ∧ (rag passed)
  Δuncertainty = rag_unc − closed_unc
  contradiction = LLM-judge contradiction(closed_answer, rag_answer)
```

## Metrics (aggregate)

| Metric | Definition |
|--------|------------|
| closed-book accuracy             | fraction of trials passing the gold-kind-aware rule from Phase A only |
| probing-augmented accuracy       | fraction passing using the post-probe answer (or closed if not probed) |
| accuracy lift                    | augmented − closed-book |
| recovery rate                    | rescued probed trials / probed trials |
| mean Δentropy on probed trials   | avg(rag_unc − closed_unc); negative = probing reduces uncertainty |
| pre/post contradiction rate      | fraction of probed trials whose pre and post answers disagree |

## Relationship to LM-Polygraph and ProbeLLM

This codebase **re-implements** ideas from two reference projects rather than importing them:

- **LM-Polygraph** (Fadeeva et al., EMNLP 2023; github.com/IINemo/lm-polygraph) provides
  a library of uncertainty estimators including SemanticEntropy (Kuhn et al., ICLR 2023).
  Our `probing/uncertainty.py` follows the same procedure — sample N → group by semantic
  equivalence → Shannon entropy over cluster sizes — but substitutes the upstream
  NLI-based equivalence relation (DeBERTa-MNLI bidirectional entailment) with a single
  Gemini judge call returning a JSON cluster assignment. Conceptually identical metric;
  one model call per uncertainty estimate, no embeddings.

- **ProbeLLM** (Hwong et al.; github.com/HowieHwong/ProbeLLM) introduces a principled
  tool/agent registry for diagnosing LLM failures. Our `probing/probing_agents.py`
  follows the same agentic decomposition pattern (multi-stage probe construction)
  without the registry plumbing — only the Generator → Critic → Challenger stages from
  the working doc.

For a publication-grade evaluation, both libraries can be swapped in: replace
`estimate_uncertainty` with `lm_polygraph.estimators.SemanticEntropy(...)`, or register
the GCC stages as ProbeLLM tools. The current implementation produces identical
evaluation artefacts (per-probe entropy, cluster assignments, audit log) so that swap
is mechanical.

## Cost / safety

- All chat calls go through Gemini Flash Lite — extremely cheap at this volume.
- Full research run (58 seeds × 5 samples + probe + RAG + judge calls): ~800 LLM
  calls, finishes in 30-40 minutes, well under free-tier quota.
- The `.env` API key is plaintext — do not commit it. `.gitignore` already excludes
  `.env` and `outputs/`.

## Legacy single-shot vs GCC comparison

The original `probing/run_experiment.py` runner is kept for reference. It runs
both **single-shot baseline** and **Generator → Critic → Challenger** probing on
the 12 hand-curated seeds and writes side-by-side metrics to `outputs/probing/`.
This is the apples-to-apples comparison of probing strategies; the
`run_research.py` runner is the end-to-end research evaluation that subsumes it
with gold-kind-aware scoring on a much larger programmatic seed set.

```
python3 -m probing.run_experiment --limit 12 --samples 5
```
