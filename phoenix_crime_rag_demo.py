#!/usr/bin/env python3
"""
One-file demo: stress-test an LLM on local Phoenix crime retrieval.

Flow:
1) Load/download official City of Phoenix crime CSV.
2) Inspect columns dynamically and select one specific incident row.
3) Generate one natural-language question from that row.
4) Ask OpenAI model once WITHOUT retrieval context.
5) Retrieve relevant dataset row(s), ask same question WITH context.
6) Print results and save a screenshot-friendly markdown report.

Requirements:
- Python 3.9+
- pandas
- openai
- OPENAI_API_KEY in environment
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from openai import OpenAI

CSV_URL = (
    "https://www.phoenixopendata.com/dataset/cc08aace-9ca9-467f-b6c1-f0879ab1a358/"
    "resource/0ce3411a-2fc6-4302-a33f-167f68608a20/download/"
    "crime-data_crime-data_crimestat.csv"
)

# Default requested by user
MODEL_NAME = "gpt-4o-mini"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOCAL_CSV = DATA_DIR / "phoenix_crime_data.csv"


def normalize_col(name: str) -> str:
    """Normalize column names to simplify flexible matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column by normalized candidate aliases."""
    normalized_map = {normalize_col(c): c for c in columns}
    for candidate in candidates:
        key = normalize_col(candidate)
        if key in normalized_map:
            return normalized_map[key]

    # Fallback: substring match for robustness when names differ slightly.
    for candidate in candidates:
        candidate_key = normalize_col(candidate)
        for norm, original in normalized_map.items():
            if candidate_key in norm or norm in candidate_key:
                return original
    return None


def to_text(value) -> str:
    """Human-friendly text conversion for dataset values."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def load_dataset() -> pd.DataFrame:
    """Load local cached CSV if present, else download from official source."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
        print(f"Loaded cached dataset: {LOCAL_CSV}")
        return df

    print("Downloading official Phoenix crime dataset...")
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as first_error:
        print(f"Direct CSV read failed ({first_error}). Retrying with HTTP headers...")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
            "Referer": "https://www.phoenixopendata.com/",
        }
        response = requests.get(CSV_URL, headers=headers, timeout=60)
        response.raise_for_status()
        LOCAL_CSV.write_bytes(response.content)
        df = pd.read_csv(LOCAL_CSV)

    df.to_csv(LOCAL_CSV, index=False)
    print(f"Saved dataset cache to: {LOCAL_CSV}")
    return df


def resolve_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Resolve key semantic fields even if exact column names vary."""
    cols = df.columns.tolist()

    schema = {
        "offense": find_col(
            cols,
            [
                "Offense", "Offense Category", "Crime Type", "UCR Crime Category",
                "UCR_CRM_CD_DESC", "Crime", "Incident Type",
            ],
        ),
        "date": find_col(
            cols,
            [
                "Occurrence Date", "Occurred Date", "Report Date", "Date",
                "From Date", "OccurDate", "OCCURRED_ON_DATE",
            ],
        ),
        "block_address": find_col(
            cols,
            [
                "100 Block Address", "Address", "Block Address", "Street Address",
                "LOCATION", "Address Number and Street",
            ],
        ),
        "zip": find_col(cols, ["Zip", "Zip Code", "ZIP_CODE", "Postal Code"]),
        "premise": find_col(cols, ["Premise Type", "Premise", "Location Type"]),
        "city": find_col(cols, ["City", "CITY", "Municipality"]),
        "incident_id": find_col(cols, ["Incident Number", "Incident ID", "DR Number", "ID"]),
    }

    return schema


def choose_incident(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> pd.Series:
    """
    Pick one candidate incident that is:
    - local/specific
    - not too old
    - likely not famous
    - suitable for a factual question
    """
    offense_col = schema["offense"]
    date_col = schema["date"]
    addr_col = schema["block_address"]
    city_col = schema["city"]

    if not offense_col or not date_col or not addr_col:
        raise ValueError(
            "Could not resolve required columns for offense/date/address. "
            "Please inspect the dataset columns and update aliases."
        )

    work = df.copy()

    work["_parsed_date"] = pd.to_datetime(work[date_col], errors="coerce")

    # Keep incidents from approximately recent years to make demo realistic.
    now = pd.Timestamp.now(tz=None)
    cutoff_years = [5, 8, 12]

    # Basic quality filters first.
    base = work[
        work[offense_col].notna()
        & work[addr_col].notna()
        & work["_parsed_date"].notna()
    ].copy()

    # If there is a city column, keep Phoenix rows.
    if city_col:
        city_series = base[city_col].astype(str).str.lower()
        phoenix_rows = base[city_series.str.contains("phoenix", na=False)]
        if len(phoenix_rows) > 0:
            base = phoenix_rows

    # Avoid likely high-profile/famous terms.
    avoid_terms = [
        "homicide", "murder", "officer", "mass", "celebrity",
        "kidnap", "terror", "arson", "manslaughter",
    ]
    offense_text = base[offense_col].astype(str).str.lower()
    mask_not_famous = ~offense_text.str.contains("|".join(avoid_terms), na=False)
    base = base[mask_not_famous].copy()

    # Apply recency progressively; fallback if strict filter gets empty.
    picked = pd.DataFrame()
    for years in cutoff_years:
        threshold = now - pd.Timedelta(days=365 * years)
        candidate = base[base["_parsed_date"] >= threshold].copy()
        if len(candidate) >= 1:
            picked = candidate
            break

    if picked.empty:
        picked = base

    if picked.empty:
        raise ValueError("No suitable incident rows found after filtering.")

    # Rank for specificity: prefer addresses with more detail + valid zip/premise.
    zip_col = schema["zip"]
    premise_col = schema["premise"]

    specificity_score = picked[addr_col].astype(str).str.len().fillna(0)

    if zip_col:
        specificity_score = specificity_score + picked[zip_col].notna().astype(int) * 10
    if premise_col:
        specificity_score = specificity_score + picked[premise_col].notna().astype(int) * 8

    picked = picked.assign(_specificity=specificity_score)
    picked = picked.sort_values(["_specificity", "_parsed_date"], ascending=[False, False])

    # Deterministic pick among top few to keep reproducible but still “automatic”.
    top_n = min(20, len(picked))
    top = picked.head(top_n)
    selected = top.sample(n=1, random_state=42).iloc[0]
    return selected


def format_date_for_question(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return to_text(value)
    return dt.strftime("%B %d, %Y")


def build_question(row: pd.Series, schema: Dict[str, Optional[str]]) -> str:
    """Generate one natural language question from selected incident row."""
    offense = to_text(row[schema["offense"]]) if schema["offense"] else "an incident"
    date_text = (
        format_date_for_question(row[schema["date"]])
        if schema["date"]
        else "that date"
    )
    address = to_text(row[schema["block_address"]]) if schema["block_address"] else "that location"

    return (
        f"What offense was reported near {address} in Phoenix on {date_text}, "
        f"and what key location details are available?"
    )


def retrieve_relevant_rows(
    df: pd.DataFrame,
    schema: Dict[str, Optional[str]],
    selected_row: pd.Series,
    question: str,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Lightweight retrieval (no vector DB):
    score rows by overlap with question + exact matches on key fields.
    """
    work = df.copy()

    date_col = schema["date"]
    offense_col = schema["offense"]
    addr_col = schema["block_address"]

    # Ensure date parsing for date match scoring.
    if date_col:
        work["_parsed_date"] = pd.to_datetime(work[date_col], errors="coerce")

    # Token overlap score against offense/address/premise text.
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))

    text_cols = [c for c in [offense_col, addr_col, schema.get("premise")] if c]
    combined_text = (
        work[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        if text_cols
        else pd.Series([""] * len(work), index=work.index)
    )

    def overlap_score(text: str) -> int:
        text_tokens = set(re.findall(r"[a-z0-9]+", text))
        return len(tokens.intersection(text_tokens))

    work["_score"] = combined_text.map(overlap_score)

    # Strong boosts for exact matches with selected row's core facts.
    if offense_col:
        selected_offense = to_text(selected_row[offense_col]).lower()
        work["_score"] += work[offense_col].astype(str).str.lower().eq(selected_offense).astype(int) * 50

    if addr_col:
        selected_addr = to_text(selected_row[addr_col]).lower()
        work["_score"] += work[addr_col].astype(str).str.lower().eq(selected_addr).astype(int) * 50

    if date_col:
        sel_date = pd.to_datetime(selected_row[date_col], errors="coerce")
        if not pd.isna(sel_date):
            work["_score"] += (
                pd.to_datetime(work[date_col], errors="coerce").dt.date.eq(sel_date.date()).astype(int) * 40
            )

    # Keep highest scoring rows; ensure selected incident is included.
    top = work.sort_values("_score", ascending=False).head(max(top_k, 1)).copy()

    # If selected row not in top, append it.
    if selected_row.name not in top.index:
        selected_df = work.loc[[selected_row.name]].copy()
        top = pd.concat([top, selected_df], axis=0)

    # Deduplicate while preserving order.
    top = top[~top.index.duplicated(keep="first")]
    return top.head(top_k)


def format_row_for_display(row: pd.Series, schema: Dict[str, Optional[str]]) -> str:
    """Readable row summary for printing and report."""
    fields = [
        ("Incident ID", schema.get("incident_id")),
        ("Offense", schema.get("offense")),
        ("Date", schema.get("date")),
        ("100-Block Address", schema.get("block_address")),
        ("ZIP", schema.get("zip")),
        ("Premise", schema.get("premise")),
        ("City", schema.get("city")),
    ]

    lines = []
    for label, col in fields:
        if col and col in row.index:
            lines.append(f"- {label} ({col}): {to_text(row[col])}")
    return "\n".join(lines)


def build_rag_context(rows: pd.DataFrame, schema: Dict[str, Optional[str]]) -> str:
    """Create readable retrieval context from selected rows."""
    context_blocks: List[str] = []

    fields = [
        ("Incident ID", schema.get("incident_id")),
        ("Offense", schema.get("offense")),
        ("Date", schema.get("date")),
        ("100-Block Address", schema.get("block_address")),
        ("ZIP", schema.get("zip")),
        ("Premise", schema.get("premise")),
        ("City", schema.get("city")),
    ]

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        lines = [f"Record {i}:"]
        for label, col in fields:
            if col and col in rows.columns:
                lines.append(f"  - {label}: {to_text(row[col])}")
        context_blocks.append("\n".join(lines))

    return "\n\n".join(context_blocks)


def call_llm_no_rag(client: OpenAI, question: str, model: str) -> str:
    """Call model with no external context."""
    system_prompt = (
        "You are a helpful assistant. Answer the user's question normally. "
        "If you are uncertain, say so clearly instead of guessing."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


def call_llm_with_rag(client: OpenAI, question: str, context: str, model: str) -> str:
    """Call model with retrieved context and grounding constraints."""
    system_prompt = (
        "You answer only from provided dataset context. "
        "Do not invent details that are not present. "
        "If context is insufficient, say what is missing. "
        "Cite which dataset fields support each key claim."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved Phoenix crime dataset context:\n{context}\n\n"
        "Answer using only the context above. Include short field citations "
        "like [Offense], [Date], [100-Block Address], [ZIP], [Premise]."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def classify_no_rag(answer: str) -> str:
    """Simple heuristic classification for demo summary."""
    text = answer.lower()

    uncertainty_markers = [
        "i'm not sure", "i am not sure", "uncertain", "cannot verify", "don't know",
        "do not know", "no specific", "without", "not available",
    ]
    if any(marker in text for marker in uncertainty_markers):
        return "appropriately uncertain"

    vague_markers = ["may", "might", "possibly", "generally", "typically", "often"]
    if any(marker in text for marker in vague_markers):
        return "vague"

    return "unsupported"


def classify_rag(answer: str) -> str:
    """Simple heuristic classification for grounded answer quality."""
    text = answer.lower()
    has_field_cites = sum(token in text for token in ["[offense]", "[date]", "[100-block address]", "[zip]", "[premise]"])

    if has_field_cites >= 2:
        return "more specific, better grounded, better sourced"

    if len(text) > 120:
        return "more specific and better grounded"

    return "somewhat improved but weakly sourced"


def save_markdown_report(
    selected_row_text: str,
    question: str,
    no_rag_answer: str,
    rag_answer: str,
    no_rag_label: str,
    rag_label: str,
    schema: Dict[str, Optional[str]],
) -> Path:
    """Save markdown report for screenshot-friendly presentation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"phoenix_rag_demo_result_{ts}.md"

    schema_lines = "\n".join([f"- {k}: {v}" for k, v in schema.items()])

    report = f"""# Phoenix Local Crime Retrieval Demo (One Incident)

## === SELECTED INCIDENT ===
{selected_row_text}

## === GENERATED QUESTION ===
{question}

## === ANSWER WITHOUT RAG ===
{no_rag_answer}

## === ANSWER WITH RAG ===
{rag_answer}

## === FINAL SUMMARY ===
- First answer assessment: **{no_rag_label}**
- RAG answer assessment: **{rag_label}**

## Resolved Dataset Schema
{schema_lines}
"""

    out_path.write_text(report, encoding="utf-8")
    return out_path


def print_sections(
    selected_row_text: str,
    question: str,
    no_rag_answer: str,
    rag_answer: str,
    no_rag_label: str,
    rag_label: str,
    report_path: Path,
) -> None:
    """Print required section headers and final summary."""
    print("\n=== SELECTED INCIDENT ===")
    print(selected_row_text)

    print("\n=== GENERATED QUESTION ===")
    print(question)

    print("\n=== ANSWER WITHOUT RAG ===")
    print(no_rag_answer)

    print("\n=== ANSWER WITH RAG ===")
    print(rag_answer)

    print("\n=== FINAL SUMMARY ===")
    print(f"No-RAG answer was: {no_rag_label}")
    print(f"RAG answer became: {rag_label}")
    print(f"Saved report: {report_path}")


def main() -> None:
    """Run full one-incident no-RAG vs RAG demo."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in your environment.")

    # Allow optional model override via env while defaulting to gpt-4o-mini.
    model_name = os.getenv("OPENAI_MODEL", MODEL_NAME)

    df = load_dataset()
    schema = resolve_schema(df)

    selected = choose_incident(df, schema)
    question = build_question(selected, schema)

    selected_text = format_row_for_display(selected, schema)

    client = OpenAI(api_key=api_key)

    no_rag_answer = call_llm_no_rag(client, question, model_name)

    retrieved = retrieve_relevant_rows(df, schema, selected, question, top_k=3)
    rag_context = build_rag_context(retrieved, schema)
    rag_answer = call_llm_with_rag(client, question, rag_context, model_name)

    no_rag_label = classify_no_rag(no_rag_answer)
    rag_label = classify_rag(rag_answer)

    report_path = save_markdown_report(
        selected_row_text=selected_text,
        question=question,
        no_rag_answer=no_rag_answer,
        rag_answer=rag_answer,
        no_rag_label=no_rag_label,
        rag_label=rag_label,
        schema=schema,
    )

    print_sections(
        selected_row_text=selected_text,
        question=question,
        no_rag_answer=no_rag_answer,
        rag_answer=rag_answer,
        no_rag_label=no_rag_label,
        rag_label=rag_label,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
