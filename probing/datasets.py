"""Dataset abstraction over the two crime CSVs.

Each dataset exposes:
  - load() -> pandas DataFrame (with parsed date / year columns)
  - schema   -> field-name map
  - retrieve(question, top_k) -> DataFrame of relevant rows (lexical scoring)
  - aggregate_stats(question)  -> short pre-computed summary string for
                                  statistical questions, or None
  - format_context(rows)       -> readable context block for an LLM
  - name, description          -> metadata for reports

Two concrete datasets:
  PhoenixCrime         — incident-level, 2015-2025, generic offense types
  USHomicide           — FBI homicide records, 1980-onward, demographics + weapon
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PHOENIX_CSV = DATA_DIR / "phoenix_crime_data.csv"
US_HOMICIDE_CSV = DATA_DIR / "US_Crime_DataSet.csv"


@dataclass
class CrimeDataset:
    name: str
    description: str
    csv_path: Path
    df: pd.DataFrame = field(repr=False)
    text_index: pd.Series = field(repr=False)

    def retrieve(self, question: str, top_k: int = 3) -> pd.DataFrame:
        raise NotImplementedError

    def aggregate_stats(self, question: str) -> Optional[str]:
        raise NotImplementedError

    def format_context(self, rows: pd.DataFrame) -> str:
        raise NotImplementedError


# --- Phoenix --------------------------------------------------------------

class PhoenixCrime(CrimeDataset):
    @classmethod
    def load(cls) -> "PhoenixCrime":
        df = pd.read_csv(PHOENIX_CSV, low_memory=False)
        df["_parsed_date"] = pd.to_datetime(df.get("OCCURRED ON"), errors="coerce")
        df["_year"] = df["_parsed_date"].dt.year
        cols = ["UCR CRIME CATEGORY", "100 BLOCK ADDR", "PREMISE TYPE", "ZIP"]
        text = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        return cls(
            name="phoenix",
            description="Phoenix, AZ open crime incidents (2015-2025), incident-level.",
            csv_path=PHOENIX_CSV,
            df=df,
            text_index=text,
        )

    def retrieve(self, question: str, top_k: int = 3) -> pd.DataFrame:
        q_lower = question.lower()
        q_tokens = set(re.findall(r"[a-z0-9]+", q_lower)) - {
            "the", "a", "an", "is", "are", "of", "in", "on", "at", "for", "to", "and",
            "what", "which", "how", "many", "in", "phoenix",
        }
        scores = self.text_index.apply(
            lambda t: sum(1 for tok in q_tokens if tok in t)
        )
        # Strongly weight any rare address-like token (street names, "19xx" block
        # prefixes) hitting the 100 BLOCK ADDR column directly. Without this the
        # lexical scorer dilutes a perfect address match against generic words.
        addr_lower = self.df["100 BLOCK ADDR"].fillna("").astype(str).str.lower()
        for tok in q_tokens:
            if len(tok) >= 4:
                scores = scores + addr_lower.str.contains(re.escape(tok), regex=True).astype(int) * 4
        m = re.search(r"\b(19|20)\d{2}\b", question)
        if m:
            scores = scores + (self.df["_year"] == int(m.group(0))).astype(int) * 5
        # Specific MM/DD/YYYY in the seed: boost rows whose parsed date matches
        # the exact day. Otherwise multiple incidents on the same year+street
        # tie and we may miss the right row.
        d = re.search(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b", question)
        if d:
            try:
                target = pd.Timestamp(year=int(d.group(3)), month=int(d.group(1)), day=int(d.group(2)))
                same_day = (self.df["_parsed_date"].dt.normalize() == target.normalize()).fillna(False)
                scores = scores + same_day.astype(int) * 12
            except (ValueError, TypeError):
                pass
        z = re.search(r"\b\d{5}\b", question)
        if z:
            try:
                zip_val = float(z.group(0))
                scores = scores + (self.df["ZIP"] == zip_val).astype(int) * 8
            except (ValueError, TypeError):
                pass
        return self.df.assign(_score=scores).sort_values("_score", ascending=False).head(top_k)

    def aggregate_stats(self, question: str) -> Optional[str]:
        q = question.lower()
        triggers = ["how many", "count", "total", "trend", "increas", "decreas", "rate",
                    "common", "most", "top", "highest", "lowest"]
        if not any(k in q for k in triggers):
            return None
        df = self.df
        lines = []
        years = sorted([y for y in df["_year"].dropna().unique() if 2010 <= y <= 2025])
        per_year = df.groupby("_year").size().reindex(years).fillna(0).astype(int)
        lines.append("Total Phoenix incidents per year:")
        for y in years[-10:]:
            lines.append(f"  {int(y)}: {int(per_year.get(y, 0))}")
        if "UCR CRIME CATEGORY" in df.columns:
            top_offenses = df["UCR CRIME CATEGORY"].value_counts().head(8)
            lines.append("Top offense categories overall:")
            for offense, count in top_offenses.items():
                lines.append(f"  {offense}: {count}")
        m = re.search(r"\b(19|20)\d{2}\b", question)
        if m:
            year = int(m.group(0))
            year_df = df[df["_year"] == year]
            if len(year_df) > 0 and "UCR CRIME CATEGORY" in df.columns:
                lines.append(f"Top offenses in {year}:")
                for offense, count in year_df["UCR CRIME CATEGORY"].value_counts().head(8).items():
                    lines.append(f"  {offense}: {count}")
        return "\n".join(lines)

    def format_context(self, rows: pd.DataFrame) -> str:
        blocks: List[str] = []
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            blocks.append(
                f"Record {i}: "
                f"INC={row.get('INC NUMBER', '')}; "
                f"OCCURRED={row.get('OCCURRED ON', '')}; "
                f"OFFENSE={row.get('UCR CRIME CATEGORY', '')}; "
                f"ADDR={row.get('100 BLOCK ADDR', '')}; "
                f"ZIP={row.get('ZIP', '')}; "
                f"PREMISE={row.get('PREMISE TYPE', '')}"
            )
        return "\n".join(blocks)


# --- US Homicide ----------------------------------------------------------

class USHomicide(CrimeDataset):
    @classmethod
    def load(cls) -> "USHomicide":
        df = pd.read_csv(US_HOMICIDE_CSV, low_memory=False)
        df["_year"] = pd.to_numeric(df["Year"], errors="coerce")
        cols = ["State", "Crime Type", "Weapon", "Relationship", "City", "Agency Name", "Year"]
        text = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        return cls(
            name="us_homicide",
            description="FBI homicide records, US-wide, 1980 onward; demographics + weapon + relationship.",
            csv_path=US_HOMICIDE_CSV,
            df=df,
            text_index=text,
        )

    def retrieve(self, question: str, top_k: int = 3) -> pd.DataFrame:
        q_lower = question.lower()
        q_tokens = set(re.findall(r"[a-z0-9]+", q_lower)) - {
            "the", "a", "an", "is", "are", "of", "in", "on", "at", "for", "to", "and",
            "what", "which", "how", "many", "between", "year", "state", "homicide",
            "homicides", "records", "record",
        }
        scores = self.text_index.apply(
            lambda t: sum(1 for tok in q_tokens if tok in t)
        )
        state_lower = self.df["State"].fillna("").astype(str).str.lower()
        for tok in q_tokens:
            if len(tok) >= 4:
                scores = scores + (state_lower == tok).astype(int) * 6
                scores = scores + state_lower.str.contains(re.escape(tok), regex=True).astype(int) * 2
        m = re.search(r"\b(19|20)\d{2}\b", question)
        if m:
            scores = scores + (self.df["_year"] == int(m.group(0))).astype(int) * 5
        return self.df.assign(_score=scores).sort_values("_score", ascending=False).head(top_k)

    def aggregate_stats(self, question: str) -> Optional[str]:
        q = question.lower()
        triggers = ["how many", "count", "total", "trend", "increas", "decreas", "rate",
                    "common", "most", "top", "highest", "lowest", "percent"]
        if not any(k in q for k in triggers):
            return None
        df = self.df
        lines = []

        m = re.search(r"\b(19|20)\d{2}\b", question)
        if m:
            year = int(m.group(0))
            yr = df[df["_year"] == year]
            lines.append(f"Year {year}: total homicide records = {len(yr)}")
            if len(yr) > 0:
                top_states = yr["State"].value_counts().head(8)
                lines.append(f"Top states by homicide count in {year}:")
                for state, count in top_states.items():
                    lines.append(f"  {state}: {count}")
                if "Weapon" in yr.columns:
                    top_weap = yr["Weapon"].value_counts().head(8)
                    lines.append(f"Top weapons in {year}:")
                    for w, c in top_weap.items():
                        lines.append(f"  {w}: {c}")
                if "Crime Solved" in yr.columns:
                    solved_rate = (yr["Crime Solved"] == "Yes").mean()
                    lines.append(f"Solved rate {year}: {solved_rate*100:.1f}%")
        else:
            top_states = df["State"].value_counts().head(8)
            lines.append("Top states by total homicide records:")
            for state, count in top_states.items():
                lines.append(f"  {state}: {count}")
            top_weap = df["Weapon"].value_counts().head(8)
            lines.append("Top weapons overall:")
            for w, c in top_weap.items():
                lines.append(f"  {w}: {c}")
        return "\n".join(lines)

    def format_context(self, rows: pd.DataFrame) -> str:
        blocks: List[str] = []
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            blocks.append(
                f"Record {i}: "
                f"State={row.get('State', '')}; "
                f"City={row.get('City', '')}; "
                f"Year={row.get('Year', '')}; "
                f"CrimeType={row.get('Crime Type', '')}; "
                f"Weapon={row.get('Weapon', '')}; "
                f"Relationship={row.get('Relationship', '')}; "
                f"Solved={row.get('Crime Solved', '')}"
            )
        return "\n".join(blocks)


def load_all() -> Dict[str, CrimeDataset]:
    return {
        "phoenix": PhoenixCrime.load(),
        "us_homicide": USHomicide.load(),
    }
