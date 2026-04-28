"""Programmatic seed generators with auto-derived ground truth.

For factual / statistical / comparative questions we can compute the gold
answer from the dataset directly, which lets the correctness judge be
deterministic. We also include a small bank of hand-curated AMBIGUOUS seeds
where the gold is "appropriate hedging" rather than a specific fact.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .datasets import PhoenixCrime, USHomicide


@dataclass
class Seed:
    text: str
    category: str
    dataset: str
    gold_answer: str
    gold_kind: str = "fact"
    metadata: Dict = field(default_factory=dict)


# --------------------------- Phoenix seeds --------------------------------

def _phoenix_factual(ds: PhoenixCrime, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    cand = df[
        df["UCR CRIME CATEGORY"].notna()
        & df["100 BLOCK ADDR"].notna()
        & df["OCCURRED ON"].notna()
    ]
    cand = cand[~cand["UCR CRIME CATEGORY"].astype(str).str.lower().str.contains("homicide|murder", na=False)]
    rows = cand.sample(n=n, random_state=rng.randint(0, 10_000))
    seeds = []
    for _, row in rows.iterrows():
        offense = str(row["UCR CRIME CATEGORY"]).strip()
        addr = str(row["100 BLOCK ADDR"]).strip()
        date_text = str(row["OCCURRED ON"]).split()[0]
        seeds.append(Seed(
            text=f"What offense was reported at {addr} in Phoenix on {date_text}?",
            category="factual",
            dataset="phoenix",
            gold_answer=offense,
            gold_kind="exact_offense",
        ))
    return seeds


def _phoenix_statistical(ds: PhoenixCrime, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    seeds = []
    years = sorted([int(y) for y in df["_year"].dropna().unique() if 2016 <= y <= 2024])
    offenses = df["UCR CRIME CATEGORY"].value_counts().head(8).index.tolist()
    pairs = [(o, y) for o in offenses for y in years]
    rng.shuffle(pairs)
    for offense, year in pairs[:n]:
        count = int(((df["UCR CRIME CATEGORY"] == offense) & (df["_year"] == year)).sum())
        seeds.append(Seed(
            text=f"How many {offense.lower()} incidents were reported in Phoenix in {year}?",
            category="statistical",
            dataset="phoenix",
            gold_answer=str(count),
            gold_kind="integer_count",
            metadata={"offense": offense, "year": year, "exact_count": count},
        ))
    return seeds


def _phoenix_comparative(ds: PhoenixCrime, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    seeds = []
    if "PREMISE TYPE" not in df.columns or "UCR CRIME CATEGORY" not in df.columns:
        return seeds
    premises = df["PREMISE TYPE"].value_counts().head(6).index.tolist()
    offenses = df["UCR CRIME CATEGORY"].value_counts().head(6).index.tolist()
    combos = []
    for offense in offenses:
        sub = df[df["UCR CRIME CATEGORY"] == offense]
        if len(sub) == 0:
            continue
        prem_counts = sub["PREMISE TYPE"].value_counts()
        if len(prem_counts) >= 2:
            top_prem = prem_counts.index[0]
            second_prem = prem_counts.index[1]
            combos.append((offense, top_prem, second_prem,
                           int(prem_counts.iloc[0]), int(prem_counts.iloc[1])))
    rng.shuffle(combos)
    for offense, p1, p2, c1, c2 in combos[:n]:
        seeds.append(Seed(
            text=f"In Phoenix, are {offense.lower()} incidents more common at {p1.lower()} or {p2.lower()}?",
            category="comparative",
            dataset="phoenix",
            gold_answer=p1,
            gold_kind="comparison_winner",
            metadata={"offense": offense, "p1": p1, "p2": p2, "c1": c1, "c2": c2},
        ))
    return seeds


def _phoenix_zip(ds: PhoenixCrime, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    if "ZIP" not in df.columns or "UCR CRIME CATEGORY" not in df.columns:
        return []
    seeds = []
    offenses = df["UCR CRIME CATEGORY"].value_counts().head(5).index.tolist()
    for offense in offenses[:n]:
        sub = df[df["UCR CRIME CATEGORY"] == offense]
        zip_counts = sub["ZIP"].value_counts()
        if len(zip_counts) == 0:
            continue
        top_zip = zip_counts.index[0]
        seeds.append(Seed(
            text=f"Which Phoenix ZIP code had the highest number of {offense.lower()} incidents overall?",
            category="comparative",
            dataset="phoenix",
            gold_answer=str(int(top_zip)) if not isinstance(top_zip, str) else str(top_zip),
            gold_kind="zip_code",
            metadata={"offense": offense, "top_count": int(zip_counts.iloc[0])},
        ))
    return seeds


# --------------------------- US Homicide seeds ----------------------------

def _ushom_temporal(ds: USHomicide, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    states = df["State"].value_counts().head(15).index.tolist()
    years = sorted([y for y in df["_year"].dropna().unique() if 1990 <= y <= 2014])
    pairs = [(s, int(y)) for s in states for y in years]
    rng.shuffle(pairs)
    seeds = []
    for state, year in pairs[:n]:
        count = int(((df["State"] == state) & (df["_year"] == year)).sum())
        seeds.append(Seed(
            text=f"How many homicide records were reported in {state} in {year}?",
            category="statistical",
            dataset="us_homicide",
            gold_answer=str(count),
            gold_kind="integer_count",
            metadata={"state": state, "year": year, "exact_count": count},
        ))
    return seeds


def _ushom_weapon(ds: USHomicide, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    states = df["State"].value_counts().head(10).index.tolist()
    years_full = sorted([int(y) for y in df["_year"].dropna().unique() if 1990 <= y <= 2014])
    decades = [(y, y + 9) for y in range(1990, 2010, 10)]
    pairs = [(s, d) for s in states for d in decades]
    rng.shuffle(pairs)
    seeds = []
    for state, (start, end) in pairs[:n]:
        sub = df[(df["State"] == state) & (df["_year"].between(start, end))]
        if len(sub) == 0:
            continue
        weap_counts = sub["Weapon"].value_counts()
        if len(weap_counts) == 0:
            continue
        top_weap = weap_counts.index[0]
        seeds.append(Seed(
            text=f"What was the most common weapon used in homicide records from {state} between {start} and {end}?",
            category="comparative",
            dataset="us_homicide",
            gold_answer=str(top_weap),
            gold_kind="exact_weapon",
            metadata={"state": state, "decade": f"{start}-{end}", "count": int(weap_counts.iloc[0])},
        ))
    return seeds


def _ushom_solved(ds: USHomicide, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    states = df["State"].value_counts().head(10).index.tolist()
    years = sorted([int(y) for y in df["_year"].dropna().unique() if 1995 <= y <= 2014])
    pairs = [(s, y) for s in states for y in years]
    rng.shuffle(pairs)
    seeds = []
    for state, year in pairs[:n]:
        sub = df[(df["State"] == state) & (df["_year"] == year)]
        if len(sub) < 10:
            continue
        solved_rate = (sub["Crime Solved"] == "Yes").mean()
        seeds.append(Seed(
            text=f"What percentage of homicide records in {state} for the year {year} were marked as solved?",
            category="statistical",
            dataset="us_homicide",
            gold_answer=f"{solved_rate*100:.1f}%",
            gold_kind="percentage",
            metadata={"state": state, "year": year, "solved_rate": float(solved_rate)},
        ))
    return seeds


def _ushom_relationship(ds: USHomicide, n: int, rng: random.Random) -> List[Seed]:
    df = ds.df
    states = df["State"].value_counts().head(8).index.tolist()
    seeds = []
    for state in states[:n]:
        sub = df[df["State"] == state]
        if len(sub) < 100:
            continue
        rel_counts = sub["Relationship"].value_counts()
        # pick most common known relationship (skip unknown)
        rel_counts = rel_counts[~rel_counts.index.astype(str).str.lower().str.contains("unknown", na=False)]
        if len(rel_counts) == 0:
            continue
        top_rel = rel_counts.index[0]
        seeds.append(Seed(
            text=f"In {state}, what is the most common known victim-perpetrator relationship in the homicide records?",
            category="comparative",
            dataset="us_homicide",
            gold_answer=str(top_rel),
            gold_kind="exact_relationship",
            metadata={"state": state, "count": int(rel_counts.iloc[0])},
        ))
    return seeds


# --------------------------- Hand-curated ambiguous -----------------------

AMBIGUOUS_PHOENIX: List[Seed] = [
    Seed(
        text="Is crime increasing in Phoenix?",
        category="ambiguous",
        dataset="phoenix",
        gold_answer="Appropriately hedged answer noting timeframe / category dependency. Per dataset, total incidents 2019-2024 trended down (68838 -> 52803).",
        gold_kind="hedged",
    ),
    Seed(
        text="Is Phoenix becoming a more dangerous city?",
        category="ambiguous",
        dataset="phoenix",
        gold_answer="Appropriately hedged: 'dangerous' depends on metric (violent vs property), per-capita vs raw, and timeframe. Total incidents trended down 2019-2024.",
        gold_kind="hedged",
    ),
]

AMBIGUOUS_USHOM: List[Seed] = [
    Seed(
        text="Are homicides going up in the United States?",
        category="ambiguous",
        dataset="us_homicide",
        gold_answer="Appropriately hedged: dataset only covers up to mid-2010s; trends depend on timeframe and state.",
        gold_kind="hedged",
    ),
    Seed(
        text="Which state in the US has the most murders?",
        category="ambiguous",
        dataset="us_homicide",
        gold_answer="Appropriately hedged: depends on raw total vs per-capita rate. By raw count in this dataset, California is highest.",
        gold_kind="hedged",
    ),
]


# --------------------------- Public API -----------------------------------

def generate_phoenix_seeds(ds: PhoenixCrime, rng: random.Random) -> List[Seed]:
    seeds: List[Seed] = []
    seeds += _phoenix_factual(ds, n=8, rng=rng)
    seeds += _phoenix_statistical(ds, n=8, rng=rng)
    seeds += _phoenix_comparative(ds, n=6, rng=rng)
    seeds += _phoenix_zip(ds, n=4, rng=rng)
    seeds += AMBIGUOUS_PHOENIX
    return seeds


def generate_ushom_seeds(ds: USHomicide, rng: random.Random) -> List[Seed]:
    seeds: List[Seed] = []
    seeds += _ushom_temporal(ds, n=10, rng=rng)
    seeds += _ushom_weapon(ds, n=8, rng=rng)
    seeds += _ushom_solved(ds, n=6, rng=rng)
    seeds += _ushom_relationship(ds, n=4, rng=rng)
    seeds += AMBIGUOUS_USHOM
    return seeds


def generate_all_seeds(seed: int = 42) -> List[Seed]:
    rng = random.Random(seed)
    phoenix = PhoenixCrime.load()
    ushom = USHomicide.load()
    return generate_phoenix_seeds(phoenix, rng) + generate_ushom_seeds(ushom, rng)
