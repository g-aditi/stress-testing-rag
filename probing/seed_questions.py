"""Crime-domain seed questions used to start each probing trace.

Twelve questions across five categories, all grounded in fields available in
the Phoenix open crime dataset:
  factual:        single-fact lookup (offense type / location / date)
  statistical:    requires aggregation across the dataset
  comparative:    forces a direct comparison between subgroups
  ambiguous:      under-specified on purpose (timeframe / scope / category)
  edge_case:      narrow ZIP / neighborhood / very recent slice

The dataset has columns: INC NUMBER, OCCURRED ON, UCR CRIME CATEGORY,
100 BLOCK ADDR, ZIP, PREMISE TYPE, GRID — so every question can in principle
be answered or refused on factual grounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SeedQuestion:
    text: str
    category: str


SEED_QUESTIONS: List[SeedQuestion] = [
    SeedQuestion("Is crime increasing in Phoenix?", "ambiguous"),
    SeedQuestion("Did violent crime increase in Phoenix after 2020?", "ambiguous"),
    SeedQuestion("What is the most common offense reported in Phoenix?", "statistical"),
    SeedQuestion("How many burglary incidents were reported in Phoenix in 2022?", "statistical"),
    SeedQuestion("Are burglaries more common in Phoenix apartments or single-family houses?", "comparative"),
    SeedQuestion("Which Phoenix ZIP code has the highest larceny-theft count?", "comparative"),
    SeedQuestion("Has theft gone down recently in the 85006 ZIP code?", "edge_case"),
    SeedQuestion("Is auto theft increasing in Phoenix's 85015 ZIP code?", "edge_case"),
    SeedQuestion("What offense was reported at 13XX E ALMERIA RD on November 1, 2015?", "factual"),
    SeedQuestion("How many drug offenses were reported in Phoenix between 2018 and 2020?", "statistical"),
    SeedQuestion("Are property crimes more common during weekends in Phoenix?", "ambiguous"),
    SeedQuestion("Which premise type sees the most aggravated assault incidents in Phoenix?", "comparative"),
]


def as_text_list() -> List[str]:
    return [s.text for s in SEED_QUESTIONS]
