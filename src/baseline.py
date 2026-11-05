"""Baseline: UNA sola chiamata al modello per ogni coppia di farmaci.

E' il punto di riferimento dell'ablation: stesso modello dell'agente, ma senza
verifica esterna, senza judge e senza loop. Tutto quello che l'architettura
agentica aggiunge si misura rispetto a questo.
"""
from __future__ import annotations

from typing import Callable

from src.llm import CachedLLM
from src.prompts import build_baseline_prompt
from src.schemas import parse_prediction


def build_baseline(llm: CachedLLM, fewshot: list[dict]) -> Callable[[dict], dict]:
    """Restituisce una funzione predict(record) -> {'pred', 'raw', 'parse_ok'}."""

    def predict(record: dict) -> dict:
        prompt = build_baseline_prompt(record, fewshot)
        raw = llm.complete(prompt)
        parsed = parse_prediction(raw)
        return {"pred": parsed["label"], "raw": raw, "parse_ok": parsed["parse_ok"]}

    return predict
