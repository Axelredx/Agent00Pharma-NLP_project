"""Schemi dei dati e parsing dell'output dei modelli.

Il modello viene istruito a rispondere con un oggetto JSON. In pratica i
modelli a volte aggiungono testo prima/dopo o racchiudono il JSON in un blocco
di codice: `extract_json` e' tollerante su questo. Una risposta da cui non si
riesce a ricavare un'etichetta valida NON viene scartata: restituisce
`INVALID_LABEL`, cosi' le metriche la contano come errore e la riportano.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

INVALID_LABEL = "INVALID"

Label = Literal["none", "effect", "mechanism", "advise", "int"]

# Varianti che un modello puo' scrivere al posto dell'etichetta esatta.
_SYNONYMS = {
    "advice": "advise",
    "no": "none",
    "no_interaction": "none",
    "no interaction": "none",
    "negative": "none",
    "interaction": "int",
}


class PairPrediction(BaseModel):
    """Output atteso dal modello per una coppia di farmaci."""

    label: Label
    rationale: str = ""

    @field_validator("label", mode="before")
    @classmethod
    def _normalize_label(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            v = _SYNONYMS.get(v, v)
        return v


def as_text(content) -> str:
    """Estrae il testo dalla risposta di LangChain.

    Alcuni modelli Gemini recenti restituiscono `content` come lista di blocchi
    (testo, firme di ragionamento...) invece che come stringa semplice.
    """
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type", "text") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)


def extract_json(text: str) -> dict | None:
    """Trova il primo oggetto JSON valido nel testo, oppure None."""
    text = text.strip()
    # Blocco di codice ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    # Primo '{' fino all'ultima '}'
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_prediction(text: str) -> dict:
    """Trasforma l'output grezzo del modello in {'label', 'rationale', 'parse_ok'}."""
    obj = extract_json(text)
    if obj is not None:
        try:
            pred = PairPrediction(**{k: obj[k] for k in ("label", "rationale") if k in obj})
            return {"label": pred.label, "rationale": pred.rationale, "parse_ok": True}
        except (ValidationError, TypeError):
            pass
    return {"label": INVALID_LABEL, "rationale": "", "parse_ok": False}


class JudgeVerdict(BaseModel):
    """Verdetto del judge su un'etichetta proposta dall'estrattore."""

    verdict: Literal["accept", "reject"]
    issue: str = "none"      # none | not_between_pair | no_interaction_stated | wrong_type
    critique: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            v = {"accepted": "accept", "approve": "accept", "approved": "accept",
                 "rejected": "reject", "refuse": "reject"}.get(v, v)
        return v


def parse_verdict(text: str) -> dict:
    """Output grezzo del judge -> {'verdict', 'issue', 'critique', 'parse_ok'}.

    Se l'output non e' leggibile il judge NON mette il veto: si restituisce "accept"
    con parse_ok=False (cosi' un guasto del judge non cancella predizioni, e il
    numero di casi del genere viene riportato).
    """
    obj = extract_json(text)
    if obj is not None:
        try:
            v = JudgeVerdict(**{k: obj[k] for k in ("verdict", "issue", "critique") if k in obj})
            return {"verdict": v.verdict, "issue": str(v.issue), "critique": str(v.critique), "parse_ok": True}
        except (ValidationError, TypeError):
            pass
    return {"verdict": "accept", "issue": "unparseable_judge_output", "critique": "", "parse_ok": False}
