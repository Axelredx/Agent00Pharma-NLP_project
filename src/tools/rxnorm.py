"""Client per l'API RxNorm (NLM) e verifica "stesso farmaco".

RxNorm e' una terminologia di nomi di farmaci: associa nomi commerciali,
generici, sali e principi attivi a codici (RxCUI). NON contiene interazioni.
Qui serve a capire se due menzioni nel testo si riferiscono allo stesso
principio attivo (es. "Coumadin" e "warfarin"): in quel caso la coppia non
puo' essere un'interazione tra due farmaci diversi.

Scelte di progetto:
  * CACHE su disco (cache/rxnorm.jsonl): ogni richiesta HTTP si fa una volta sola;
    uno studio rilanciato non rifa' chiamate e si puo' riprendere se si interrompe.
  * RATE LIMIT gentile (default 8 richieste/secondo; l'NLM chiede di restare sotto 20).
  * RETRY con attesa crescente su errori di rete e 429/5xx.
  * CONSERVATIVO: due nomi sono "lo stesso farmaco" solo se i loro INSIEMI di
    principi attivi sono UGUALI. Un'unione sbagliata trasformerebbe vere
    interazioni in "none", quindi nel dubbio non si unisce.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import requests

from config import CACHE_DIR

BASE_URL = "https://rxnav.nlm.nih.gov/REST"
INGREDIENT_TTYS = ("IN",)  # IN = principio attivo ("warfarin")


class RxNormError(RuntimeError):
    """Errore non recuperabile nel parlare con l'API RxNorm."""


class _Retryable(Exception):
    pass


def normalize_name(name: str) -> str:
    """Normalizzazione minima per il confronto di stringhe: minuscolo, spazi singoli."""
    return " ".join(name.casefold().split()).strip(" .,;:")


@dataclass
class Resolution:
    query: str
    rxcuis: list[str] = field(default_factory=list)
    ingredients: frozenset[str] = frozenset()     # RxCUI dei principi attivi
    ingredient_names: dict[str, str] = field(default_factory=dict)
    status: str = "not_found"                     # ok | not_found | no_ingredient


class RxNormClient:
    def __init__(
        self,
        cache_path: Path | None = None,
        rps: float = 8.0,
        session=None,
        max_attempts: int = 5,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self._session = session or requests.Session()
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self.max_attempts = max_attempts
        self._sleep, self._clock = sleep, clock
        self._last = None
        self.http_calls = 0
        self.cache_hits = 0

        self._cache_path = cache_path or (CACHE_DIR / "rxnorm.jsonl")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        if self._cache_path.exists():
            with self._cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["key"]] = row["data"]
        self._resolutions: dict[str, Resolution] = {}

    # -- HTTP con cache, rate limit e retry --------------------------------
    def _get(self, path: str, params: dict) -> dict:
        key = path + "?" + urlencode(sorted(params.items()))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        data: dict | None = None
        for attempt in range(self.max_attempts):
            if self._last is not None:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            try:
                self._last = self._clock()
                self.http_calls += 1
                r = self._session.get(BASE_URL + path, params=params, timeout=20,
                                      headers={"Accept": "application/json"})
                if r.status_code == 404:
                    data = {}  # risorsa inesistente: risposta valida ("niente")
                    break
                if r.status_code == 429 or r.status_code >= 500:
                    raise _Retryable(f"HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise RxNormError(f"HTTP {r.status_code} su {path} {params}")
                data = r.json()
                break
            except (requests.RequestException, ValueError, _Retryable) as e:
                if attempt == self.max_attempts - 1:
                    raise RxNormError(f"RxNorm non raggiungibile ({e}) su {path} {params}") from e
                self._sleep(min(30.0, 2.0 * 2**attempt))
        assert data is not None
        self._cache[key] = data
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "data": data}, ensure_ascii=False) + "\n")
        return data

    # -- Funzioni di alto livello ------------------------------------------
    def rxcuis_by_name(self, name: str) -> list[str]:
        d = self._get("/rxcui.json", {"name": name})
        return list((d.get("idGroup") or {}).get("rxnormId") or [])

    def term_type(self, rxcui: str) -> str | None:
        d = self._get(f"/rxcui/{rxcui}/properties.json", {})
        return (d.get("properties") or {}).get("tty")

    def ingredients_of(self, rxcui: str) -> dict[str, str]:
        """Principi attivi (IN) collegati a un RxCUI, come {rxcui: nome}."""
        if self.term_type(rxcui) in INGREDIENT_TTYS:
            d = self._get(f"/rxcui/{rxcui}/properties.json", {})
            return {rxcui: (d.get("properties") or {}).get("name", "")}
        d = self._get(f"/rxcui/{rxcui}/related.json", {"tty": "IN"})
        out: dict[str, str] = {}
        for grp in (d.get("relatedGroup") or {}).get("conceptGroup") or []:
            for c in grp.get("conceptProperties") or []:
                if c.get("tty") in INGREDIENT_TTYS and c.get("rxcui"):
                    out[c["rxcui"]] = c.get("name", "")
        return out

    def resolve(self, name: str) -> Resolution:
        key = normalize_name(name)
        if key in self._resolutions:
            return self._resolutions[key]
        res = Resolution(query=name)
        rxcuis = self.rxcuis_by_name(name.strip())
        if rxcuis:
            res.rxcuis = rxcuis
            names: dict[str, str] = {}
            for c in rxcuis:
                names.update(self.ingredients_of(c))
            res.ingredient_names = names
            res.ingredients = frozenset(names)
            res.status = "ok" if names else "no_ingredient"
        self._resolutions[key] = res
        return res


def same_drug(a: str, b: str, client: RxNormClient | None) -> tuple[bool, str]:
    """Le due menzioni indicano lo stesso farmaco?

    Restituisce (si/no, motivo):
      "string"      stessi caratteri a meno di maiuscole e spazi
      "rxnorm"      nomi diversi ma stesso insieme di principi attivi in RxNorm
      "different"   principi attivi diversi
      "unresolved"  almeno uno dei due non e' stato riconosciuto da RxNorm
    """
    if normalize_name(a) == normalize_name(b):
        return True, "string"
    if client is None:
        return False, "unresolved"
    ra, rb = client.resolve(a), client.resolve(b)
    if ra.status != "ok" or rb.status != "ok":
        return False, "unresolved"
    if ra.ingredients == rb.ingredients:
        return True, "rxnorm"
    return False, "different"
