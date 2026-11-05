"""Accesso ai modelli Gemini con cache su disco, rate limiting e retry.

Perche' serve:
  * CACHE: ogni risposta viene salvata (cache/<modello>.jsonl). Rilanciare uno
    stesso esperimento non consuma quota, e gli esperimenti sono ripetibili
    anche se il modello ignora la temperatura.
  * RATE LIMIT: il free tier ha pochi richieste/minuto; il programma si
    impone un intervallo minimo tra due chiamate (GEMINI_RPM nel .env).
  * RETRY: gli errori temporanei (429 al minuto, 5xx) vengono ritentati con
    attesa crescente. L'esaurimento della quota GIORNALIERA invece non si
    risolve aspettando: si solleva DailyQuotaExceeded e si riprende il giorno dopo
    (i progressi sono salvati nella cache e nei file di risultati).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

from config import CACHE_DIR, GEMINI_RPM
from src.schemas import as_text

try:  # httpx e' una dipendenza della libreria Google: se manca, si ignora
    import httpx

    _NETWORK_ERRORS: tuple = (httpx.TransportError, ConnectionError, TimeoutError)
except ImportError:  # pragma: no cover
    _NETWORK_ERRORS = (ConnectionError, TimeoutError)


class DailyQuotaExceeded(RuntimeError):
    """La quota giornaliera del free tier e' esaurita: riprovare domani."""


def _is_daily_quota(msg: str) -> bool:
    m = msg.lower()
    return "perday" in m.replace(" ", "").replace("_", "") or "per day" in m or "daily" in m


def _is_retryable(msg: str, exc: BaseException | None = None) -> bool:
    """Errori temporanei: quota al minuto, errori del server, problemi di rete.

    Gli errori di rete (connessione interrotta, timeout) sono tipici quando il
    computer va in stop o il Wi-Fi cade a meta' di una richiesta.
    """
    e = exc
    while e is not None:  # controlla anche la catena delle cause
        if isinstance(e, _NETWORK_ERRORS):
            return True
        e = e.__cause__
    m = msg.lower()
    markers = ("429", "resource_exhausted", "rate limit", "500", "502", "503", "504",
               "unavailable", "deadline", "timeout", "timed out", "overloaded", "internal",
               "connection", "reset by peer", "server disconnected", "network")
    return any(k in m for k in markers)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class CachedLLM:
    """Wrapper minimale: `complete(prompt) -> str`, con cache, rate limit e retry."""

    def __init__(
        self,
        model_name: str,
        rpm: float = GEMINI_RPM,
        cache_dir: Path = CACHE_DIR,
        client=None,
        max_attempts: int = 6,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        if not model_name:
            raise ValueError("Nome del modello vuoto: controlla il file .env")
        self.model_name = model_name
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.max_attempts = max_attempts
        self._client = client  # se None viene creato al primo uso
        self._sleep = sleep
        self._clock = clock
        self._last_call = None

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_dir / f"{_safe_name(model_name)}.jsonl"
        self._cache: dict[str, str] = {}
        if self._cache_path.exists():
            with self._cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["key"]] = row["response"]

        self.calls = 0       # chiamate reali all'API
        self.cache_hits = 0  # risposte servite dalla cache

    # -- interni ---------------------------------------------------------
    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{prompt}".encode("utf-8")).hexdigest()

    def _get_client(self):
        if self._client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # max_retries=1: i retry li gestiamo noi, altrimenti la libreria
            # ne aggiungerebbe di nascosti consumando altra quota.
            self._client = ChatGoogleGenerativeAI(model=self.model_name, max_retries=1, timeout=120)
        return self._client

    def _respect_rate_limit(self) -> None:
        if self._last_call is not None:
            wait = self.min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)

    # -- API pubblica ----------------------------------------------------
    def complete(self, prompt: str) -> str:
        key = self._key(prompt)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        for attempt in range(self.max_attempts):
            self._respect_rate_limit()
            try:
                self._last_call = self._clock()
                self.calls += 1
                response = as_text(self._get_client().invoke(prompt).content)
                break
            except Exception as e:  # noqa: BLE001 - classifichiamo noi l'errore
                msg = str(e)
                if _is_daily_quota(msg):
                    raise DailyQuotaExceeded(
                        f"Quota giornaliera esaurita per {self.model_name}. "
                        f"Rilancia lo stesso comando domani: riprende da dove si e' fermato.\n"
                        f"Dettaglio: {msg[:300]}"
                    ) from e
                if not _is_retryable(msg, e) or attempt == self.max_attempts - 1:
                    raise
                delay = min(90.0, 5.0 * 2**attempt) + random.uniform(0, 2)
                print(f"  [retry {attempt + 1}/{self.max_attempts - 1}] errore temporaneo, attendo {delay:.0f}s: {msg[:120]}")
                self._sleep(delay)

        self._cache[key] = response
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        return response
