"""Prepara i dati in formato JSONL a partire dal DDI Corpus (XML).

Uso (dalla radice del progetto, con il venv attivo):

    python -m scripts.prepare_splits

Produce in data/processed/:
    train.jsonl        train ufficiale meno i documenti del dev (tutte le coppie)
    dev.jsonl          ~10% dei documenti del train, scelti A DOCUMENTO INTERO
    test.jsonl         test ufficiale "DDI Extraction task" (tutte le coppie)
    dev_sample.jsonl   campione stratificato del dev (per mettere a punto i prompt)
    test_sample.jsonl  campione stratificato del test (per la valutazione finale)
    summary.json       conteggi di tutto quanto sopra

Perche' il dev si sceglie per documento: frasi dello stesso documento sono
molto simili tra loro; se finissero in insiemi diversi (es. dev e pool di
esempi few-shot) si avrebbe "leakage", cioe' risultati gonfiati.

Il campione e' stratificato PER CLASSE (quote fisse) perche' circa l'83% delle
coppie non ha interazione: un campione casuale conterrebbe pochissimi esempi
positivi e le metriche per classe sarebbero instabili. Conseguenza da dichiarare
nel report: le metriche sul campione NON sono direttamente confrontabili con
quelle della letteratura, calcolate sul test completo.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from config import (
    NEGATIVE_LABEL,
    POSITIVE_LABELS,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_XML_DIR,
    TRAIN_XML_DIR,
)
from src.data_loader import load_split


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_counts(records: list[dict]) -> dict[str, int]:
    return dict(collections.Counter(r["label"] for r in records))


def split_dev_by_document(
    train: list[dict], dev_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Separa il train in (train, dev) assegnando interi documenti al dev."""
    doc_ids = sorted({r["doc_id"] for r in train})
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    n_dev_docs = max(1, round(len(doc_ids) * dev_fraction))
    dev_docs = set(doc_ids[:n_dev_docs])
    new_train = [r for r in train if r["doc_id"] not in dev_docs]
    dev = [r for r in train if r["doc_id"] in dev_docs]
    return new_train, dev


def stratified_sample(
    records: list[dict], per_positive: int, n_negative: int, seed: int
) -> list[dict]:
    """Campione con quota fissa per ciascuna classe positiva e per la negativa.

    Se una classe ha meno esempi della quota, vengono presi tutti.
    L'ordine finale e' mescolato (con seme fisso) per non avere blocchi per classe.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    sample: list[dict] = []
    for label in POSITIVE_LABELS:
        pool = sorted(by_label[label], key=lambda r: r["pair_id"])
        rng.shuffle(pool)
        sample.extend(pool[:per_positive])
    pool = sorted(by_label[NEGATIVE_LABEL], key=lambda r: r["pair_id"])
    rng.shuffle(pool)
    sample.extend(pool[:n_negative])

    rng.shuffle(sample)
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-fraction", type=float, default=0.10)
    ap.add_argument("--test-per-positive", type=int, default=40,
                    help="coppie per ciascuna classe positiva nel campione di test")
    ap.add_argument("--test-negative", type=int, default=100,
                    help="coppie senza interazione nel campione di test")
    ap.add_argument("--dev-per-positive", type=int, default=20)
    ap.add_argument("--dev-negative", type=int, default=40)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    train_all, skipped_train = load_split(TRAIN_XML_DIR)
    test, skipped_test = load_split(TEST_XML_DIR)
    if skipped_train or skipped_test:
        print(f"Coppie scartate (etichetta mancante): train={skipped_train} test={skipped_test}")

    train, dev = split_dev_by_document(train_all, args.dev_fraction, args.seed)

    # Controllo: nessun documento in comune tra train e dev
    assert not ({r["doc_id"] for r in train} & {r["doc_id"] for r in dev}), "leakage train/dev"

    dev_sample = stratified_sample(dev, args.dev_per_positive, args.dev_negative, args.seed)
    test_sample = stratified_sample(test, args.test_per_positive, args.test_negative, args.seed)

    outputs = {
        "train": train,
        "dev": dev,
        "test": test,
        "dev_sample": dev_sample,
        "test_sample": test_sample,
    }
    summary = {}
    for name, recs in outputs.items():
        write_jsonl(recs, PROCESSED_DIR / f"{name}.jsonl")
        summary[name] = {
            "pairs": len(recs),
            "documents": len({r["doc_id"] for r in recs}),
            "labels": label_counts(recs),
            "sources": dict(collections.Counter(r["source"] for r in recs)),
        }
    summary["_meta"] = {
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "skipped_pairs": skipped_train + skipped_test,
    }
    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{'insieme':<12}{'coppie':>8}{'doc':>6}   etichette")
    for name in outputs:
        s = summary[name]
        print(f"{name:<12}{s['pairs']:>8}{s['documents']:>6}   {s['labels']}")
    print(f"\nFile scritti in {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
