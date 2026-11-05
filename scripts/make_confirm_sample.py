"""Crea il secondo campione di test, per la valutazione di CONFERMA.

Uso (dalla radice del progetto, con il venv attivo):

    python -m scripts.make_confirm_sample

Scrive data/processed/test_confirm.jsonl e report/test_confirm_pair_ids.txt (gli id delle coppie,
da mettere su git: il campione si rigenera comunque in modo deterministico dal seme).

Regole (da dichiarare nel report, scritte PRIMA di eseguire le configurazioni):
  * popolazione: le coppie del test ufficiale ("DDI Extraction task") NON presenti in test_sample;
  * per default si escludono tutte le coppie delle FRASI gia' usate in test_sample (--exclude
    sentences), cosi' il campione e' indipendente da quello usato per scegliere la configurazione
    (con --exclude pairs si escludono solo le coppie gia' usate: meno puro);
  * quote come in test_sample: 100 coppie senza interazione e fino a 40 per ciascuna classe
    positiva, nei limiti di quanto resta disponibile (la classe `int` e' quasi esaurita);
  * seme fisso (default 2026), ordine e scelta deterministici.
Le metriche riponderate (eval.reweight) tengono conto delle quote diverse con i pesi N_classe / n_classe.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from config import NEGATIVE_LABEL, POSITIVE_LABELS, PROCESSED_DIR


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_sample(test: list[dict], used: list[dict], per_positive: int, n_negative: int, seed: int,
                 exclude: str = "sentences") -> list[dict]:
    used_pairs = {r["pair_id"] for r in used}
    used_sents = {r["sentence_id"] for r in used}
    if exclude == "sentences":
        pool_all = [r for r in test if r["sentence_id"] not in used_sents]
    else:
        pool_all = [r for r in test if r["pair_id"] not in used_pairs]
    by_label = collections.defaultdict(list)
    for r in pool_all:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    sample = []
    for label in POSITIVE_LABELS:
        pool = sorted(by_label[label], key=lambda r: r["pair_id"])
        rng.shuffle(pool)
        sample.extend(pool[:per_positive])
    pool = sorted(by_label[NEGATIVE_LABEL], key=lambda r: r["pair_id"])
    rng.shuffle(pool)
    sample.extend(pool[:n_negative])
    rng.shuffle(sample)

    # controlli di indipendenza
    assert not ({r["pair_id"] for r in sample} & used_pairs), "coppie in comune con test_sample"
    if exclude == "sentences":
        assert not ({r["sentence_id"] for r in sample} & used_sents), "frasi in comune con test_sample"
    return sample


def main(argv: list[str] | None = None, processed_dir=PROCESSED_DIR, report_dir=None) -> list[dict]:
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-positive", type=int, default=40)
    ap.add_argument("--negative", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--exclude", choices=["sentences", "pairs"], default="sentences")
    args = ap.parse_args(argv)

    test = read_jsonl(processed_dir / "test.jsonl")
    used = read_jsonl(processed_dir / "test_sample.jsonl")
    sample = build_sample(test, used, args.per_positive, args.negative, args.seed, args.exclude)

    with (processed_dir / "test_confirm.jsonl").open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rdir = Path(report_dir) if report_dir else Path("report")
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "test_confirm_pair_ids.txt").write_text("\n".join(sorted(r["pair_id"] for r in sample)) + "\n")

    counts = collections.Counter(r["label"] for r in sample)
    print(f"Campione di conferma: {len(sample)} coppie, {len({r['sentence_id'] for r in sample})} frasi, "
          f"{len({r['doc_id'] for r in sample})} documenti (seme {args.seed}, esclusione per {args.exclude})")
    for label in [NEGATIVE_LABEL] + POSITIVE_LABELS:
        want = args.negative if label == NEGATIVE_LABEL else args.per_positive
        note = "" if counts[label] >= want else f"   <-- meno delle {want} richieste: classe quasi esaurita"
        print(f"  {label:<10}{counts[label]:>5}{note}")
    return sample


if __name__ == "__main__":
    main()
