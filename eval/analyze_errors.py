"""Elenca gli errori di una run, raggruppati per tipo di confusione.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.analyze_errors --run baseline_s2_dev_sample
    python -m eval.analyze_errors --run baseline_s2_dev_sample --only none:mechanism
    python -m eval.analyze_errors --run baseline_s2_dev_sample --save

`--run` e' il nome del file in results/ senza estensione (es. baseline_s2_dev_sample).
Ogni errore e' mostrato come "gold -> predetto", con la frase in cui i due farmaci
sono marcati da <D1>..</D1> e <D2>..</D2>. Con --save il testo viene scritto anche in
results/errors_<run>.txt, utile come materiale per l'analisi critica del report.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from config import PROCESSED_DIR, RESULTS_DIR
from src.prompts import mark_entities

SPLITS = ("dev_sample", "test_sample", "test_confirm", "dev", "test")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of(run: str) -> str:
    for s in SPLITS:  # "dev_sample" prima di "dev"
        if run.endswith("_" + s):
            return s
    raise SystemExit(f"Non riesco a ricavare lo split dal nome {run!r}")


def build_report(run: str, only: str | None = None, results_dir: Path = RESULTS_DIR,
                 processed_dir: Path = PROCESSED_DIR) -> str:
    split = split_of(run)
    path = results_dir / f"{run}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in results_dir.glob("*.jsonl"))
        raise SystemExit(f"File {path.name} non trovato. Run disponibili: {available}")
    preds = read_jsonl(path)
    data = {r["pair_id"]: r for r in read_jsonl(processed_dir / f"{split}.jsonl")}

    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = collections.defaultdict(list)
    for p in preds:
        if p["gold"] != p["pred"]:
            groups[(p["gold"], p["pred"])].append((p, data[p["pair_id"]]))

    n_err = sum(len(v) for v in groups.values())
    lines = [f"Run: {run}   errori: {n_err} / {len(preds)}", ""]
    lines.append("Riepilogo (gold -> predetto : numero):")
    for (g, pr), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {g} -> {pr} : {len(items)}")

    for (g, pr), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if only and f"{g}:{pr}" != only:
            continue
        lines += ["", "=" * 78, f"{g} -> {pr}   ({len(items)} casi)", "=" * 78]
        for p, rec in items:
            lines.append(f"[{rec['pair_id']}]  ({rec['source']})")
            lines.append(f"  Farmaci: D1 = {rec['e1']['text']!r}   D2 = {rec['e2']['text']!r}")
            lines.append(f"  Frase:   {mark_entities(rec)}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", default=None, help="mostra solo un tipo di errore, es. none:mechanism")
    ap.add_argument("--save", action="store_true", help="salva anche in results/errors_<run>.txt")
    args = ap.parse_args()

    text = build_report(args.run, args.only)
    print(text)
    if args.save:
        out = RESULTS_DIR / f"errors_{args.run}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
