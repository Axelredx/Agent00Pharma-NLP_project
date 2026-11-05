"""Mostra i verdetti del judge di una run, soprattutto quelli sbagliati rispetto al gold.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.show_judge --run judge_s2_dev_sample            # solo i verdetti sbagliati
    python -m eval.show_judge --run agent3_s2_dev_sample --all     # tutti i casi giudicati
    python -m eval.show_judge --run judge_s2_dev_sample --save     # scrive anche results/judge_<run>.txt

Si considera il PRIMO tentativo dell'estrattore e il primo verdetto del judge. Un verdetto e':
  - GIUSTO  se rifiuta un'etichetta sbagliata o accetta un'etichetta giusta
  - SBAGLIATO se rifiuta un'etichetta giusta (danno: una predizione corretta rischia di andare persa)
              o accetta un'etichetta sbagliata (occasione mancata)
Nessuna chiamata a modelli: legge solo i file in results/.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from config import PROCESSED_DIR, RESULTS_DIR
from eval.analyze_errors import read_jsonl, split_of
from src.prompts import mark_entities

KINDS = {
    ("giusta", "reject"): "SBAGLIATO: etichetta giusta RIFIUTATA",
    ("sbagliata", "accept"): "SBAGLIATO: etichetta sbagliata ACCETTATA",
    ("giusta", "accept"): "giusto: etichetta giusta accettata",
    ("sbagliata", "reject"): "giusto: etichetta sbagliata rifiutata",
}


def build_report(run: str, show_all: bool = False, results_dir: Path = RESULTS_DIR,
                 processed_dir: Path = PROCESSED_DIR) -> str:
    path = results_dir / f"{run}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in results_dir.glob("*.jsonl"))
        raise SystemExit(f"File {path.name} non trovato. Run disponibili: {available}")
    rows = read_jsonl(path)
    data = {r["pair_id"]: r for r in read_jsonl(processed_dir / f"{split_of(run)}.jsonl")}

    cases = []
    for r in rows:
        extracts = [t for t in r.get("trace", []) if t["step"] == "extract"]
        judges = [t for t in r.get("trace", []) if t["step"] == "judge"]
        if not extracts or not judges:
            continue
        proposed, j = extracts[0]["label"], judges[0]
        kind = KINDS[("giusta" if proposed == r["gold"] else "sbagliata", j["verdict"])]
        cases.append((kind, r, proposed, j))

    counts = collections.Counter(k for k, *_ in cases)
    lines = [f"Run: {run}   casi giudicati: {len(cases)}", ""]
    for k in KINDS.values():
        lines.append(f"  {k}: {counts.get(k, 0)}")
    n_wrong = sum(v for k, v in counts.items() if k.startswith("SBAGLIATO"))
    lines.append(f"  -> verdetti sbagliati: {n_wrong} su {len(cases)}")

    for kind, r, proposed, j in cases:
        if not show_all and not kind.startswith("SBAGLIATO"):
            continue
        rec = data[r["pair_id"]]
        lines += ["", "-" * 78, f"{kind}   [{r['pair_id']}]",
                  f"  gold: {r['gold']}   proposta: {proposed}   verdetto: {j['verdict']}   issue: {j['issue']}",
                  f"  critica: {j['critique']}",
                  f"  Farmaci: D1 = {rec['e1']['text']!r}   D2 = {rec['e2']['text']!r}",
                  f"  Frase:   {mark_entities(rec)}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--all", action="store_true", help="mostra anche i verdetti giusti")
    ap.add_argument("--save", action="store_true", help="salva anche in results/judge_<run>.txt")
    args = ap.parse_args()
    text = build_report(args.run, args.all)
    print(text)
    if args.save:
        out = RESULTS_DIR / f"judge_{args.run}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
