"""Applica la regola "stesso farmaco => none" a una run gia' fatta, senza chiamare l'LLM.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.apply_identity --run baseline_s2_dev_sample

Confronta tre versioni delle STESSE predizioni:
  - originale   : le predizioni del modello cosi' come sono
  - +string     : se i due nomi sono uguali (a meno di maiuscole/spazi) -> none
  - +rxnorm     : come sopra, e in piu' se RxNorm li riconduce allo stesso principio attivo -> none

La regola e' deterministica e non dipende dal modello, quindi applicarla dopo equivale ad
applicarla prima della chiamata (nell'agente sara' un nodo che evita anche la chiamata).
Scrive results/<config>+string_s..., results/<config>+rxnorm_s... (predizioni e metriche) e
stampa un confronto con l'elenco delle predizioni cambiate, per controllare che non si
rompa nulla.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PROCESSED_DIR, RESULTS_DIR
from eval.analyze_errors import read_jsonl, split_of
from eval.metrics import compute_metrics
from src.tools.rxnorm import RxNormClient, same_drug


def variant_name(run: str, variant: str) -> str:
    config, rest = run.split("_s", 1)
    return f"{config}+{variant}_s{rest}"


def apply_rule(preds: list[dict], data: dict[str, dict], client: RxNormClient | None) -> list[dict]:
    """Restituisce nuove predizioni dove la regola ha cambiato l'etichetta in `none`."""
    out = []
    for p in preds:
        rec = data[p["pair_id"]]
        same, why = same_drug(rec["e1"]["text"], rec["e2"]["text"], client)
        q = dict(p)
        if same and p["pred"] != "none":
            q.update(orig_pred=p["pred"], pred="none", rule=why)
        out.append(q)
    return out


def summarize(label: str, rows: list[dict]) -> dict:
    m = compute_metrics([r["gold"] for r in rows], [r["pred"] for r in rows])
    fp_none = sum(1 for r in rows if r["gold"] == "none" and r["pred"] != "none")
    n_none = sum(1 for r in rows if r["gold"] == "none")
    m["false_positives_on_none"] = {"count": fp_none, "of": n_none}
    return m


def main(argv: list[str] | None = None, client: RxNormClient | None = None,
         results_dir: Path = RESULTS_DIR, processed_dir: Path = PROCESSED_DIR) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)

    split = split_of(args.run)
    path = results_dir / f"{args.run}.jsonl"
    if not path.exists():
        raise SystemExit(f"File {path.name} non trovato in {results_dir}")
    preds = read_jsonl(path)
    data = {r["pair_id"]: r for r in read_jsonl(processed_dir / f"{split}.jsonl")}

    client = client or RxNormClient()
    versions = {
        "originale": preds,
        "string": apply_rule(preds, data, None),       # solo confronto di stringhe
        "rxnorm": apply_rule(preds, data, client),     # stringhe + RxNorm
    }

    results = {}
    for name, rows in versions.items():
        results[name] = summarize(name, rows)
        if name != "originale":
            out_run = variant_name(args.run, name)
            with (results_dir / f"{out_run}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            (results_dir / f"{out_run}_metrics.json").write_text(
                json.dumps(results[name], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Run: {args.run}  ({len(preds)} coppie)\n")
    print(f"{'versione':<12}{'detP':>7}{'detR':>7}{'detF1':>7}{'microP':>8}{'microR':>8}{'microF1':>9}{'macroF1':>9}   FP su none")
    for name, m in results.items():
        d, mi, ma, fp = m["detection"], m["classification_micro"], m["classification_macro"], m["false_positives_on_none"]
        print(f"{name:<12}{d['precision']:>7.3f}{d['recall']:>7.3f}{d['f1']:>7.3f}"
              f"{mi['precision']:>8.3f}{mi['recall']:>8.3f}{mi['f1']:>9.3f}{ma['f1']:>9.3f}   {fp['count']}/{fp['of']}")

    changed = [(o, n) for o, n in zip(preds, versions["rxnorm"]) if "rule" in n]
    print(f"\nPredizioni cambiate dalla regola (stringhe + RxNorm): {len(changed)}")
    fixed = harmed = neutral = 0
    for o, n in changed:
        rec = data[o["pair_id"]]
        if o["gold"] == "none":
            verdict, fixed = "CORRETTA ora", fixed + 1
        elif o["gold"] == o["pred"]:
            verdict, harmed = "ROVINATA (era giusta)", harmed + 1
        else:
            verdict, neutral = "era gia' sbagliata", neutral + 1
        print(f"  [{n['rule']}] {rec['e1']['text']!r} ~ {rec['e2']['text']!r}: "
              f"{o['pred']} -> none (gold: {o['gold']})  {verdict}  [{o['pair_id']}]")
    print(f"\nRiepilogo: corrette {fixed}, rovinate {harmed}, gia' sbagliate {neutral}")
    return results


if __name__ == "__main__":
    main()
