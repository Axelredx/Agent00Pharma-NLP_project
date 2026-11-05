"""Rivaluta una run del judge con politiche di veto diverse, SENZA nuove chiamate ai modelli.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.judge_policies --run judge_s2_dev_sample

Ogni riga della run salva la traccia (etichetta dell'estrattore, verdetto e `issue` del judge),
quindi si puo' ricalcolare cosa succederebbe se il judge avesse un potere di veto diverso:

  veto_all             ogni rifiuto cancella l'interazione (la politica usata nella run)
  veto_existence       solo i rifiuti che contestano l'ESISTENZA dell'interazione tra le due
                       entita' (not_between_pair, no_interaction_stated); un rifiuto per il TIPO
                       (wrong_type) non cancella nulla
  veto_not_between     solo not_between_pair
  no_veto              nessun veto = identity (riferimento)

Si usa solo il PRIMO tentativo dell'estrattore e il PRIMO verdetto del judge. Le politiche si
scelgono guardando il dev: va dichiarato nel report (e nessuna scelta va fatta sul test).

Per le run con loop (configurazione agent) stampa anche il confronto tra le due politiche di
FALLBACK sul risultato finale: coppie rimaste "uncertain" -> none (veto) oppure -> ultima
etichetta dell'estrattore (keep). Con keep, l'estrattore ha l'ultima parola sul judge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import RESULTS_DIR
from eval.analyze_errors import read_jsonl
from eval.metrics import compute_metrics

EXISTENCE_ISSUES = {"not_between_pair", "no_interaction_stated"}
POLICIES: dict[str, set[str] | None] = {   # None = qualunque rifiuto
    "veto_all": None,
    "veto_existence": EXISTENCE_ISSUES,
    "veto_not_between": {"not_between_pair"},
    "no_veto": set(),
}


def apply_policy(row: dict, veto_issues: set[str] | None) -> str:
    if row.get("status") == "identity":
        return "none"
    trace = row.get("trace", [])
    extracts = [t for t in trace if t["step"] == "extract"]
    if not extracts:
        return row["pred"]
    first = extracts[0]["label"]
    judges = [t for t in trace if t["step"] == "judge"]
    if first != "INVALID" and judges and judges[0]["verdict"] == "reject":
        if veto_issues is None or judges[0]["issue"] in veto_issues:
            return "none"
    return first


def _metrics(gold: list[str], pred: list[str]) -> dict:
    m = compute_metrics(gold, pred)
    m["fp_none"] = sum(1 for g, p in zip(gold, pred) if g == "none" and p != "none")
    m["lost_positives"] = sum(1 for g, p in zip(gold, pred) if g != "none" and p == "none")
    return m


def evaluate(rows: list[dict]) -> dict[str, dict]:
    gold = [r["gold"] for r in rows]
    return {name: _metrics(gold, [apply_policy(r, issues) for r in rows]) for name, issues in POLICIES.items()}


def evaluate_fallbacks(rows: list[dict]) -> dict[str, dict]:
    """Risultato FINALE della run con le due politiche per le coppie rimaste `uncertain`."""
    gold = [r["gold"] for r in rows]
    keep = [r["uncertain_label"] if r.get("status") == "uncertain" and r.get("uncertain_label") else r["pred"]
            for r in rows]
    return {
        "fallback_none": _metrics(gold, [r["pred"] for r in rows]),   # cosi' com'e' salvata nella run
        "fallback_keep": _metrics(gold, keep),
    }


def format_table(results: dict[str, dict], n_none: int, n_pos: int) -> str:
    lines = [f"{'politica':<18}{'detP':>7}{'detR':>7}{'detF1':>7}{'microF1':>9}{'macroF1':>9}"
             f"{'FP su none':>14}{'positivi persi':>16}"]
    for name, m in results.items():
        d = m["detection"]
        lines.append(f"{name:<18}{d['precision']:>7.3f}{d['recall']:>7.3f}{d['f1']:>7.3f}"
                     f"{m['classification_micro']['f1']:>9.3f}{m['classification_macro']['f1']:>9.3f}"
                     f"{str(m['fp_none']) + '/' + str(n_none):>14}{str(m['lost_positives']) + '/' + str(n_pos):>16}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, results_dir: Path = RESULTS_DIR) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    path = results_dir / f"{args.run}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in results_dir.glob("*.jsonl"))
        raise SystemExit(f"File {path.name} non trovato. Run disponibili: {available}")
    rows = read_jsonl(path)
    if not rows or "trace" not in rows[0]:
        raise SystemExit("Questa run non ha tracce: usa una run delle configurazioni judge o agent.")
    n_none = sum(r["gold"] == "none" for r in rows)
    n_pos = len(rows) - n_none
    results = evaluate(rows)
    print(f"Run: {args.run}  ({len(rows)} coppie: {n_none} none, {n_pos} con interazione)\n")
    print("Politiche di veto sul PRIMO tentativo e sul primo verdetto:")
    print(format_table(results, n_none, n_pos))
    n_uncertain = sum(r.get("status") == "uncertain" for r in rows)
    if n_uncertain:
        fb = evaluate_fallbacks(rows)
        print(f"\nRisultato FINALE della run ({n_uncertain} coppie `uncertain`):")
        print(format_table(fb, n_none, n_pos))
        results = {**results, **fb}
    return results


if __name__ == "__main__":
    main()
