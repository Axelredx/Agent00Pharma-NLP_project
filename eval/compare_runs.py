"""Confronto APPAIATO tra due run sulle stesse coppie, con test esatto e casi discordanti.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.compare_runs --a identity_s2_dev_sample --b agent3all_s2_dev_sample --b-fallback keep
    python -m eval.compare_runs --a baseline_s2_dev_sample --b identity_s2_dev_sample --save

Perche': con poche decine o centinaia di coppie, una differenza di F1 di 0,02 puo' essere una
sola coppia. Qui si contano le coppie in cui le due run DIFFERISCONO nell'essere giuste:
  - b "ripara":  a sbagliata, b giusta
  - b "rompe":   a giusta,    b sbagliata
e si applica il test dei segni esatto (binomiale a due code, equivalente al test di McNemar
esatto) per chiedersi se lo squilibrio puo' essere frutto del caso. Si fa sia sull'etichetta
esatta sia sulla sola detection (c'e' un'interazione si/no).

--a-fallback / --b-fallback: per le run con loop, `keep` conta le coppie rimaste `uncertain`
con l'ultima etichetta dell'estrattore invece che con `none` (vedi eval.judge_policies).
Nessuna chiamata a modelli.
"""
from __future__ import annotations

import argparse
import collections
from math import comb
from pathlib import Path

from config import PROCESSED_DIR, RESULTS_DIR
from eval.analyze_errors import read_jsonl, split_of
from src.prompts import mark_entities


def final_pred(row: dict, fallback: str) -> str:
    if fallback == "keep" and row.get("status") == "uncertain" and row.get("uncertain_label"):
        return row["uncertain_label"]
    return row["pred"]


def sign_test_two_sided(b: int, c: int) -> float:
    """p-value esatto a due code per b ripari e c rotture sotto l'ipotesi di parita'."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def compare(rows_a: list[dict], rows_b: list[dict], fa: str = "none", fb: str = "none") -> dict:
    a = {r["pair_id"]: r for r in rows_a}
    b = {r["pair_id"]: r for r in rows_b}
    ids = [r["pair_id"] for r in rows_a if r["pair_id"] in b]
    out = {"n": len(ids), "only_in_a": len(a) - len(ids), "only_in_b": len(b) - len(ids)}
    levels = {
        "label": lambda g, p: p == g,
        "detection": lambda g, p: (p != "none") == (g != "none"),
    }
    for level, ok in levels.items():
        fixes, breaks = [], []
        for pid in ids:
            g = a[pid]["gold"]
            pa, pb = final_pred(a[pid], fa), final_pred(b[pid], fb)
            ca, cb = ok(g, pa), ok(g, pb)
            if cb and not ca:
                fixes.append((pid, g, pa, pb))
            elif ca and not cb:
                breaks.append((pid, g, pa, pb))
        out[level] = {
            "a_correct": sum(ok(a[p]["gold"], final_pred(a[p], fa)) for p in ids),
            "b_correct": sum(ok(b[p]["gold"], final_pred(b[p], fb)) for p in ids),
            "fixes": fixes, "breaks": breaks,
            "p_value": sign_test_two_sided(len(fixes), len(breaks)),
        }
    return out


def build_report(run_a: str, run_b: str, fa: str, fb: str, max_show: int = 30,
                 results_dir: Path = RESULTS_DIR, processed_dir: Path = PROCESSED_DIR) -> str:
    sa, sb = split_of(run_a), split_of(run_b)
    if sa != sb:
        raise SystemExit(f"Le due run usano insiemi diversi ({sa} / {sb})")
    paths = [results_dir / f"{r}.jsonl" for r in (run_a, run_b)]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"File {p.name} non trovato. Run disponibili: "
                             f"{sorted(x.stem for x in results_dir.glob('*.jsonl'))}")
    rows_a, rows_b = read_jsonl(paths[0]), read_jsonl(paths[1])
    data = {r["pair_id"]: r for r in read_jsonl(processed_dir / f"{sa}.jsonl")}
    res = compare(rows_a, rows_b, fa, fb)

    lines = [f"A = {run_a} (fallback {fa})", f"B = {run_b} (fallback {fb})",
             f"Coppie in comune: {res['n']}  (solo in A: {res['only_in_a']}, solo in B: {res['only_in_b']})", ""]
    for level, title in (("label", "ETICHETTA ESATTA"), ("detection", "DETECTION (interazione si/no)")):
        r = res[level]
        lines += [f"{title}: giuste A {r['a_correct']}  B {r['b_correct']}   "
                  f"B ripara {len(r['fixes'])}, B rompe {len(r['breaks'])}   p (test dei segni esatto) = {r['p_value']:.3f}"]
    # A livello di FRASE: le coppie della stessa frase non sono indipendenti (una frase con un elenco
    # lungo puo' dare decine di coppie quasi identiche), quindi si conta anche quante FRASI migliorano o peggiorano.
    net = collections.defaultdict(int)
    for pid, *_ in res["label"]["fixes"]:
        net[data[pid]["sentence_id"]] += 1
    for pid, *_ in res["label"]["breaks"]:
        net[data[pid]["sentence_id"]] -= 1
    up, down = sum(v > 0 for v in net.values()), sum(v < 0 for v in net.values())
    lines.append(f"A LIVELLO DI FRASE (etichetta esatta): frasi con piu' coppie riparate che rotte {up}, "
                 f"con piu' rotte che riparate {down}, compensate {sum(v == 0 for v in net.values())}   "
                 f"p (test dei segni esatto sulle frasi) = {sign_test_two_sided(up, down):.3f}")
    lines.append("")
    lines.append("Il test dice solo se lo squilibrio tra 'ripara' e 'rompe' e' compatibile col caso; "
                 "con pochi casi discordanti non puo' mai dare p piccoli.")

    for kind, key in (("B RIPARA (A sbagliata, B giusta)", "fixes"), ("B ROMPE (A giusta, B sbagliata)", "breaks")):
        items = res["label"][key]
        lines += ["", "=" * 78, f"{kind}: {len(items)} coppie (etichetta esatta)", "=" * 78]
        for pid, g, pa, pb in items[:max_show]:
            rec = data[pid]
            lines += [f"[{pid}]  gold: {g}   A: {pa}   B: {pb}",
                      f"  Farmaci: D1 = {rec['e1']['text']!r}   D2 = {rec['e2']['text']!r}",
                      f"  Frase:   {mark_entities(rec)}", ""]
        if len(items) > max_show:
            lines.append(f"... altre {len(items) - max_show} coppie (usa --max-show)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run di riferimento (es. identity_s2_dev_sample)")
    ap.add_argument("--b", required=True, help="run da confrontare")
    ap.add_argument("--a-fallback", default="none", choices=["none", "keep"])
    ap.add_argument("--b-fallback", default="none", choices=["none", "keep"])
    ap.add_argument("--max-show", type=int, default=30)
    ap.add_argument("--save", action="store_true", help="salva anche in results/compare_<a>__<b>.txt")
    args = ap.parse_args()
    text = build_report(args.a, args.b, args.a_fallback, args.b_fallback, args.max_show)
    print(text)
    if args.save:
        tag = lambda run, fb: run + ("+keep" if fb == "keep" else "")
        out = RESULTS_DIR / f"compare_{tag(args.a, args.a_fallback)}__{tag(args.b, args.b_fallback)}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
