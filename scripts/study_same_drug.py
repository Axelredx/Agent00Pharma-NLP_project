"""Studio offline sul TRAIN: quanto vale la normalizzazione RxNorm per riconoscere "stesso farmaco"?

    python -m scripts.study_same_drug                 # studio completo (qualche minuto)
    python -m scripts.study_same_drug --max-names 30  # prova veloce

Domande a cui risponde, SENZA usare l'LLM (nessuna quota Gemini):
  1. Tra le coppie `none` con nomi diversi (stringhe diverse), quante sono in realta'
     lo stesso principio attivo secondo RxNorm? (= guadagno oltre il confronto di stringhe)
  2. Tra le coppie con un'interazione, quante verrebbero unite per errore? (= danno)
     La regola e' utilizzabile solo se questo numero e' (quasi) zero.
  3. Quanti nomi RxNorm riconosce, per tipo di entita' (drug, brand, drug_n)?

Le entita' di tipo `group` (classi come "antacids") sono escluse: non sono farmaci singoli.
Le richieste a RxNorm sono salvate in cache/rxnorm.jsonl: se si interrompe, rilanciare riprende.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from src.tools.rxnorm import RxNormClient, RxNormError, normalize_name, same_drug


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--none-sample", type=int, default=3000, help="coppie `none` da campionare")
    ap.add_argument("--max-names", type=int, default=None, help="limita i nomi distinti (solo per prove)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    with (PROCESSED_DIR / "train.jsonl").open(encoding="utf-8") as f:
        train = [json.loads(l) for l in f if l.strip()]

    # Solo coppie di farmaci singoli con stringhe diverse (le uguali le gestisce il confronto di stringhe)
    cand = [r for r in train
            if r["e1"]["type"] != "group" and r["e2"]["type"] != "group"
            and normalize_name(r["e1"]["text"]) != normalize_name(r["e2"]["text"])]
    pos = [r for r in cand if r["label"] != "none"]
    none_all = [r for r in cand if r["label"] == "none"]
    rng = random.Random(args.seed)
    none = rng.sample(none_all, min(args.none_sample, len(none_all)))
    sample = pos + none

    names: dict[str, str] = {}          # nome normalizzato -> testo originale
    types: dict[str, set[str]] = collections.defaultdict(set)
    for r in sample:
        for e in (r["e1"], r["e2"]):
            n = normalize_name(e["text"])
            names.setdefault(n, e["text"])
            types[n].add(e["type"])
    ordered = sorted(names)
    if args.max_names:
        ordered = ordered[: args.max_names]
        keep = set(ordered)
        sample = [r for r in sample if normalize_name(r["e1"]["text"]) in keep and normalize_name(r["e2"]["text"]) in keep]
    print(f"Coppie nel campione: {len(sample)} (positive {sum(r['label'] != 'none' for r in sample)}, "
          f"none {sum(r['label'] == 'none' for r in sample)}); nomi distinti da risolvere: {len(ordered)}")

    client = RxNormClient()
    try:
        for i, n in enumerate(ordered, 1):
            client.resolve(names[n])
            if i % 100 == 0 or i == len(ordered):
                print(f"  risolti {i}/{len(ordered)} nomi  (HTTP: {client.http_calls}, cache: {client.cache_hits})")
    except RxNormError as e:
        print(f"\nERRORE: {e}\nI progressi sono in cache: rilancia lo stesso comando per riprendere.")
        raise SystemExit(1)

    # --- 3. Copertura per tipo di entita' ---
    cov = collections.defaultdict(collections.Counter)
    for n in ordered:
        st = client.resolve(names[n]).status
        for t in types[n]:
            cov[t][st] += 1

    # --- 1 e 2. Unioni per classe ---
    tally = {"none": collections.Counter(), "positive": collections.Counter()}
    merges = {"none": [], "positive": []}
    for r in sample:
        cls = "none" if r["label"] == "none" else "positive"
        ok, why = same_drug(r["e1"]["text"], r["e2"]["text"], client)
        tally[cls][why] += 1
        if ok:
            merges[cls].append(r)

    lines = ["STUDIO: normalizzazione RxNorm per il riconoscimento di 'stesso farmaco' (TRAIN)", ""]
    lines.append("Copertura RxNorm sui nomi (per tipo di entita'):")
    for t in sorted(cov):
        tot = sum(cov[t].values())
        ok_n = cov[t]["ok"]
        lines.append(f"  {t:<8} {ok_n}/{tot} riconosciuti con un principio attivo ({100 * ok_n / tot:.0f}%)   {dict(cov[t])}")
    lines.append("")
    n_none, n_pos = sum(tally["none"].values()), sum(tally["positive"].values())
    lines.append(f"Coppie `none` con nomi diversi (campione {n_none}):")
    lines.append(f"  stesso principio attivo secondo RxNorm: {tally['none']['rxnorm']}  "
                 f"({100 * tally['none']['rxnorm'] / max(1, n_none):.1f}%)   dettaglio: {dict(tally['none'])}")
    lines.append(f"Coppie con INTERAZIONE e nomi diversi (tutte, {n_pos}):")
    lines.append(f"  unite per errore da RxNorm: {tally['positive']['rxnorm']}  "
                 f"({100 * tally['positive']['rxnorm'] / max(1, n_pos):.1f}%)   dettaglio: {dict(tally['positive'])}")
    if n_none:
        est = tally["none"]["rxnorm"] / n_none * len(none_all)
        lines.append(f"\nStima sul train completo: circa {est:.0f} coppie `none` in piu' riconoscibili "
                     f"rispetto al solo confronto di stringhe (su {len(none_all)} `none` con nomi diversi tra farmaci singoli).")

    def show(title, items, limit):
        lines.append("")
        lines.append(title)
        for r in items[:limit]:
            a, b = client.resolve(r["e1"]["text"]), client.resolve(r["e2"]["text"])
            ing = ", ".join(a.ingredient_names.values())
            lines.append(f"  [{r['label']}] {r['e1']['text']!r} ~ {r['e2']['text']!r}   -> {ing}   ({r['pair_id']})")
    show("Esempi di unioni su coppie `none` (primi 25):", merges["none"], 25)
    show("TUTTE le unioni su coppie con interazione (da guardare una per una):", merges["positive"], 100)

    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "study_same_drug.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
