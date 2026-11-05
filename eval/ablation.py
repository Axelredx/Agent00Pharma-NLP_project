"""Studio di ablation con ERRORE STANDARD (nessuna chiamata a modelli).

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.ablation                       # test_sample e test_confirm affiancati
    python -m eval.ablation --splits test_sample  # un solo insieme

Per ciascun passo dell'ablation (una sola cosa cambia tra A e B) stampa la variazione del micro F1
(B meno A) con il suo errore standard, e la variazione dei falsi positivi su `none` e delle interazioni
perse. L'errore standard e' la deviazione standard di n_boot differenze ottenute ricampionando le FRASI
(bootstrap a cluster appaiato: stessi ricampionamenti per A e B). FP e persi sono conteggi: la variazione
e' esatta e non ha errore standard. Servono le run baseline, identity, judge, judgev2 e agent3allv2
sull'insieme scelto (nomi: <config>_s2_<insieme>).
"""
from __future__ import annotations

import argparse

import numpy as np

from config import PROCESSED_DIR, RESULTS_DIR
from eval.make_table import LABELS, conf_matrix, final_pred, metrics_from_conf, read_jsonl

STEPS = [
    ("Controllo di identit\\`a (baseline $\\rightarrow$ identity)", "baseline", "identity"),
    ("Giudice v2, veto (identity $\\rightarrow$ judge v2)", "identity", "judgev2"),
    ("Ciclo e controllo dei \\texttt{none} (judge v2 $\\rightarrow$ agente v2, veto)", "judgev2", "agent3allv2:none"),
    ("Politica \\textit{keep} (agente v2: veto $\\rightarrow$ keep)", "agent3allv2:none", "agent3allv2:keep"),
    ("Rubrica v2 invece di v1 (judge v1 $\\rightarrow$ judge v2)", "judge", "judgev2"),
]


def sgn(x: float, nd: int = 3) -> str:
    return (f"{x:+.{nd}f}").replace(".", "{,}").replace("-", "-")


def run_split(split: str, n_boot: int, seed: int, results_dir=RESULTS_DIR, processed_dir=PROCESSED_DIR):
    specs = sorted({s for _, a, b in STEPS for s in (a, b)})
    arrays, gold, ids = {}, None, None
    for spec in specs:
        name, _, fb = spec.partition(":")
        path = results_dir / f"{name}_s2_{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"File {path.name} non trovato (serve per l'ablation su {split}).")
        rows = read_jsonl(path)
        if ids is None:
            ids = [r["pair_id"] for r in rows]
            gold = np.array([LABELS.index(r["gold"]) for r in rows])
        by = {r["pair_id"]: r for r in rows}
        if set(by) != set(ids):
            raise SystemExit(f"{path.name}: coppie diverse dalle altre run")
        fbk = fb or "none"
        arrays[spec] = np.array([LABELS.index(final_pred(by[i], fbk)) if final_pred(by[i], fbk) in LABELS else 0 for i in ids])
    sent_of = {r["pair_id"]: r["sentence_id"] for r in read_jsonl(processed_dir / f"{split}.jsonl")}
    sent = np.array([sent_of[i] for i in ids])
    uniq = list(dict.fromkeys(sent.tolist()))
    members = [np.where(sent == s)[0] for s in uniq]
    rng = np.random.default_rng(seed)
    diffs = [[] for _ in STEPS]
    for _ in range(n_boot):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(uniq), size=len(uniq))])
        cache = {}
        for spec, pr in arrays.items():
            cache[spec] = metrics_from_conf(conf_matrix(gold[idx], pr[idx]))["micro"]
        for j, (_, a, b) in enumerate(STEPS):
            diffs[j].append(cache[b] - cache[a])
    out = []
    for j, (label, a, b) in enumerate(STEPS):
        ma, mb = metrics_from_conf(conf_matrix(gold, arrays[a])), metrics_from_conf(conf_matrix(gold, arrays[b]))
        out.append({"label": label, "dF1": mb["micro"] - ma["micro"], "sd": float(np.std(diffs[j], ddof=1)),
                    "dFP": mb["FP"] - ma["FP"], "dpersi": mb["persi"] - ma["persi"]})
    return out, len(uniq)


def main(argv=None, **kw):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["test_sample", "test_confirm"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    res = {}
    for sp in args.splits:
        res[sp], nsent = run_split(sp, args.n_boot, args.seed, **kw)
        print(f"{sp}: {nsent} frasi, {args.n_boot} ricampionamenti")
    print()
    print(f"{'Componente':<64}" + "".join(f"{sp:>34}" for sp in args.splits))
    for j, (label, _, _) in enumerate(STEPS):
        line = f"{label[:62]:<64}"
        for sp in args.splits:
            r = res[sp][j]
            line += f"{r['dF1']:>+10.3f} +- {r['sd']:.3f}  FP {r['dFP']:+d} persi {r['dpersi']:+d}".rjust(34)
        print(line)
    print("\n--- LaTeX (righe della tabella di ablation) ---")
    for j, (label, _, _) in enumerate(STEPS):
        cells = []
        for sp in args.splits:
            r = res[sp][j]
            cells += [f"${sgn(r['dF1'])}\\pm{r['sd']:.3f}$".replace(".", "{,}"), f"${r['dFP']:+d}$", f"${r['dpersi']:+d}$"]
        print(label + " & " + " & ".join(cells) + "\\\\")
    return res


if __name__ == "__main__":
    main()