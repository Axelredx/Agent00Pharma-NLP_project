"""Tabella riassuntiva di piu' run, nello stile delle tabelle del report (nessuna chiamata a modelli).

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.make_table --runs baseline_s2_test_confirm identity_s2_test_confirm judge_s2_test_confirm agent3all_s2_test_confirm:keep judgev2_s2_test_confirm agent3allv2_s2_test_confirm:keep
    python -m eval.make_table --runs ... --sd          # aggiunge l'errore standard delle metriche (bootstrap per frasi)
    python -m eval.make_table --runs ... --save        # scrive anche results/table_<insieme>.tex e .txt

Per ogni run stampa: micro F1, macro F1 (sulle 4 classi positive, `none` esclusa), detection P / R / F1,
FP (coppie senza interazione scambiate per interazioni) e persi (interazioni vere finite in `none`).
Il suffisso `:keep` conta le coppie rimaste `uncertain` con l'ultima etichetta dell'estrattore invece che
`none` (come in eval.judge_policies). La macro F1 e' la media delle F1 delle quattro classi positive.
Un'etichetta non valida e' trattata come `none`.

Con --sd si calcola anche l'ERRORE STANDARD di ogni metrica: deviazione standard delle stime ottenute
ricampionando le FRASI (con le loro coppie, bootstrap a cluster, n_boot ricampionamenti, stessi
ricampionamenti per tutte le run). E' l'incertezza dovuta al campione, sulle metriche DEL CAMPIONE (non
riponderate; per quelle riponderate vedi eval.reweight --sd). FP e persi sono conteggi e non hanno errore standard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import PROCESSED_DIR, RESULTS_DIR

LABELS = ["none", "effect", "mechanism", "advise", "int"]
NAMES = {
    "baseline": "baseline", "identity": "identity",
    "judge": "judge v1 (veto)", "judgev2": "judge v2 (veto)",
    "agent3all": "agente v1, anche none", "agent3allv2": "agente v2, anche none",
    "agent3": "agente v1, solo int.", "agent3v2": "agente v2, solo int.",
}
METRICS = [("micro", "micro F1"), ("macro", "macro F1"), ("dP", "detection P"), ("dR", "detection R"), ("dF", "detection F1")]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of(run: str) -> str:
    for s in ("dev_sample", "test_sample", "test_confirm"):
        if run.endswith("_" + s):
            return s
    raise SystemExit(f"Run non su dev_sample, test_sample o test_confirm: {run!r}")


def final_pred(row: dict, fallback: str) -> str:
    if fallback == "keep" and row.get("status") == "uncertain" and row.get("uncertain_label"):
        return row["uncertain_label"]
    return row["pred"]


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def conf_matrix(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(g * 5 + p, minlength=25).reshape(5, 5)      # righe = gold, colonne = predetto


def metrics_from_conf(C: np.ndarray) -> dict:
    tp, fp, fn = C[1:, 1:].sum(), C[0, 1:].sum(), C[1:, 0].sum()
    dp, dr = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    TP = FP = FN = 0
    f1s = []
    for k in range(1, 5):
        tpk, fpk, fnk = C[k, k], C[:, k].sum() - C[k, k], C[k, :].sum() - C[k, k]
        TP, FP, FN = TP + tpk, FP + fpk, FN + fnk
        pk, rk = (tpk / (tpk + fpk) if tpk + fpk else 0.0), (tpk / (tpk + fnk) if tpk + fnk else 0.0)
        f1s.append(f1(pk, rk))
    mp, mr = (TP / (TP + FP) if TP + FP else 0.0), (TP / (TP + FN) if TP + FN else 0.0)
    return {"micro": f1(mp, mr), "macro": float(np.mean(f1s)), "dP": dp, "dR": dr, "dF": f1(dp, dr),
            "FP": int(fp), "persi": int(fn)}


def label_of(name: str, fallback: str) -> str:
    base = name.split("_s2_")[0]
    lab = NAMES.get(base, base)
    if base.startswith("agent"):
        lab += ", keep" if fallback == "keep" else ", veto"
    return lab


def comma(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def main(argv: list[str] | None = None, results_dir: Path = RESULTS_DIR, processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run (con suffisso :keep se serve), nell'ordine delle righe")
    ap.add_argument("--sd", action="store_true", help="aggiunge l'errore standard (bootstrap per frasi)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    specs, splits, loaded = [], set(), []
    for spec in args.runs:
        name, _, fb = spec.partition(":")
        fb = fb or "none"
        if fb not in ("none", "keep"):
            raise SystemExit(f"Politica sconosciuta in {spec!r} (usa :keep oppure niente)")
        path = results_dir / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"File {path.name} non trovato. Run disponibili: {sorted(p.stem for p in results_dir.glob('*.jsonl'))}")
        splits.add(split_of(name))
        loaded.append((name, fb, read_jsonl(path)))
    if len(splits) != 1:
        raise SystemExit(f"Le run usano insiemi diversi: {splits}")
    split = splits.pop()

    ids = [r["pair_id"] for r in loaded[0][2]]
    gold = np.array([LABELS.index(r["gold"]) for r in loaded[0][2]])
    preds, out = [], []
    for name, fb, rows in loaded:
        by = {r["pair_id"]: r for r in rows}
        if set(by) != set(ids):
            raise SystemExit(f"{name}: coppie diverse dalle altre run")
        pr = np.array([LABELS.index(final_pred(by[i], fb)) if final_pred(by[i], fb) in LABELS else 0 for i in ids])
        preds.append(pr)
        m = metrics_from_conf(conf_matrix(gold, pr))
        m["label"] = label_of(name, fb)
        out.append(m)
    n, n_none = len(ids), int((gold == 0).sum())
    n_pos = n - n_none

    if args.sd:
        sent_of = {r["pair_id"]: r["sentence_id"] for r in read_jsonl(processed_dir / f"{split}.jsonl")}
        sent = np.array([sent_of[i] for i in ids])
        uniq = list(dict.fromkeys(sent.tolist()))
        members = [np.where(sent == s)[0] for s in uniq]
        rng = np.random.default_rng(args.seed)
        boot = [{k: [] for k, _ in METRICS} for _ in out]
        for _ in range(args.n_boot):
            chosen = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[c] for c in chosen])
            for j, pr in enumerate(preds):
                m = metrics_from_conf(conf_matrix(gold[idx], pr[idx]))
                for k, _ in METRICS:
                    boot[j][k].append(m[k])
        for j, m in enumerate(out):
            m["sd"] = {k: float(np.std(boot[j][k], ddof=1)) for k, _ in METRICS}

    head = f"{split}: {n} coppie ({n_none} senza interazione, {n_pos} con interazione). FP su {n_none}; persi su {n_pos}."
    tex = ["\\begin{tabular}{lcccrr}", "\\toprule",
           "Configurazione & micro F1 & macro F1 & detection P / R / F1 & FP & persi\\\\", "\\midrule"]
    txt = [f"{'Configurazione':<30}{'micro F1':>9}{'macro F1':>9}   {'detection P / R / F1':<22}{'FP':>4}{'persi':>7}"]
    for s in out:
        det = f"{comma(s['dP'])} / {comma(s['dR'])} / {comma(s['dF'])}"
        tex.append(f"{s['label']} & {comma(s['micro'])} & {comma(s['macro'])} & {det} & {s['FP']} & {s['persi']}\\\\")
        txt.append(f"{s['label']:<30}{comma(s['micro']):>9}{comma(s['macro']):>9}   {det:<22}{s['FP']:>4}{s['persi']:>7}")
    tex += ["\\bottomrule", "\\end{tabular}"]
    print(head + "\n")
    print("\n".join(txt))
    print("\n--- LaTeX (da incollare dentro \\begin{table}...\\end{table}) ---")
    print("\n".join(tex))

    sdtxt, sdtex = [], []
    if args.sd:
        sdtex = ["\\begin{tabular}{lccccc}", "\\toprule",
                 "Configurazione & micro F1 & macro F1 & detection P & detection R & detection F1\\\\", "\\midrule"]
        sdtxt = [f"{'Configurazione':<30}" + "".join(f"{lab:>14}" for _, lab in METRICS)]
        for s in out:
            sdtex.append(f"{s['label']} & " + " & ".join(comma(s['sd'][k]) for k, _ in METRICS) + "\\\\")
            sdtxt.append(f"{s['label']:<30}" + "".join(f"{comma(s['sd'][k]):>14}" for k, _ in METRICS))
        sdtex += ["\\bottomrule", "\\end{tabular}"]
        print(f"\nErrore standard (deviazione standard del bootstrap per frasi: {len(uniq)} frasi, {args.n_boot} ricampionamenti)\n")
        print("\n".join(sdtxt))
        print("\n--- LaTeX errore standard ---")
        print("\n".join(sdtex))
    if args.save:
        results_dir.mkdir(parents=True, exist_ok=True)
        body = head + "\n\n" + "\n".join(txt) + "\n" + (("\nErrore standard (bootstrap per frasi)\n\n" + "\n".join(sdtxt) + "\n") if args.sd else "")
        (results_dir / f"table_{split}.txt").write_text(body, encoding="utf-8")
        (results_dir / f"table_{split}.tex").write_text("\n".join(tex) + "\n" + (("\n% errore standard\n" + "\n".join(sdtex) + "\n") if args.sd else ""), encoding="utf-8")
        print(f"\nSalvato in {results_dir / ('table_' + split + '.tex')} e .txt")
    return out


if __name__ == "__main__":
    main()