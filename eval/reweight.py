"""Stima RIPONDERATA sulla prevalenza reale, con intervalli di confidenza per ricampionamento.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.reweight --runs identity_s2_test_sample agent3all_s2_test_sample:keep judge_s2_test_sample
    python -m eval.reweight --runs baseline_s2_test_sample identity_s2_test_sample --ref baseline_s2_test_sample
    python -m eval.reweight --runs ... --cluster none      # bootstrap sulle coppie (piu' ottimista)

Perche': i campioni (dev_sample, test_sample, test_confirm) sono bilanciati per classe (40 per
classe positiva, 100 `none`), mentre nei dati reali circa l'83% delle coppie e' `none`. Le
metriche sul campione pesano i falsi positivi molto meno di quanto pesino a prevalenza reale.
Qui ogni coppia riceve il peso N_classe_vera(insieme completo) / n_classe_vera(campione), cosi'
il campione "rappresenta" l'insieme completo (test_sample e test_confirm -> test.jsonl;
dev_sample -> dev.jsonl), e si ricalcolano precision, recall e F1.

ATTENZIONE: e' un'analisi approssimata. Assume che i tassi osservati nel campione valgano anche
sull'insieme completo. Su test_sample e' ESPLORATORIA; su test_confirm e' l'endpoint primario.

Intervalli (95%), due modalita':
  --cluster sentence (DEFAULT): bootstrap A CLUSTER. Si ricampionano le FRASI (con le loro coppie),
      perche' le coppie della stessa frase non sono indipendenti (una sola frase con un elenco
      lungo puo' dare decine di coppie quasi identiche); i pesi di classe si ricalcolano a ogni
      ricampionamento (post-stratificazione). E' la modalita' onesta.
  --cluster none: si ricampionano le coppie dentro ogni classe. Ignora la correlazione tra coppie
      della stessa frase e quindi sottostima l'incertezza.
Le differenze rispetto a --ref usano lo STESSO ricampionamento per tutte le run (bootstrap appaiato).

Nome della run con suffisso `:keep` = coppie rimaste `uncertain` con l'ultima etichetta
dell'estrattore invece che `none`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from config import PROCESSED_DIR, RESULTS_DIR

LABELS = ["none", "effect", "mechanism", "advise", "int"]
POS = LABELS[1:]
FULL = {"dev_sample": "dev", "test_sample": "test", "test_confirm": "test"}


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


def metrics(gold: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    """gold/pred: indici in LABELS; w: pesi. Detection (binaria) e micro F1 sulle 4 classi positive."""
    gpos, ppos = gold > 0, pred > 0
    tp = w[gpos & ppos].sum()
    fp = w[~gpos & ppos].sum()
    fn = w[gpos & ~ppos].sum()
    det_p = tp / (tp + fp) if tp + fp else 0.0
    det_r = tp / (tp + fn) if tp + fn else 0.0
    det_f = 2 * det_p * det_r / (det_p + det_r) if det_p + det_r else 0.0
    TP = FP = FN = 0.0
    for k in range(1, 5):
        TP += w[(gold == k) & (pred == k)].sum()
        FP += w[(gold != k) & (pred == k)].sum()
        FN += w[(gold == k) & (pred != k)].sum()
    mp = TP / (TP + FP) if TP + FP else 0.0
    mr = TP / (TP + FN) if TP + FN else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    return {"det_P": det_p, "det_R": det_r, "det_F1": det_f, "micro_F1": mf}


def load_run(spec: str, results_dir: Path):
    name, _, fb = spec.partition(":")
    fb = fb or "none"
    if fb not in ("none", "keep"):
        raise SystemExit(f"Fallback sconosciuto in {spec!r}")
    path = results_dir / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"File {path.name} non trovato. Run disponibili: "
                         f"{sorted(p.stem for p in results_dir.glob('*.jsonl'))}")
    return name, fb, read_jsonl(path)


def run(specs: list[str], ref: str | None = None, n_boot: int = 2000, seed: int = 42,
        results_dir: Path = RESULTS_DIR, processed_dir: Path = PROCESSED_DIR, cluster: str = "sentence") -> dict:
    if cluster not in ("sentence", "none"):
        raise SystemExit("--cluster deve essere 'sentence' o 'none'")
    runs = [load_run(s, results_dir) for s in specs]
    splits = {split_of(n) for n, _, _ in runs}
    if len(splits) != 1:
        raise SystemExit(f"Le run usano insiemi diversi: {splits}")
    split = splits.pop()

    ids = [r["pair_id"] for r in runs[0][2]]
    gold = np.array([LABELS.index(r["gold"]) for r in runs[0][2]])
    preds = {}
    for name, fb, rows in runs:
        by = {r["pair_id"]: r for r in rows}
        if set(by) != set(ids):
            raise SystemExit(f"{name}: coppie diverse dalle altre run")
        preds[(name, fb)] = np.array([LABELS.index(final_pred(by[i], fb)) if final_pred(by[i], fb) in LABELS else 0
                                      for i in ids])

    sample_recs = {r["pair_id"]: r for r in read_jsonl(processed_dir / f"{split}.jsonl")}
    sent = np.array([sample_recs[i]["sentence_id"] for i in ids])

    full = Counter(r["label"] for r in read_jsonl(processed_dir / f"{FULL[split]}.jsonl"))
    n_sample = Counter(int(g) for g in gold)
    weights = np.array([full[LABELS[g]] / n_sample[g] for g in gold])
    total = sum(full[l] for l in LABELS)
    point = {k: metrics(gold, p, weights) for k, p in preds.items()}

    rng = np.random.default_rng(seed)
    boot = {k: {m: [] for m in point[k]} for k in preds}
    n_valid = 0
    if cluster == "none":
        by_class = [np.where(gold == c)[0] for c in range(5)]
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
            for k, p in preds.items():
                for key, v in metrics(gold[idx], p[idx], weights[idx]).items():
                    boot[k][key].append(v)
            n_valid += 1
    else:
        uniq = list(dict.fromkeys(sent.tolist()))
        members = [np.where(sent == s)[0] for s in uniq]
        for _ in range(n_boot):
            chosen = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[c] for c in chosen])
            counts = np.bincount(gold[idx], minlength=5)
            if (counts == 0).any():           # una classe sparita dal ricampionamento: si scarta
                continue
            w = np.array([full[LABELS[c]] / counts[c] for c in range(5)])[gold[idx]]   # post-stratificazione
            for k, p in preds.items():
                for key, v in metrics(gold[idx], p[idx], w).items():
                    boot[k][key].append(v)
            n_valid += 1
    ci = {k: {m: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for m, v in d.items()} for k, d in boot.items()}
    sd = {k: {m: float(np.std(v, ddof=1)) for m, v in d.items()} for k, d in boot.items()}   # errore standard (SD del bootstrap)

    if ref is not None:
        rn, _, rf = ref.partition(":")
        ref_key = (rn, rf or "none")
        if ref_key not in preds:
            raise SystemExit(f"--ref {ref!r} non e' tra le run indicate")
    else:
        ref_key = list(preds)[0]
    diff, diff_sd = {}, {}
    for k in preds:
        if k == ref_key:
            continue
        d, dd = {}, {}
        for m in point[k]:
            arr = np.array(boot[k][m]) - np.array(boot[ref_key][m])
            d[m] = (point[k][m] - point[ref_key][m], np.percentile(arr, 2.5), np.percentile(arr, 97.5), float((arr > 0).mean()))
            dd[m] = float(np.std(arr, ddof=1))
        diff[k] = d
        diff_sd[k] = dd
    return {"split": split, "full_counts": dict(full), "total": total, "point": point, "ci": ci,
            "ref": ref_key, "diff": diff, "sd": sd, "diff_sd": diff_sd, "n_boot": n_boot, "n_valid": n_valid, "cluster": cluster,
            "n_sentences": len(set(sent.tolist())), "n_pairs": len(ids)}


def format_result(res: dict, show_sd: bool = False) -> str:
    fc = res["full_counts"]
    mode = (f"per FRASI ({res['n_sentences']} frasi, {res['n_pairs']} coppie; ricampionamenti validi {res['n_valid']}/{res['n_boot']})"
            if res["cluster"] == "sentence" else f"sulle coppie, stratificato per classe ({res['n_boot']} ricampionamenti)")
    status = ("VALUTAZIONE DI CONFERMA (endpoint primario: differenza di detection F1 rispetto al riferimento)."
              if res["split"] == "test_confirm" else "ESPLORATORIO.")
    lines = [f"Insieme: {res['split']}   pesi dall'insieme completo: " +
             ", ".join(f"{l} {fc[l]}" for l in LABELS) + f"   ({fc['none'] / res['total']:.0%} none)",
             f"Bootstrap {mode}, intervalli al 95%. {status}", ""]
    lines.append(f"{'run':<36}{'detection P':>21}{'detection R':>21}{'detection F1':>21}{'micro F1 (4 cl.)':>23}")
    for k, m in res["point"].items():
        c = res["ci"][k]
        label = f"{k[0]}:{k[1]}"
        cells = [f"{m[x]:.3f} [{c[x][0]:.2f},{c[x][1]:.2f}]" for x in ("det_P", "det_R", "det_F1", "micro_F1")]
        lines.append(f"{label:<36}{cells[0]:>21}{cells[1]:>21}{cells[2]:>21}{cells[3]:>23}")
    r = res["ref"]
    lines += ["", f"Differenza rispetto a {r[0]}:{r[1]} (bootstrap appaiato): valore [IC 95%]  (quota di ricampionamenti con differenza > 0)"]
    for k, d in res["diff"].items():
        lines.append(f"  {k[0]}:{k[1]}")
        for m, lab in (("det_P", "detection P"), ("det_R", "detection R"), ("det_F1", "detection F1"), ("micro_F1", "micro F1")):
            v, lo, hi, share = d[m]
            lines.append(f"      {lab:<14}{v:+.3f} [{lo:+.3f}, {hi:+.3f}]   ({share:.0%})")
    if show_sd:
        lines += ["", "Deviazione standard del bootstrap (= errore standard della stima: quanto varierebbe la stima con altri campioni simili)"]
        lines.append(f"{'run':<36}{'detection P':>14}{'detection R':>14}{'detection F1':>14}{'micro F1':>14}")
        for k in res["point"]:
            sd = res["sd"][k]
            lines.append(f"{k[0] + ':' + k[1]:<36}" + "".join(f"{sd[x]:>14.3f}" for x in ("det_P", "det_R", "det_F1", "micro_F1")))
        lines += ["", f"Deviazione standard della DIFFERENZA rispetto a {r[0]}:{r[1]}"]
        for k, dd in res["diff_sd"].items():
            lines.append(f"{k[0] + ':' + k[1]:<36}" + "".join(f"{dd[x]:>14.3f}" for x in ("det_P", "det_R", "det_F1", "micro_F1")))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run da confrontare, es. identity_s2_test_sample agent3all_s2_test_sample:keep")
    ap.add_argument("--ref", default=None, help="run di riferimento per le differenze (default: la prima)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cluster", choices=["sentence", "none"], default="sentence",
                    help="sentence (default): ricampiona le frasi; none: ricampiona le coppie")
    ap.add_argument("--save", action="store_true", help="salva anche in results/reweight_<split>[_pairs].txt")
    ap.add_argument("--sd", action="store_true", help="stampa anche la deviazione standard del bootstrap (errore standard)")
    args = ap.parse_args(argv)
    res = run(args.runs, args.ref, args.n_boot, args.seed, cluster=args.cluster)
    text = format_result(res, args.sd)
    print(text)
    if args.save:
        suffix = "" if args.cluster == "sentence" else "_pairs"
        out = RESULTS_DIR / f"reweight_{res['split']}{suffix}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"\nSalvato in {out}")
    return res


if __name__ == "__main__":
    main()
