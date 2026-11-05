"""Figure per il report, dalle stesse stime di `eval.reweight` (nessuna chiamata a modelli).

Uso (dalla radice del progetto, con il venv attivo):

    python -m scripts.make_figures --runs baseline_s2_test_sample identity_s2_test_sample judge_s2_test_sample judgev2_s2_test_sample agent3all_s2_test_sample:keep agent3allv2_s2_test_sample:keep --ref identity_s2_test_sample

Produce in report/figures/:
  punti_operativi.png   precision (asse x) e recall (asse y) della DETECTION a prevalenza reale,
                        con intervalli al 95% e curve a F1 costante
  differenze.png        differenza rispetto al riferimento (detection F1 e micro F1) con IC 95%

Sono stime ESPLORATORIE (campione riponderato sulla prevalenza reale, bootstrap): vedi eval.reweight.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.reweight import run

PRETTY = {"baseline": "baseline", "identity": "identity", "judge": "judge v1 (veto)",
          "judgev2": "judge v2 (veto)", "agent3all": "agente v1", "agent3allv2": "agente v2",
          "agent3": "agente v1 (solo interazioni)", "agent3v2": "agente v2 (solo interazioni)"}
COLORS = {"baseline": "#888888", "identity": "#444444", "judge": "#d98c1f", "judgev2": "#1f6fd9",
          "agent3all": "#e0b36a", "agent3allv2": "#7fb0ec"}


def base_name(key: tuple[str, str]) -> str:
    name = key[0]
    for suffix in ("_s2_test_sample", "_s2_dev_sample"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.split("_s")[0]


def label_of(key: tuple[str, str]) -> str:
    b = base_name(key)
    lab = PRETTY.get(b, b)
    if b.startswith("agent"):
        lab += f" [{key[1]}]"
    return lab


# spostamenti (in punti) delle etichette, per evitare sovrapposizioni tra punti vicini
OFFSETS = {"baseline": (-8, -14, "right"), "identity": (8, 6, "left")}


def plot_operating_points(res: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    x_lo, x_hi, y_lo, y_hi = 0.30, 0.90, 0.70, 1.0
    recall = np.linspace(y_lo, y_hi, 300)
    for f in (0.5, 0.6, 0.7, 0.8):
        prec = f * recall / (2 * recall - f)          # curva a F1 costante: P = fR / (2R - f)
        ok = (prec >= x_lo) & (prec <= x_hi)
        ax.plot(prec[ok], recall[ok], color="#cccccc", lw=0.9, zorder=0)
        if ok.any():
            j = np.where(ok)[0][-1]
            ax.annotate(f"F1 = {f}", (prec[j], recall[j]), textcoords="offset points", xytext=(3, -10),
                        fontsize=7, color="#999999")
    for key, m in res["point"].items():
        c = res["ci"][key]
        b = base_name(key)
        color = COLORS.get(b, "#2a9d8f")
        ax.errorbar(m["det_P"], m["det_R"],
                    xerr=[[m["det_P"] - c["det_P"][0]], [c["det_P"][1] - m["det_P"]]],
                    yerr=[[m["det_R"] - c["det_R"][0]], [c["det_R"][1] - m["det_R"]]],
                    fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.8, zorder=1)
        ax.scatter([m["det_P"]], [m["det_R"]], color=color, s=48, marker="s" if b.startswith("agent") else "o",
                   zorder=2)
        dx, dy, ha = OFFSETS.get(b, (7, 6, "left"))
        ax.annotate(label_of(key), (m["det_P"], m["det_R"]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, ha=ha)
    ax.set_xlabel("Precision della detection (a prevalenza reale, 83% di coppie senza interazione)")
    ax.set_ylabel("Recall della detection")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title("Punti operativi (stima esplorativa, IC 95% bootstrap)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_differences(res: dict, out: Path) -> None:
    ref = res["ref"]
    items = list(res["diff"].items())
    metrics = (("det_F1", "detection F1"), ("micro_F1", "micro F1 (4 classi)"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(items) + 1.6), sharey=True)
    for ax, (mk, title) in zip(axes, metrics):
        for i, (key, d) in enumerate(items):
            v, lo, hi, _ = d[mk]
            color = COLORS.get(base_name(key), "#2a9d8f")
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=color, ecolor=color, capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([label_of(k) for k, _ in items], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel(f"differenza rispetto a {label_of(ref)}")
    fig.suptitle("Differenze a prevalenza reale con IC 95% (bootstrap appaiato, esplorativo)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None, outdir: Path | None = None, **kwargs) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--outdir", default="report/figures")
    args = ap.parse_args(argv)
    res = run(args.runs, args.ref, args.n_boot, **kwargs)
    out = outdir or Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot_operating_points(res, out / "punti_operativi.png")
    plot_differences(res, out / "differenze.png")
    print(f"Figure salvate in {out}/: punti_operativi.png, differenze.png")
    return res


if __name__ == "__main__":
    main()
