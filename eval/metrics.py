"""Metriche per il task DDI (coppie di farmaci -> tipo di interazione).

Seguono lo schema della gara DDIExtraction 2013:

  * DETECTION: la coppia ha un'interazione (qualsiasi tipo) oppure no?
    Precision / recall / F1 sulla classe "interazione presente".
  * CLASSIFICATION: di che tipo e' l'interazione? Precision / recall / F1 per
    ciascuna delle 4 classi positive (effect, mechanism, advise, int) e la loro
    media micro (e' la cifra che la letteratura riporta di solito) e macro.
    La classe "none" NON entra nelle medie: includerla gonfierebbe i punteggi,
    perche' e' la classe di gran lunga piu' frequente.

Una previsione con un'etichetta non valida viene trattata come "none" e contata
in `invalid_predictions`, cosi' non scompare silenziosamente.
"""
from __future__ import annotations

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import ALL_LABELS, NEGATIVE_LABEL, POSITIVE_LABELS


def _prf(gold: list, pred: list, **kwargs) -> tuple:
    p, r, f, s = precision_recall_fscore_support(gold, pred, zero_division=0, **kwargs)
    return p, r, f, s


def compute_metrics(gold: list[str], pred: list[str]) -> dict:
    """Calcola tutte le metriche. `gold` e `pred` sono liste di etichette."""
    if len(gold) != len(pred):
        raise ValueError(f"Lunghezze diverse: gold={len(gold)} pred={len(pred)}")
    if not gold:
        raise ValueError("Liste vuote")

    invalid = sum(1 for p in pred if p not in ALL_LABELS)
    pred = [p if p in ALL_LABELS else NEGATIVE_LABEL for p in pred]

    # --- Detection (binaria) ---
    gold_bin = [g != NEGATIVE_LABEL for g in gold]
    pred_bin = [p != NEGATIVE_LABEL for p in pred]
    dp, dr, df, _ = _prf(gold_bin, pred_bin, average="binary", pos_label=True)

    # --- Classification (per classe + micro/macro sulle 4 classi positive) ---
    p, r, f, s = _prf(gold, pred, labels=POSITIVE_LABELS, average=None)
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(POSITIVE_LABELS)
    }
    mp, mr, mf, _ = _prf(gold, pred, labels=POSITIVE_LABELS, average="micro")
    Mp, Mr, Mf, _ = _prf(gold, pred, labels=POSITIVE_LABELS, average="macro")

    cm = confusion_matrix(gold, pred, labels=ALL_LABELS)
    confusion = {
        g: {p_: int(cm[i][j]) for j, p_ in enumerate(ALL_LABELS)}
        for i, g in enumerate(ALL_LABELS)
    }

    return {
        "n": len(gold),
        "invalid_predictions": invalid,
        "detection": {"precision": float(dp), "recall": float(dr), "f1": float(df)},
        "classification_micro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "classification_macro": {"precision": float(Mp), "recall": float(Mr), "f1": float(Mf)},
        "per_class": per_class,
        "confusion_matrix": confusion,  # righe = gold, colonne = predetto
    }


def format_report(m: dict) -> str:
    """Tabella testuale leggibile da stampare a terminale."""
    lines = [f"Coppie valutate: {m['n']}   (previsioni non valide: {m['invalid_predictions']})", ""]
    lines.append(f"{'':<22}{'P':>8}{'R':>8}{'F1':>8}{'n':>8}")
    d = m["detection"]
    lines.append(f"{'Detection':<22}{d['precision']:>8.3f}{d['recall']:>8.3f}{d['f1']:>8.3f}")
    for lab, v in m["per_class"].items():
        lines.append(f"{lab:<22}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['support']:>8}")
    for key, name in [("classification_micro", "Micro (4 classi)"), ("classification_macro", "Macro (4 classi)")]:
        v = m[key]
        lines.append(f"{name:<22}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}")
    lines += ["", "Matrice di confusione (righe = gold, colonne = predetto):"]
    labels = list(m["confusion_matrix"])
    lines.append(f"{'':<12}" + "".join(f"{l:>11}" for l in labels))
    for g in labels:
        lines.append(f"{g:<12}" + "".join(f"{m['confusion_matrix'][g][p]:>11}" for p in labels))
    return "\n".join(lines)
