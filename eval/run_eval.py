"""Lancia una configurazione su un insieme di dati e calcola le metriche.

Uso (dalla radice del progetto, con il venv attivo):

    python -m eval.run_eval --config baseline --split dev_sample --shots 2
    python -m eval.run_eval --config agent --max-attempts 3 --split dev_sample --limit 5   # prova veloce

Configurazioni (cumulative: ognuna aggiunge qualcosa alla precedente):
    baseline   una chiamata al modello per coppia
    identity   + controllo "stesso farmaco" (RxNorm) prima della chiamata
    judge      + judge che puo' porre il veto a un'interazione (un solo tentativo, nessun loop)
    agent      + loop: il judge rifiuta con una critica e l'estrattore riprova (--max-attempts)

Le predizioni vengono scritte una per riga in results/<nome>.jsonl SUBITO dopo ogni coppia:
se il programma si interrompe (quota, rete, Ctrl+C) basta rilanciare lo stesso comando,
riprende dalle coppie mancanti.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from config import (
    GEMINI_EXTRACTOR_MODEL,
    GEMINI_JUDGE_MODEL,
    GEMINI_JUDGE_RPM,
    GEMINI_RPM,
    MAX_ITERATIONS,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from eval.metrics import compute_metrics, format_report
from src.baseline import build_baseline
from src.graph import build_agent
from src.llm import CachedLLM, DailyQuotaExceeded
from src.prompts import select_fewshot
from src.tools.rxnorm import RxNormClient

CONFIGS = ("baseline", "identity", "judge", "agent")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def config_label(config: str, max_attempts: int, judge_scope: str = "positives", judge_version: int = 1) -> str:
    """Nome usato nei file dei risultati: 'agent' include il numero di tentativi (agent3);
    con --judge-scope all si aggiunge il suffisso 'all' (agent3all, judgeall)."""
    label = f"agent{max_attempts}" if config == "agent" else config
    if config in ("judge", "agent") and judge_scope == "all":
        label += "all"
    if config in ("judge", "agent") and judge_version == 2:
        label += "v2"
    return label


def build_predictor(config: str, extractor: CachedLLM, judge: CachedLLM | None,
                    rx: RxNormClient | None, shots: int, max_attempts: int, judge_scope: str = "positives",
                    judge_version: int = 1):
    if config not in CONFIGS:
        raise SystemExit(f"Configurazione sconosciuta: {config!r} (disponibili: {', '.join(CONFIGS)})")
    fewshot = select_fewshot(read_jsonl(PROCESSED_DIR / "train.jsonl"), per_label=shots)
    if config == "baseline":
        return build_baseline(extractor, fewshot)
    if config == "identity":
        return build_agent(extractor, None, fewshot, rx, use_identity=True, use_judge=False, max_attempts=1)
    if config == "judge":
        return build_agent(extractor, judge, fewshot, rx, use_identity=True, use_judge=True, max_attempts=1,
                           judge_scope=judge_scope, judge_version=judge_version)
    return build_agent(extractor, judge, fewshot, rx, use_identity=True, use_judge=True, max_attempts=max_attempts,
                       judge_scope=judge_scope, judge_version=judge_version)


def agent_summary(rows: list[dict]) -> dict | None:
    """Statistiche specifiche dell'agente, inclusa la valutazione del judge contro il gold."""
    if not rows or "status" not in rows[0]:
        return None
    summary = {
        "status_counts": dict(collections.Counter(r["status"] for r in rows)),
        "mean_attempts": round(sum(r.get("attempts", 0) for r in rows) / len(rows), 3),
        "judge_calls": sum(r.get("judge_calls", 0) for r in rows),
        "judge_parse_failures": sum(r.get("judge_parse_failures", 0) for r in rows),
    }
    # Valutazione del judge sul PRIMO tentativo: il verdetto e' giusto rispetto al gold?
    two_by_two = collections.Counter()
    for r in rows:
        steps = r.get("trace", [])
        extracts = [t for t in steps if t["step"] == "extract"]
        judges = [t for t in steps if t["step"] == "judge"]
        if not extracts or not judges:
            continue
        first_correct = extracts[0]["label"] == r["gold"]
        two_by_two[("etichetta giusta" if first_correct else "etichetta sbagliata", judges[0]["verdict"])] += 1
    summary["judge_vs_gold_first_attempt"] = {f"{k[0]} -> {k[1]}": v for k, v in sorted(two_by_two.items())}
    return summary


def format_agent_summary(a: dict) -> str:
    lines = ["", "Agente:",
             f"  esiti: {a['status_counts']}",
             f"  tentativi medi dell'estrattore: {a['mean_attempts']}   chiamate al judge: {a['judge_calls']}   "
             f"output del judge illeggibili: {a['judge_parse_failures']}"]
    jg = a["judge_vs_gold_first_attempt"]
    if jg:
        lines.append("  judge sul primo tentativo (etichetta dell'estrattore rispetto al gold -> verdetto):")
        for k, v in jg.items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, llm: CachedLLM | None = None,
         judge_llm: CachedLLM | None = None, rx_client: RxNormClient | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="baseline", choices=CONFIGS)
    ap.add_argument("--split", default="dev_sample", choices=["dev_sample", "test_sample", "test_confirm", "dev", "test"])
    ap.add_argument("--shots", type=int, default=2, help="esempi few-shot per etichetta (0 = zero-shot)")
    ap.add_argument("--max-attempts", type=int, default=MAX_ITERATIONS,
                    help="tentativi massimi dell'estrattore nella configurazione agent (default: MAX_ITERATIONS)")
    ap.add_argument("--judge-scope", default="positives", choices=["positives", "all"],
                    help="positives: il judge controlla solo le interazioni proposte; all: anche i `none` "
                         "(incluso un `none` ottenuto cambiando idea dopo una critica)")
    ap.add_argument("--judge-version", type=int, default=1, choices=[1, 2],
                    help="rubrica del judge: 1 = originale, 2 = basata sulle convenzioni di annotazione del corpus")
    ap.add_argument("--limit", type=int, default=None, help="valuta solo le prime N coppie")
    args = ap.parse_args(argv)

    records = read_jsonl(PROCESSED_DIR / f"{args.split}.jsonl")
    if args.limit:
        records = records[: args.limit]

    llm = llm or CachedLLM(GEMINI_EXTRACTOR_MODEL, rpm=GEMINI_RPM)
    needs_judge = args.config in ("judge", "agent")
    judge = (judge_llm or CachedLLM(GEMINI_JUDGE_MODEL, rpm=GEMINI_JUDGE_RPM)) if needs_judge else None
    rx = (rx_client or RxNormClient()) if args.config != "baseline" else None
    predict = build_predictor(args.config, llm, judge, rx, args.shots, args.max_attempts, args.judge_scope,
                              args.judge_version)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label = config_label(args.config, args.max_attempts, args.judge_scope, args.judge_version)
    run_name = f"{label}_s{args.shots}_{args.split}"
    out_path = RESULTS_DIR / f"{run_name}.jsonl"

    done: dict[str, dict] = {}
    if out_path.exists():
        done = {r["pair_id"]: r for r in read_jsonl(out_path)}
        print(f"Ripresa: {len(done)} coppie gia' valutate in {out_path.name}")

    todo = [r for r in records if r["pair_id"] not in done]
    models = llm.model_name + (f" + judge {judge.model_name}" if judge else "")
    print(f"{label} | modelli={models} | split={args.split} | shots={args.shots} | "
          f"da fare: {len(todo)} / {len(records)}")

    def counters() -> str:
        s = f"chiamate API: {llm.calls}, da cache: {llm.cache_hits}"
        if judge:
            s += f" | judge: {judge.calls} / cache {judge.cache_hits}"
        return s

    try:
        with out_path.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(todo, 1):
                out = predict(rec)
                row = {"pair_id": rec["pair_id"], "gold": rec["label"], **out}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["pair_id"]] = row
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  ({counters()})")
    except DailyQuotaExceeded as e:
        print(f"\nSTOP: {e}")
        print(f"Progressi salvati: {len(done)}/{len(records)} coppie.")
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"\nInterrotto. Progressi salvati: {len(done)}/{len(records)} coppie. Rilancia per riprendere.")
        sys.exit(130)
    except Exception:
        print(f"\nERRORE. Progressi salvati: {len(done)}/{len(records)} coppie. "
              f"Rilancia lo stesso comando per riprendere.")
        raise

    rows = [done[r["pair_id"]] for r in records]
    metrics = compute_metrics([r["gold"] for r in rows], [r["pred"] for r in rows])
    metrics["run"] = {"config": label, "model": llm.model_name, "judge_model": judge.model_name if judge else None,
                      "split": args.split, "shots": args.shots,
                      "max_attempts": args.max_attempts if args.config == "agent" else 1,
                      "judge_scope": args.judge_scope if needs_judge else None,
                      "judge_version": args.judge_version if needs_judge else None,
                      "n_parse_failures": sum(not r["parse_ok"] for r in rows)}
    summary = agent_summary(rows)
    if summary:
        metrics["agent"] = summary
    (RESULTS_DIR / f"{run_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + format_report(metrics))
    if summary:
        print(format_agent_summary(summary))
    return metrics


if __name__ == "__main__":
    main()
