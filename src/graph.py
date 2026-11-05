"""L'agente LangGraph: controllo di identita', estrattore, judge e loop di riflessione.

Flusso (ogni coppia di farmaci attraversa il grafo):

    identity ──same drug──────────────────────────────────────────► finalize (none)
       │
       └─different──► extract ──label none (e judge_scope="positives")──► finalize
                         ▲          │
                         │          └─label interazione──► judge ──accept──► finalize
                         │                                   │
                         └──── feedback (critica) ◄── reject ┘
                                   (se restano tentativi e l'estrattore non si ripete;
                                    altrimenti: finalize come "uncertain")

Scelte di progetto (da discutere nel report):
  * Il PRIMO tentativo dell'estrattore usa esattamente il prompt della baseline: cosi'
    ogni differenza di risultato si deve a identity/judge/loop, non a un prompt diverso.
  * Il judge giudica solo le predizioni di interazione (judge_scope="positives"): i falsi
    positivi sono il punto debole misurato (la recall della baseline e' gia' alta).
  * Il judge NON cambia l'etichetta: la contesta con una critica, e l'estrattore decide.
    Cosi' i due ruoli restano separati.
  * Il loop e' limitato (max_attempts) e si ferma se l'estrattore ripete la stessa risposta
    (stallo). Se alla fine il judge rifiuta ancora, la coppia e' "uncertain": come etichetta
    finale si usa `fallback` ("none" = il judge ha potere di veto; "keep" = si tiene
    l'ultima etichetta dell'estrattore). `uncertain_label` e' salvata in ogni caso, cosi' le
    due politiche si confrontano senza rifare le chiamate.
  * Un output illeggibile del judge non cancella predizioni (si tratta come "accept") ma e'
    segnalato; un output illeggibile dell'estrattore resta INVALID e viene contato.
"""
from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from src.llm import CachedLLM
from src.prompts import build_baseline_prompt, build_judge_prompt
from src.schemas import INVALID_LABEL, parse_prediction, parse_verdict
from src.tools.rxnorm import RxNormClient, same_drug


class AgentState(TypedDict, total=False):
    record: dict
    attempts: int                # tentativi gia' fatti dall'estrattore
    labels: list[str]            # etichetta proposta a ogni tentativo
    raw: str                     # ultima risposta grezza dell'estrattore
    feedback: str | None         # critica del judge per il prossimo tentativo
    verdict: str | None          # accept | reject | None (judge non chiamato)
    judge_parse_ok: bool
    status: str
    pred: str
    uncertain_label: str | None
    trace: list[dict]


def build_agent(
    extractor_llm: CachedLLM,
    judge_llm: CachedLLM | None,
    fewshot: list[dict],
    rx_client: RxNormClient | None = None,
    *,
    use_identity: bool = True,
    use_judge: bool = True,
    max_attempts: int = 3,
    judge_scope: str = "positives",
    fallback: str = "none",
    judge_version: int = 1,
) -> Callable[[dict], dict]:
    """Costruisce il grafo e restituisce predict(record) -> dict (come la baseline)."""
    if use_judge and judge_llm is None:
        raise ValueError("use_judge=True richiede un judge_llm")
    if fallback not in ("none", "keep"):
        raise ValueError("fallback deve essere 'none' o 'keep'")
    if max_attempts < 1:
        raise ValueError("max_attempts deve essere >= 1")

    # ---- nodi -----------------------------------------------------------
    def identity_node(state: AgentState) -> dict:
        rec = state["record"]
        same, why = same_drug(rec["e1"]["text"], rec["e2"]["text"], rx_client)
        trace = state.get("trace", []) + [{"step": "identity", "same": same, "reason": why}]
        if same:
            return {"status": "identity", "pred": "none", "uncertain_label": None, "trace": trace}
        return {"trace": trace}

    def extract_node(state: AgentState) -> dict:
        prompt = build_baseline_prompt(state["record"], fewshot, feedback=state.get("feedback"))
        raw = extractor_llm.complete(prompt)
        label = parse_prediction(raw)["label"]
        attempts = state.get("attempts", 0) + 1
        trace = state.get("trace", []) + [{"step": "extract", "attempt": attempts, "label": label}]
        return {"raw": raw, "attempts": attempts, "labels": state.get("labels", []) + [label],
                "feedback": None, "trace": trace}

    def judge_node(state: AgentState) -> dict:
        label = state["labels"][-1]
        raw = judge_llm.complete(build_judge_prompt(state["record"], label, judge_version))
        v = parse_verdict(raw)
        trace = state.get("trace", []) + [{"step": "judge", "verdict": v["verdict"], "issue": v["issue"],
                                           "critique": v["critique"], "parse_ok": v["parse_ok"]}]
        out = {"verdict": v["verdict"], "judge_parse_ok": v["parse_ok"], "trace": trace}
        if v["verdict"] == "reject":
            out["feedback"] = f"{v['critique']} (issue: {v['issue']})".strip()
        return out

    def finalize_node(state: AgentState) -> dict:
        if state.get("status") == "identity":  # gia' deciso dal controllo di identita'
            return {}
        labels = state["labels"]
        last = labels[-1]
        status = state.get("status")
        if last == INVALID_LABEL:
            return {"status": "invalid", "pred": INVALID_LABEL, "uncertain_label": None}
        if status == "uncertain":
            return {"pred": "none" if fallback == "none" else last, "uncertain_label": last}
        if status is None:
            status = "accepted_none" if last == "none" and use_judge else \
                     "accepted" if use_judge else "no_judge"
        return {"status": status, "pred": last, "uncertain_label": None}

    # ---- instradamento ----------------------------------------------------
    def after_identity(state: AgentState) -> str:
        return "finalize" if state.get("status") == "identity" else "extract"

    def stalled(state: AgentState) -> bool:
        """L'estrattore ha ripetuto la stessa etichetta dopo la critica del judge."""
        labels = state["labels"]
        return len(labels) >= 2 and labels[-1] == labels[-2]

    def after_extract(state: AgentState) -> str:
        last = state["labels"][-1]
        if not use_judge or last == INVALID_LABEL:
            return "finalize"
        if stalled(state):  # il judge ha gia' rifiutato questa stessa etichetta: inutile richiamarlo
            return "uncertain"
        if last == "none" and judge_scope == "positives":
            return "finalize"
        return "judge"

    def can_retry(state: AgentState) -> bool:
        return state["attempts"] < max_attempts

    def mark_uncertain(state: AgentState) -> dict:
        """Il judge ha rifiutato e non si puo' (o non serve) riprovare."""
        reason = "stalled" if stalled(state) else "max_attempts"
        trace = state.get("trace", []) + [{"step": "uncertain", "reason": reason}]
        return {"status": "uncertain", "trace": trace}

    def judge_router(state: AgentState) -> str:
        if state["verdict"] != "reject":
            return "finalize"
        return "extract" if can_retry(state) else "uncertain"

    # ---- grafo --------------------------------------------------------------
    g = StateGraph(AgentState)
    g.add_node("identity", identity_node)
    g.add_node("extract", extract_node)
    g.add_node("judge", judge_node)
    g.add_node("uncertain", mark_uncertain)
    g.add_node("finalize", finalize_node)

    if use_identity:
        g.add_edge(START, "identity")
        g.add_conditional_edges("identity", after_identity, {"finalize": "finalize", "extract": "extract"})
    else:
        g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract,
                            {"finalize": "finalize", "judge": "judge", "uncertain": "uncertain"})
    g.add_conditional_edges("judge", judge_router,
                            {"finalize": "finalize", "extract": "extract", "uncertain": "uncertain"})
    g.add_edge("uncertain", "finalize")
    g.add_edge("finalize", END)
    app = g.compile()

    def predict(record: dict) -> dict:
        s = app.invoke({"record": record, "attempts": 0, "labels": [], "trace": []})
        judges = [t for t in s["trace"] if t["step"] == "judge"]
        return {
            "pred": s["pred"],
            "raw": s.get("raw", ""),
            "parse_ok": s["pred"] != INVALID_LABEL,
            "status": s["status"],
            "attempts": s.get("attempts", 0),
            "uncertain_label": s.get("uncertain_label"),
            "judge_calls": len(judges),
            "judge_parse_failures": sum(not j["parse_ok"] for j in judges),
            "trace": s["trace"],
        }

    predict.graph = app  # per ispezionare/disegnare il grafo
    return predict
