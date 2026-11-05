"""Tutti i prompt del progetto, in un solo posto (cosi' si confrontano e si citano nel report).

Le definizioni delle etichette seguono le linee guida di annotazione del DDI
Corpus (Herrero-Zazo et al., 2013), riassunte in modo operativo.
"""
from __future__ import annotations

import json
import random

from config import ALL_LABELS, RANDOM_SEED

LABEL_DEFINITIONS = """\
- advise: the sentence gives a recommendation or advice about the concomitant use of the two drugs (e.g. "should not be given together", "use with caution", "dose adjustment may be necessary").
- effect: the sentence describes the effect or clinical outcome of the interaction (e.g. increased risk of bleeding, reduced efficacy, enhanced sedation), including pharmacodynamic effects.
- mechanism: the sentence describes a pharmacokinetic mechanism of the interaction (absorption, distribution, metabolism or excretion), e.g. one drug increases or decreases the plasma level of the other, or inhibits its metabolism.
- int: the sentence only states that the two drugs interact, without giving the effect, the mechanism or any advice.
- none: the sentence does NOT describe an interaction between these two specific drugs (for example they are only listed together, compared, or the interaction concerns other drugs)."""

BASELINE_INSTRUCTIONS = f"""\
You are an expert in pharmacology and biomedical text mining.

Task: drug-drug interaction (DDI) classification. You are given a sentence from a drug-related text in which two drug mentions are marked with <D1>...</D1> and <D2>...</D2>. Decide whether the sentence states an interaction BETWEEN THESE TWO SPECIFIC mentions and, if so, of which type.

Labels:
{LABEL_DEFINITIONS}

Judge only the two marked drugs and only what the sentence says, not what you know from elsewhere. Choose the single best label."""

OUTPUT_FORMAT = 'Answer with a JSON object only, with no other text: {"label": "<one of: none, effect, mechanism, advise, int>"}'


def _spans_overlap(a: list[list[int]], b: list[list[int]]) -> bool:
    return any(not (e1 < s2 or e2 < s1) for s1, e1 in a for s2, e2 in b)


def mark_entities(record: dict) -> str:
    """Restituisce la frase con le due entita' marcate da <D1>..</D1> e <D2>..</D2>.

    Si marca per OFFSET (non per testo), cosi' se un farmaco compare piu' volte
    nella frase si sa quale occorrenza e' quella della coppia. Se i due
    intervalli si sovrappongono (coordinazioni come "diagnostic or therapeutic
    monoclonal antibodies", 41 casi su ~26.000 nel train) i tag annidati
    renderebbero il testo illeggibile: in quel caso la frase resta senza tag e le
    entita' sono comunque indicate nelle righe "Drug 1 / Drug 2" del prompt.
    """
    text = record["text"]
    e1, e2 = record["e1"]["offsets"], record["e2"]["offsets"]
    if _spans_overlap(e1, e2):
        return text.strip()

    inserts = []  # (posizione, tag)
    for spans, tag in ((e1, "D1"), (e2, "D2")):
        for start, end in spans:
            inserts.append((start, f"<{tag}>"))
            inserts.append((end + 1, f"</{tag}>"))
    # dal fondo verso l'inizio, cosi' gli offset precedenti non si spostano
    for pos, tag in sorted(inserts, key=lambda x: x[0], reverse=True):
        text = text[:pos] + tag + text[pos:]
    return text.strip()


def format_case(record: dict) -> str:
    return (
        f"Sentence: {mark_entities(record)}\n"
        f"Drug 1: {record['e1']['text']}\n"
        f"Drug 2: {record['e2']['text']}"
    )


def select_fewshot(train: list[dict], per_label: int, seed: int = RANDOM_SEED,
                   max_chars: int = 250) -> list[dict]:
    """Sceglie in modo deterministico `per_label` esempi per etichetta dal TRAIN.

    Si preferiscono frasi corte (max_chars) per tenere breve il prompt. Gli
    esempi vengono mescolati, per non avere blocchi di una stessa etichetta.
    Il train non ha documenti in comune con dev e test, quindi nessun leakage.
    """
    if per_label <= 0:
        return []
    rng = random.Random(seed)
    chosen: list[dict] = []
    for label in ALL_LABELS:
        pool = sorted(
            (r for r in train if r["label"] == label and len(r["text"]) <= max_chars),
            key=lambda r: r["pair_id"],
        )
        rng.shuffle(pool)
        chosen.extend(pool[:per_label])
    rng.shuffle(chosen)
    return chosen


def build_baseline_prompt(record: dict, fewshot: list[dict], feedback: str | None = None) -> str:
    """Prompt dell'estrattore. Senza `feedback` e' ESATTAMENTE il prompt della baseline
    (stesso testo => stessa chiave di cache => il primo tentativo dell'agente non costa
    chiamate in piu' e il confronto con la baseline e' pulito). Con `feedback` aggiunge il
    commento del judge dopo un rifiuto, per il tentativo successivo (loop)."""
    parts = [BASELINE_INSTRUCTIONS]
    if fewshot:
        parts.append("Examples:")
        for ex in fewshot:
            parts.append(f"{format_case(ex)}\nAnswer: {json.dumps({'label': ex['label']})}")
    parts.append(OUTPUT_FORMAT)
    if feedback:
        parts.append(
            "A reviewer rejected your previous answer for the case below. "
            f"Reviewer's comment: {feedback}\n"
            "Reconsider the case from scratch, taking the comment into account. "
            "Keep your answer only if you are sure the reviewer is wrong."
        )
    parts.append(f"Now classify this case.\n{format_case(record)}\nAnswer:")
    return "\n\n".join(parts)


JUDGE_INSTRUCTIONS = f"""\
You are a careful reviewer of drug-drug interaction (DDI) annotations in biomedical text.

A sentence contains two drug mentions, marked <D1>...</D1> and <D2>...</D2>. A system proposed a label for the relation between D1 and D2. Decide whether the proposed label is justified by the sentence alone.

Labels:
{LABEL_DEFINITIONS}

Check in order:
1. Between the pair. Does the sentence state that D1 and D2 act on EACH OTHER (one changes the effect, level or handling of the other, or the advice concerns using them together)? It is NOT an interaction between D1 and D2 if: they are only listed or coordinated in an enumeration; each one interacts with a third drug but not with the other; one is a drug class and the other a member of that class; they are only compared; or the sentence describes a study design or a general property without stating an interaction between them.
2. Type. If there is an interaction between them, does the proposed type match the definitions above?

Answer with a JSON object only, with no other text:
{{"verdict": "accept" or "reject", "issue": "<one of: none, not_between_pair, no_interaction_stated, wrong_type>", "critique": "<one or two sentences addressed to the system, saying what to reconsider>"}}
Use "accept" only if both checks pass; otherwise "reject"."""


JUDGE_INSTRUCTIONS_V2 = f"""\
You are a careful reviewer of drug-drug interaction (DDI) annotations in biomedical text.

A sentence contains two drug mentions, marked <D1>...</D1> and <D2>...</D2>. Here "drug" includes brand names, abbreviations and drug classes (for example "diuretics" or "MAO inhibitors"). A system proposed a label for the relation between D1 and D2. Decide whether the proposed label is justified by the sentence.

Labels:
{LABEL_DEFINITIONS}

Annotation conventions of this corpus:
- A relation is annotated between two marked drugs when the sentence states or clearly implies that they interact: one changes the effect, level or handling of the other, or the sentence advises about using them together. A sentence that only announces an interaction between them, even as a title ("Interaction of X and Y in rats"), supports the label "int".
- Drug classes count as drugs. If the sentence says that drug X interacts with a class, or with examples of that class, then X interacts with the class and with each of those examples.
- In an enumeration ("A, B and C interact with E") each of A, B and C interacts with E; A, B and C do not interact with each other unless the sentence says so.
- A class and one of its own members (for example "methylxanthines such as caffeine") do not interact with each other.
- When a drug is followed by a description of its class in apposition or parentheses ("temazepam (a benzodiazepine)"), the class mention is only a description, not a second interacting drug.
- A sentence that describes an experiment or a general property without stating any interaction between the two drugs has no interaction.

Check in order:
1. Does the sentence state an interaction between D1 and D2 under these conventions?
2. Only if it does: is the proposed type supported by the sentence? Do not reject only because another type would also be defensible or because the sentence contains several kinds of information: accept if the proposed type is supported at least in part. Reject for the type only if it is clearly unsupported.

Reject only when you are confident that the label is not supported. If in doubt, accept.

Answer with a JSON object only, with no other text:
{{"verdict": "accept" or "reject", "issue": "<one of: none, not_between_pair, no_interaction_stated, wrong_type>", "critique": "<one or two sentences addressed to the system, saying what to reconsider>"}}"""


def build_judge_prompt(record: dict, proposed_label: str, version: int = 1) -> str:
    """Prompt del judge. version=1: rubrica originale (non modificata, cosi' la cache resta valida);
    version=2: rubrica basata sulle convenzioni di annotazione del DDI Corpus."""
    instructions = JUDGE_INSTRUCTIONS_V2 if version == 2 else JUDGE_INSTRUCTIONS
    return (
        f"{instructions}\n\n"
        f"Review this case.\n{format_case(record)}\n"
        f"Proposed label: {proposed_label}\n"
        "Answer:"
    )
