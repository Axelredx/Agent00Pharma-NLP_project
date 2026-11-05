"""Parsing del DDI Corpus (DDIExtraction 2013, formato XML) in record semplici.

Ogni record corrisponde a UNA COPPIA di farmaci in una frase:

    {
      "pair_id": "DDI-DrugBank.d16.s2.p0",
      "doc_id": "DDI-DrugBank.d16",
      "sentence_id": "DDI-DrugBank.d16.s2",
      "source": "DrugBank" | "MedLine",
      "text": "<frase>",
      "e1": {"id", "text", "type", "offsets": [[start, end], ...]},
      "e2": {...},
      "label": "none" | "effect" | "mechanism" | "advise" | "int"
    }

Gli offset sono quelli del corpus: indici di carattere nella frase, con estremo
finale INCLUSO (es. "41-51" -> text[41:52]). Le entita' discontinue (poche, 47
in tutto il corpus) hanno piu' intervalli.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from config import NEGATIVE_LABEL, POSITIVE_LABELS


def _parse_offsets(char_offset: str) -> list[list[int]]:
    """'41-51' -> [[41, 51]];  '44-46;70-86' -> [[44, 46], [70, 86]]."""
    spans = []
    for part in char_offset.split(";"):
        start, end = part.split("-")
        spans.append([int(start), int(end)])
    return spans


def _parse_entity(node: ET.Element, sentence_text: str) -> dict:
    offsets = _parse_offsets(node.get("charOffset"))
    ent = {
        "id": node.get("id"),
        "text": node.get("text"),
        "type": node.get("type"),
        "offsets": offsets,
    }
    # Controllo di coerenza: per le entita' con un solo intervallo, il testo
    # estratto dagli offset deve coincidere con l'attributo "text" del corpus.
    if len(offsets) == 1:
        s, e = offsets[0]
        if sentence_text[s : e + 1] != ent["text"]:
            raise ValueError(
                f"Offset non coerente in {ent['id']}: "
                f"{sentence_text[s:e + 1]!r} != {ent['text']!r}"
            )
    return ent


def _pair_label(pair: ET.Element) -> str | None:
    """Etichetta della coppia, oppure None se l'annotazione e' incompleta.

    Nel corpus esiste una coppia (train, DDI-DrugBank.d236.s29.p0) con ddi="true"
    ma senza l'attributo "type": l'etichetta non e' definita e la coppia viene
    scartata (e segnalata da load_split).
    """
    if pair.get("ddi") == "false":
        return NEGATIVE_LABEL
    label = pair.get("type")
    if label is None:
        return None
    if label not in POSITIVE_LABELS:
        raise ValueError(f"Tipo di interazione inatteso in {pair.get('id')}: {label!r}")
    return label


def parse_xml_file(path: Path, skipped: list[str] | None = None) -> Iterator[dict]:
    """Legge un file XML del corpus e restituisce un record per ogni coppia.

    Le coppie senza etichetta valida vengono saltate; i loro id sono aggiunti
    alla lista `skipped`, se fornita.
    """
    root = ET.parse(path).getroot()
    doc_id = root.get("id")
    source = "DrugBank" if "DrugBank" in doc_id else "MedLine"

    for sent in root.iter("sentence"):
        text = sent.get("text") or ""
        entities = {e.get("id"): _parse_entity(e, text) for e in sent.findall("entity")}
        for pair in sent.findall("pair"):
            label = _pair_label(pair)
            if label is None:
                if skipped is not None:
                    skipped.append(pair.get("id"))
                continue
            yield {
                "pair_id": pair.get("id"),
                "doc_id": doc_id,
                "sentence_id": sent.get("id"),
                "source": source,
                "text": text,
                "e1": entities[pair.get("e1")],
                "e2": entities[pair.get("e2")],
                "label": label,
            }


def load_split(xml_dir: Path) -> tuple[list[dict], list[str]]:
    """Carica tutti i file XML sotto xml_dir (DrugBank + MedLine), in ordine stabile.

    Restituisce (record, id_delle_coppie_scartate).
    """
    files = sorted(xml_dir.rglob("*.xml"))
    if not files:
        raise FileNotFoundError(f"Nessun file XML trovato in {xml_dir}")
    records: list[dict] = []
    skipped: list[str] = []
    for f in files:
        records.extend(parse_xml_file(f, skipped))
    return records, skipped
