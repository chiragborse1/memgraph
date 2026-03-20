"""
Extraction pipeline
───────────────────
Given a text chunk, calls the LLM to extract:
  - entities: [{name, type, description}]
  - relations: [{subject, predicate, object}]

Returns structured Python dicts ready to upsert into Neo4j.
"""

import json
import re
import structlog
import spacy

from graph.llm_client import chat_json
from config import get_settings

log = structlog.get_logger()
settings = get_settings()

# Load spaCy for cheap pre-filtering (avoids sending garbage to the LLM)
_nlp = spacy.load("en_core_web_sm")

ENTITY_TYPES = {"PERSON", "ORG", "CONCEPT", "PLACE", "EVENT", "THING"}

EXTRACTION_SYSTEM = """You are a knowledge graph extraction engine.
Given a text passage, extract entities and relationships.
Return ONLY valid JSON — no explanation, no markdown fences.

Output format:
{
  "entities": [
    {"name": "...", "type": "PERSON|ORG|CONCEPT|PLACE|EVENT|THING", "description": "one sentence"}
  ],
  "relations": [
    {"subject": "entity name", "predicate": "verb phrase", "object": "entity name"}
  ]
}

Rules:
- Extract 3-10 entities per chunk. Quality over quantity.
- Only include relations between entities you listed.
- Use consistent entity names (full names, not pronouns).
- type must be one of: PERSON, ORG, CONCEPT, PLACE, EVENT, THING
"""


def _has_enough_entities(text: str) -> bool:
    """Quick spaCy check — skip chunks with no recognisable entities."""
    doc = _nlp(text[:1000])  # only check first 1000 chars for speed
    return len(doc.ents) >= 1


def _safe_parse(raw: str) -> dict:
    """Strip markdown fences if the model hallucinated them, then parse JSON."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("extraction.json_parse_failed", error=str(e), raw=raw[:200])
        return {"entities": [], "relations": []}


def _validate(data: dict) -> tuple[list[dict], list[dict]]:
    entities = []
    for e in data.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        etype = str(e.get("type", "THING")).upper()
        if not name:
            continue
        if etype not in ENTITY_TYPES:
            etype = "THING"
        entities.append({
            "name": name,
            "type": etype,
            "description": str(e.get("description", ""))[:300],
        })

    entity_names = {e["name"].lower() for e in entities}
    relations = []
    for r in data.get("relations", []):
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subject", "")).strip()
        pred = str(r.get("predicate", "")).strip()
        obj = str(r.get("object", "")).strip()
        if not all([subj, pred, obj]):
            continue
        # Only keep relations where both entities were extracted
        if subj.lower() in entity_names and obj.lower() in entity_names:
            relations.append({"subject": subj, "predicate": pred, "object": obj})

    return entities, relations


async def extract_from_chunk(text: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (entities, relations).
    entities  = [{"name", "type", "description"}]
    relations = [{"subject", "predicate", "object"}]
    """
    if not _has_enough_entities(text):
        log.debug("extraction.skipped", reason="no_spacy_entities")
        return [], []

    prompt = f"Extract entities and relationships from this text:\n\n{text}"
    raw = await chat_json(prompt=prompt, system=EXTRACTION_SYSTEM)
    data = _safe_parse(raw)
    entities, relations = _validate(data)

    log.info(
        "extraction.done",
        entities=len(entities),
        relations=len(relations),
        text_len=len(text),
    )
    return entities, relations
