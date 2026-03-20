import pytest
from unittest.mock import AsyncMock, patch
from extraction.extractor import _safe_parse, _validate, extract_from_chunk


def test_safe_parse_clean_json():
    raw = '{"entities": [], "relations": []}'
    assert _safe_parse(raw) == {"entities": [], "relations": []}


def test_safe_parse_strips_markdown_fences():
    raw = '```json\n{"entities": [], "relations": []}\n```'
    assert _safe_parse(raw) == {"entities": [], "relations": []}


def test_safe_parse_bad_json_returns_empty():
    raw = "not valid json at all"
    result = _safe_parse(raw)
    assert result == {"entities": [], "relations": []}


def test_validate_filters_invalid_type():
    data = {
        "entities": [
            {"name": "Alice", "type": "WIZARD", "description": "test"},
            {"name": "Acme", "type": "ORG", "description": "a company"},
        ],
        "relations": []
    }
    entities, _ = _validate(data)
    types = {e["type"] for e in entities}
    assert "WIZARD" not in types
    assert "THING" in types  # fallback


def test_validate_drops_relations_with_missing_entities():
    data = {
        "entities": [
            {"name": "Alice", "type": "PERSON", "description": ""},
        ],
        "relations": [
            {"subject": "Alice", "predicate": "works_at", "object": "Acme"},  # Acme not in entities
        ]
    }
    _, relations = _validate(data)
    assert relations == []


@pytest.mark.asyncio
async def test_extract_from_chunk_returns_structured():
    mock_response = '''{"entities": [{"name": "Alice", "type": "PERSON", "description": "An engineer"}], "relations": []}'''
    with patch("extraction.extractor.chat_json", new_callable=AsyncMock, return_value=mock_response):
        entities, relations = await extract_from_chunk("Alice is a software engineer at Acme Corp.")
    assert any(e["name"] == "Alice" for e in entities)
