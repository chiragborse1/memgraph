import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from graph.neo4j_client import GraphDB


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.run = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_upsert_entity_returns_id():
    graph = GraphDB()
    with patch.object(graph, "session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(run=AsyncMock()))
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        entity = {"name": "Alice Smith", "type": "PERSON", "description": "An engineer"}
        eid = await graph.upsert_entity(entity, source_id="doc.pdf")
        assert eid == "person:alice_smith"


@pytest.mark.asyncio
async def test_entity_id_format():
    graph = GraphDB()
    with patch.object(graph, "session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(run=AsyncMock()))
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        entity = {"name": "OpenAI", "type": "ORG", "description": "AI company"}
        eid = await graph.upsert_entity(entity, source_id="doc.pdf")
        assert eid == "org:openai"
        assert ":" in eid
        assert " " not in eid
