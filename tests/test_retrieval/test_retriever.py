import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_answer_returns_required_keys():
    """answer() must always return answer, sources, chunks_used, graph_triples_used."""
    from retrieval.retriever import answer

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(
        mappings=MagicMock(return_value=MagicMock(
            all=MagicMock(return_value=[
                {"source_id": "test.pdf", "source_type": "pdf",
                 "content": "Alice works at Acme Corp.",
                 "metadata": {}, "score": 0.9}
            ])
        ))
    ))

    mock_graph = AsyncMock()
    mock_graph.get_neighbourhood = AsyncMock(return_value=[])

    with patch("retrieval.retriever.embed_one", return_value=[0.1] * 384), \
         patch("retrieval.retriever.chat", new_callable=AsyncMock, return_value="Alice works at Acme Corp. Sources: [test.pdf]"):

        result = await answer("Where does Alice work?", db=mock_db, graph=mock_graph)

    assert "answer" in result
    assert "sources" in result
    assert "chunks_used" in result
    assert "graph_triples_used" in result
    assert result["chunks_used"] == 1


@pytest.mark.asyncio
async def test_answer_with_graph_triples():
    """Graph triples should be appended to context."""
    from retrieval.retriever import answer

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(
        mappings=MagicMock(return_value=MagicMock(
            all=MagicMock(return_value=[
                {"source_id": "notes.md", "source_type": "text",
                 "content": "Bob manages the infra team.",
                 "metadata": {}, "score": 0.85}
            ])
        ))
    ))

    mock_graph = AsyncMock()
    mock_graph.get_neighbourhood = AsyncMock(return_value=[
        {"from_name": "Bob", "from_type": "PERSON",
         "predicate": "manages",
         "to_name": "Infra Team", "to_type": "ORG",
         "weight": 1.5}
    ])

    with patch("retrieval.retriever.embed_one", return_value=[0.1] * 384), \
         patch("retrieval.retriever.chat", new_callable=AsyncMock, return_value="Bob manages the infra team."):

        result = await answer("Who manages infra?", db=mock_db, graph=mock_graph)

    assert result["graph_triples_used"] == 1
