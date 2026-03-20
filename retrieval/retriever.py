"""
Hybrid retriever
────────────────
1. Embed the question → find top-k similar chunks via pgvector
2. Extract seed entity names from those chunks → look them up in Neo4j
3. Traverse the graph N hops → collect neighbourhood context
4. Assemble final context window: chunks + graph triples
5. Call LLM for answer with cited sources
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from retrieval.embedder import embed_one
from graph.neo4j_client import GraphDB
from graph.llm_client import chat
from config import get_settings

log = structlog.get_logger()
settings = get_settings()

QA_SYSTEM = """You are a precise knowledge assistant with access to a personal knowledge graph.
Answer the question using ONLY the context provided.
At the end of your answer, list the sources you used as "Sources: [source1, source2, ...]".
If the context doesn't contain enough information, say so clearly.
"""


async def retrieve_chunks(
    question: str,
    db: AsyncSession,
    top_k: int | None = None,
) -> list[dict]:
    """Vector similarity search over DocumentChunk."""
    k = top_k or settings.top_k_vector
    vector = embed_one(question)

    result = await db.execute(
        text("""
            SELECT id, source_id, source_type, content, metadata,
                   1 - (embedding <=> :vec::vector) AS score
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :vec::vector
            LIMIT :k
        """),
        {"vec": str(vector), "k": k},
    )
    rows = result.mappings().all()
    return [dict(r) for r in rows]


def _extract_entity_names(chunks: list[dict]) -> list[str]:
    """
    Simple heuristic: capitalised 2-3 word sequences in chunks.
    A full approach would re-run spaCy here — this is the fast path.
    """
    import re
    names = set()
    for chunk in chunks:
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", chunk["content"]):
            names.add(match.group(1))
    return list(names)[:20]


def _names_to_entity_ids(names: list[str]) -> list[str]:
    """Convert entity names to Neo4j id format."""
    return [f"person:{n.lower().replace(' ', '_')}" for n in names] + \
           [f"org:{n.lower().replace(' ', '_')}" for n in names] + \
           [f"concept:{n.lower().replace(' ', '_')}" for n in names]


async def answer(
    question: str,
    db: AsyncSession,
    graph: GraphDB,
) -> dict:
    """
    Full retrieval + generation pipeline.
    Returns {"answer": str, "sources": list[str], "chunks": list, "graph_triples": list}
    """
    # Step 1: vector retrieval
    chunks = await retrieve_chunks(question, db)
    log.info("retrieval.chunks", count=len(chunks))

    # Step 2: graph retrieval
    entity_names = _extract_entity_names(chunks)
    entity_ids = _names_to_entity_ids(entity_names)
    graph_triples = []
    if entity_ids:
        graph_triples = await graph.get_neighbourhood(
            entity_ids, hops=settings.top_k_graph_hops
        )
    log.info("retrieval.graph_triples", count=len(graph_triples))

    # Step 3: build context
    chunk_context = "\n\n---\n\n".join(
        f"[Source: {c['source_id']}]\n{c['content']}" for c in chunks
    )
    graph_context = ""
    if graph_triples:
        lines = [
            f"{t['from_name']} ({t['from_type']}) --[{t['predicate']}]--> {t['to_name']} ({t['to_type']})"
            for t in graph_triples
        ]
        graph_context = "Knowledge graph context:\n" + "\n".join(lines)

    full_context = f"{chunk_context}\n\n{graph_context}".strip()

    prompt = f"""Context:
{full_context}

Question: {question}

Answer:"""

    answer_text = await chat(prompt=prompt, system=QA_SYSTEM)
    sources = list({c["source_id"] for c in chunks})

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(chunks),
        "graph_triples_used": len(graph_triples),
    }
