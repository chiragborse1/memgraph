"""
Celery workers
──────────────
- ingest_source_task: full pipeline for one source (parse → chunk → embed → extract → graph)
- apply_decay_task:   nightly memory decay job
"""

import asyncio
from datetime import datetime, UTC
from celery import Celery
from celery.schedules import crontab
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()

celery_app = Celery(
    "memgraph",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    beat_schedule={
        "nightly-decay": {
            "task": "workers.celery_app.apply_decay_task",
            "schedule": crontab(hour=2, minute=0),
        }
    },
)


def _run(coro):
    """Run an async coroutine from a sync Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_source_task(self, source: str, job_id: str):
    """
    Full ingestion pipeline for a single source.
    source can be a file path or URL.
    """
    from ingestion.connectors.parsers import ingest_source
    from ingestion.chunking.splitter import chunk_text
    from retrieval.embedder import embed
    from extraction.extractor import extract_from_chunk
    from graph.neo4j_client import GraphDB
    from database import AsyncSessionLocal, DocumentChunk, IngestionJob
    from sqlalchemy import select, update
    import uuid

    async def _pipeline():
        async with AsyncSessionLocal() as db:
            # Mark job as running
            await db.execute(
                update(IngestionJob)
                .where(IngestionJob.id == uuid.UUID(job_id))
                .values(status="running", celery_task_id=self.request.id)
            )
            await db.commit()

            try:
                # 1. Parse
                source_id, source_type, text, metadata = await ingest_source(source)
                log.info("ingest.parsed", source_id=source_id, chars=len(text))

                # 2. Chunk
                chunks = chunk_text(text, source_id=source_id, source_type=source_type, metadata=metadata)
                log.info("ingest.chunked", chunks=len(chunks))

                # 3. Embed all chunks in one batch
                embeddings = embed([c.content for c in chunks])

                # 4. Persist chunks to Postgres
                db_chunks = []
                for chunk, vector in zip(chunks, embeddings):
                    db_chunk = DocumentChunk(
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        content=chunk.content,
                        chunk_index=chunk.index,
                        embedding=vector,
                        metadata_=chunk.metadata,
                    )
                    db.add(db_chunk)
                    db_chunks.append(db_chunk)
                await db.commit()
                log.info("ingest.chunks_saved", count=len(db_chunks))

                # 5. Extract entities + relations from each chunk → push to Neo4j
                graph = await GraphDB.connect()
                total_entities = 0

                for chunk in chunks:
                    entities, relations = await extract_from_chunk(chunk.content)
                    total_entities += len(entities)

                    # Upsert entities
                    entity_id_map: dict[str, str] = {}
                    for entity in entities:
                        eid = await graph.upsert_entity(entity, source_id=source_id)
                        entity_id_map[entity["name"].lower()] = eid

                    # Upsert relations
                    for rel in relations:
                        from_id = entity_id_map.get(rel["subject"].lower())
                        to_id = entity_id_map.get(rel["object"].lower())
                        if from_id and to_id:
                            await graph.upsert_relation(
                                from_id, to_id, rel["predicate"], source_id=source_id
                            )

                # 6. Mark job done
                await db.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == uuid.UUID(job_id))
                    .values(
                        status="done",
                        chunks_created=len(chunks),
                        entities_extracted=total_entities,
                        finished_at=datetime.now(UTC),
                    )
                )
                await db.commit()
                log.info("ingest.complete", source_id=source_id, entities=total_entities)

            except Exception as exc:
                log.error("ingest.failed", error=str(exc))
                await db.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == uuid.UUID(job_id))
                    .values(status="failed", error=str(exc), finished_at=datetime.now(UTC))
                )
                await db.commit()
                raise self.retry(exc=exc)

    _run(_pipeline())


@celery_app.task
def apply_decay_task():
    from graph.neo4j_client import GraphDB

    async def _decay():
        graph = await GraphDB.connect()
        deleted = await graph.apply_decay(
            half_life_days=settings.decay_half_life_days,
            min_score=settings.min_importance_score,
        )
        log.info("decay.complete", deleted=deleted)

    _run(_decay())
