from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import uuid
import tempfile
import os
import structlog

from database import init_db, get_db, IngestionJob
from graph.neo4j_client import GraphDB
from retrieval.retriever import answer
from workers.celery_app import ingest_source_task

log = structlog.get_logger()
_graph: GraphDB | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    await init_db()
    _graph = await GraphDB.connect()
    await _graph.init_schema()
    log.info("app.started")
    yield
    await GraphDB.close()
    log.info("app.shutdown")


app = FastAPI(title="MemGraph API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_graph() -> GraphDB:
    return _graph


# ── Models ────────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    chunks_created: int
    entities_extracted: int
    error: str | None


# ── Ingestion routes ──────────────────────────────────────────────────────────

@app.post("/ingest/url", status_code=202)
async def ingest_url(req: IngestURLRequest, db: AsyncSession = Depends(get_db)):
    """Kick off async ingestion of a web URL."""
    job = IngestionJob(source=req.url, source_type="web")
    db.add(job)
    await db.commit()
    await db.refresh(job)

    ingest_source_task.delay(req.url, str(job.id))
    return {"job_id": str(job.id), "status": "queued"}


@app.post("/ingest/file", status_code=202)
async def ingest_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a file (PDF, DOCX, TXT, MD) for ingestion."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save to temp file — Celery worker needs a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job = IngestionJob(source=file.filename, source_type=ext.lstrip("."))
    db.add(job)
    await db.commit()
    await db.refresh(job)

    ingest_source_task.delay(tmp_path, str(job.id))
    return {"job_id": str(job.id), "status": "queued", "filename": file.filename}


@app.get("/ingest/status/{job_id}", response_model=JobStatusResponse)
async def ingest_status(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == uuid.UUID(job_id)))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        chunks_created=job.chunks_created,
        entities_extracted=job.entities_extracted,
        error=job.error,
    )


# ── QA route ──────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(
    req: QuestionRequest,
    db: AsyncSession = Depends(get_db),
    graph: GraphDB = Depends(get_graph),
):
    """Answer a question using hybrid graph-RAG."""
    result = await answer(question=req.question, db=db, graph=graph)
    return result


# ── Graph exploration ─────────────────────────────────────────────────────────

@app.get("/graph/top-entities")
async def top_entities(limit: int = 20, graph: GraphDB = Depends(get_graph)):
    """Return top entities by importance score (for the graph explorer UI)."""
    entities = await graph.top_entities(limit=limit)
    return {"entities": entities}


@app.get("/graph/neighbourhood")
async def neighbourhood(
    entity_id: str,
    hops: int = 2,
    graph: GraphDB = Depends(get_graph),
):
    """Return the neighbourhood of a specific entity."""
    triples = await graph.get_neighbourhood([entity_id], hops=hops)
    return {"entity_id": entity_id, "triples": triples}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
