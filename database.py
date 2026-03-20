from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime, UTC
from config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    """Raw document chunks with embeddings — the flat RAG layer."""
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(String(255), nullable=False, index=True)   # filename / URL
    source_type = Column(String(50), nullable=False)              # pdf, web, email, docx
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(384))                               # all-MiniLM-L6-v2 dim
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def __repr__(self):
        return f"<DocumentChunk {self.source_id}[{self.chunk_index}]>"


class IngestionJob(Base):
    """Tracks async ingestion tasks."""
    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celery_task_id = Column(String(255), unique=True, index=True)
    source = Column(String(1024), nullable=False)
    source_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")   # pending / running / done / failed
    error = Column(Text, nullable=True)
    chunks_created = Column(Integer, default=0)
    entities_extracted = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    finished_at = Column(DateTime(timezone=True), nullable=True)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Create tables and pgvector extension."""
    async with engine.begin() as conn:
        await conn.execute(__import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector"))
        await Base.metadata.create_all(conn)
