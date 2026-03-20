from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Free models on OpenRouter — swap as needed
    extraction_model: str = "mistralai/mistral-7b-instruct:free"
    qa_model: str = "mistralai/mistral-7b-instruct:free"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # local

    # Postgres
    database_url: str = "postgresql+asyncpg://memgraph:memgraph_secret@localhost:5432/memgraph"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memgraph_secret"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Memory decay
    decay_half_life_days: float = 30.0   # importance halves every 30 days
    min_importance_score: float = 0.05   # prune nodes below this

    # Retrieval
    top_k_vector: int = 5
    top_k_graph_hops: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
