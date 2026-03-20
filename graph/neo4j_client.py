from neo4j import AsyncGraphDatabase, AsyncDriver
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import Any
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()


class GraphDB:
    """
    Thin async wrapper around Neo4j.

    Node schema
    -----------
    (:Entity {
        id,           # slug: "person:alice_smith"
        name,         # "Alice Smith"
        type,         # PERSON | ORG | CONCEPT | PLACE | EVENT | THING
        description,
        importance_score,   # float 0-1, decays over time
        access_count,
        created_at,
        last_accessed_at,
        source_ids    # list of DocumentChunk source_ids
    })

    Relationship schema
    -------------------
    (:Entity)-[:RELATION {
        predicate,    # "works_at", "mentioned_with", etc.
        weight,       # float, reinforced on re-extraction
        created_at,
        source_id
    }]->(:Entity)
    """

    _driver: AsyncDriver | None = None

    @classmethod
    async def connect(cls) -> "GraphDB":
        cls._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await cls._driver.verify_connectivity()
        log.info("neo4j.connected", uri=settings.neo4j_uri)
        return cls()

    @classmethod
    async def close(cls):
        if cls._driver:
            await cls._driver.close()

    @asynccontextmanager
    async def session(self):
        async with self._driver.session(database="neo4j") as s:
            yield s

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    async def init_schema(self):
        async with self.session() as s:
            await s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            await s.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            await s.run("CREATE INDEX entity_importance IF NOT EXISTS FOR (e:Entity) ON (e.importance_score)")
        log.info("neo4j.schema_initialized")

    # ------------------------------------------------------------------
    # Upsert entity (merge on id, reinforce importance if already exists)
    # ------------------------------------------------------------------
    async def upsert_entity(self, entity: dict[str, Any], source_id: str) -> str:
        entity_id = f"{entity['type'].lower()}:{entity['name'].lower().replace(' ', '_')}"
        now = datetime.now(UTC).isoformat()

        async with self.session() as s:
            await s.run(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.description = $description,
                    e.importance_score = 1.0,
                    e.access_count = 0,
                    e.created_at = $now,
                    e.last_accessed_at = $now,
                    e.source_ids = [$source_id]
                ON MATCH SET
                    e.importance_score = min(1.0, e.importance_score + 0.1),
                    e.access_count = e.access_count + 1,
                    e.last_accessed_at = $now,
                    e.source_ids = CASE
                        WHEN NOT $source_id IN e.source_ids
                        THEN e.source_ids + [$source_id]
                        ELSE e.source_ids
                    END
                """,
                id=entity_id,
                name=entity["name"],
                type=entity["type"].upper(),
                description=entity.get("description", ""),
                now=now,
                source_id=source_id,
            )
        return entity_id

    # ------------------------------------------------------------------
    # Upsert relation between two entities
    # ------------------------------------------------------------------
    async def upsert_relation(
        self, from_id: str, to_id: str, predicate: str, source_id: str
    ):
        now = datetime.now(UTC).isoformat()
        async with self.session() as s:
            await s.run(
                """
                MATCH (a:Entity {id: $from_id}), (b:Entity {id: $to_id})
                MERGE (a)-[r:RELATION {predicate: $predicate}]->(b)
                ON CREATE SET r.weight = 1.0, r.created_at = $now, r.source_id = $source_id
                ON MATCH SET  r.weight = r.weight + 0.5
                """,
                from_id=from_id,
                to_id=to_id,
                predicate=predicate,
                now=now,
                source_id=source_id,
            )

    # ------------------------------------------------------------------
    # Retrieve neighbourhood for retrieval
    # ------------------------------------------------------------------
    async def get_neighbourhood(self, entity_ids: list[str], hops: int = 2) -> list[dict]:
        """Return all entities and relations within N hops of the seed entities."""
        async with self.session() as s:
            result = await s.run(
                """
                MATCH path = (seed:Entity)-[:RELATION*1..$hops]-(neighbour:Entity)
                WHERE seed.id IN $ids
                WITH nodes(path) AS ns, relationships(path) AS rs
                UNWIND rs AS r
                RETURN
                    startNode(r).name AS from_name,
                    startNode(r).type AS from_type,
                    r.predicate       AS predicate,
                    endNode(r).name   AS to_name,
                    endNode(r).type   AS to_type,
                    r.weight          AS weight
                ORDER BY r.weight DESC
                LIMIT 60
                """,
                ids=entity_ids,
                hops=hops,
            )
            return [dict(record) async for record in result]

    # ------------------------------------------------------------------
    # Memory decay — called nightly by Celery beat
    # ------------------------------------------------------------------
    async def apply_decay(self, half_life_days: float, min_score: float):
        """
        Exponential decay: score *= 0.5^(days_since_access / half_life)
        Nodes below min_score are detached and deleted.
        """
        now = datetime.now(UTC).isoformat()
        async with self.session() as s:
            await s.run(
                """
                MATCH (e:Entity)
                WITH e,
                     duration.between(datetime(e.last_accessed_at), datetime($now)).days AS days_old
                SET e.importance_score = e.importance_score * (0.5 ^ (toFloat(days_old) / $half_life))
                """,
                now=now,
                half_life=half_life_days,
            )
            result = await s.run(
                """
                MATCH (e:Entity)
                WHERE e.importance_score < $min_score
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                min_score=min_score,
            )
            record = await result.single()
            deleted = record["deleted"] if record else 0
        log.info("neo4j.decay_applied", deleted=deleted)
        return deleted

    # ------------------------------------------------------------------
    # Top entities by importance (for dashboard)
    # ------------------------------------------------------------------
    async def top_entities(self, limit: int = 20) -> list[dict]:
        async with self.session() as s:
            result = await s.run(
                "MATCH (e:Entity) RETURN e ORDER BY e.importance_score DESC LIMIT $limit",
                limit=limit,
            )
            return [dict(record["e"]) async for record in result]
