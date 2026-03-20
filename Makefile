.PHONY: up down build test lint shell neo4j-shell

up:
	docker compose up --build -d

down:
	docker compose down -v

build:
	docker compose build

logs:
	docker compose logs -f api worker

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check . --fix

# Open a Python shell inside the running API container
shell:
	docker compose exec api python

# Open Neo4j cypher-shell
neo4j-shell:
	docker compose exec neo4j cypher-shell -u neo4j -p memgraph_secret

# Quick smoke test — ingest a URL and ask a question
smoke:
	@echo "Ingesting Wikipedia article..."
	@JOB=$$(curl -s -X POST http://localhost:8000/ingest/url \
		-H "Content-Type: application/json" \
		-d '{"url":"https://en.wikipedia.org/wiki/Knowledge_graph"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])"); \
	echo "Job ID: $$JOB"; \
	echo "Waiting 30s for ingestion..."; \
	sleep 30; \
	echo "Status:"; \
	curl -s http://localhost:8000/ingest/status/$$JOB | python3 -m json.tool; \
	echo "\nAsking question..."; \
	curl -s -X POST http://localhost:8000/ask \
		-H "Content-Type: application/json" \
		-d '{"question":"What is a knowledge graph used for?"}' | python3 -m json.tool
