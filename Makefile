.PHONY: install install-dev test test-unit test-integration lint typecheck fmt clean docker-build docker-run

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src/scout_ai --cov-report=term-missing

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

docker-build:
	docker build -f docker/Dockerfile -t scout-ai:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d
