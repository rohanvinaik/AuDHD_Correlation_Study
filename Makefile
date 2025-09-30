.PHONY: help install dev-install test lint format typecheck clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install        Install package and dependencies"
	@echo "  dev-install    Install package with dev dependencies"
	@echo "  test           Run tests with coverage"
	@echo "  lint           Run linting (ruff)"
	@echo "  format         Format code (black + ruff)"
	@echo "  typecheck      Run type checking (mypy)"
	@echo "  clean          Clean build artifacts and caches"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,notebooks]"
	pre-commit install

test:
	pytest -v --cov=src/audhd_correlation --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t audhd-correlation:latest -f docker/Dockerfile .

docker-run:
	docker run -it --rm -v $(PWD)/data:/workspace/data audhd-correlation:latest

# Pipeline commands (using CLI)
download-data:
	audhd-omics download

build-features:
	audhd-omics build-features

integrate:
	audhd-omics integrate

cluster:
	audhd-omics cluster

validate:
	audhd-omics validate

report:
	audhd-omics report

run-pipeline:
	audhd-omics pipeline