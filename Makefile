.PHONY: setup fmt lint type test run clean docker-build docker-run

setup:
	conda env update -n audhd-omics -f env/environment.yml

fmt:
	ruff --fix .
	black src tests
	isort src tests

lint:
	flake8 src tests
	ruff check .

type:
	mypy src

test:
	pytest -q

run:
	audhd-omics pipeline --cfg configs/defaults.yaml

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete