.PHONY: test coverage lint clean

test:
	uv run pytest

coverage:
	uv run pytest --cov=. --cov-report=html --cov-report=xml

lint:
	uvx flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv
	uvx flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=.venv

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
