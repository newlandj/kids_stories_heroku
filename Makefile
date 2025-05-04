# Makefile for kids-story-aws project

# Define the default shell
SHELL := /bin/bash

# Prevent commands from being echoed
.SILENT:

# Define phony targets (targets that don't represent files)
.PHONY: format test_local check install clean lambda_layer

# ==============================================================================
# Development Tasks
# ==============================================================================

install:
	@echo "Installing dependencies using Poetry..."
	poetry install

format:
	@echo "Formatting code with Ruff..."
	poetry run ruff format .
	@echo "Linting and auto-fixing code with Ruff..."
	poetry run ruff check . --fix

check:
	@echo "Checking code style with Ruff (no changes)..."
	poetry run ruff check .

test_local:
	@echo "Running local test script..."
	poetry run python -m tests.local_test_script

# ==============================================================================
# Cleanup Tasks
# ==============================================================================

clean:
	@echo "Cleaning up build artifacts and caches..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

# ============================================================================== 
# Dev Environment (App + Redis + Celery)
# ==============================================================================

dev:
	@echo "Starting Redis, FastAPI app, and Celery worker in separate terminals..."
	@echo "If you have tmux, use: make dev-tmux"
	@echo "Otherwise, open three terminals and run:"
	@echo "  1. redis-server"
	@echo "  2. poetry run uvicorn app.main:app --reload"
	@echo "  3. poetry run celery -A app.celery_worker.celery_app worker --loglevel=info"

# Run all dev services in tmux panes (if tmux is installed)
dev-tmux:
	tmux new-session -d -s kidsstory 'redis-server'
	tmux split-window -h -t kidsstory 'poetry run uvicorn app.main:app --reload'
	tmux split-window -v -t kidsstory:0.1 'poetry run celery -A app.celery_worker.celery_app worker --loglevel=info'
	tmux select-layout -t kidsstory tiled
	tmux attach -t kidsstory

# ============================================================================== 
# Help Target
# ==============================================================================

help:
	@echo "Available commands:"
	@echo "  make install     - Install project dependencies using Poetry."
	@echo "  make format      - Format code using Ruff and apply auto-fixes."
	@echo "  make check       - Check code style using Ruff (read-only)."
	@echo "  make test_local  - Run the local test script (tests/local_test_script.py)."
	@echo "  make clean       - Remove temporary Python files and build artifacts."
	@echo "  make dev         - Print instructions to start Redis, FastAPI, and Celery."
	@echo "  make dev-tmux    - Start all dev services in a tmux session (if installed)."

# Default target (runs when typing just 'make')
default: help
