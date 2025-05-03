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
# AWS Lambda Deployment Tasks
# ==============================================================================

lambda_layer:
	@echo "Creating Lambda layer with dependencies..."

	# Create a temporary directory for building the layer
	mkdir -p build_layer/python/lib/python3.12/site-packages

	# Export dependencies to requirements.txt
	poetry export -f requirements.txt --output requirements.txt --without-hashes

	# Use Docker to build the layer in an Amazon Linux environment
	docker run --rm \
		-v "$(PWD)/requirements.txt:/tmp/requirements.txt" \
		-v "$(PWD)/build_layer/python/lib/python3.12/site-packages:/layer" \
		public.ecr.aws/sam/build-python3.12 \
		/bin/sh -c "pip install -r /tmp/requirements.txt -t /layer"

	# Create ZIP file for the layer
	cd build_layer && zip -r ../lambda_layer.zip .

	@echo "Lambda layer created: lambda_layer.zip"
	@echo "Upload this file to AWS Lambda as a layer."

	# Clean up the build directory
	rm -rf build_layer

# ============================================================================== 
# Lambda Function Zipping Tasks
# ==============================================================================

# Directories for lambda zips
LAMBDA_SRC=lambdas
DB_DEPS=../db
CREATE_DEPS=../shared $(DB_DEPS)
GET_DEPS=../shared $(DB_DEPS)
GENERATE_DEPS=../shared ../storytelling ../utils ../services ../storage $(DB_DEPS)

zip_create_book:
	@echo "Zipping create_book lambda..."
	zip -r create_book.zip lambdas/create_book.py shared db -x "*.pyc" -x "__pycache__/*"
	zip -j create_book.zip settings.py
	@echo "Created create_book.zip"

zip_generate_book:
	@echo "Zipping generate_book lambda..."
	zip -r generate_book.zip lambdas/generate_book.py shared storytelling utils services storage db -x "*.pyc" -x "__pycache__/*"
	zip -j generate_book.zip settings.py
	@echo "Created generate_book.zip"

zip_get_book:
	@echo "Zipping get_book lambda..."
	zip -r get_book.zip lambdas/get_book.py shared db -x "*.pyc" -x "__pycache__/*"
	zip -j get_book.zip settings.py
	@echo "Created get_book.zip"

zip_all_lambdas: zip_create_book zip_generate_book zip_get_book
	@echo "All lambda functions zipped."

# ==============================================================================
# Cleanup Tasks
# ==============================================================================

clean:
	@echo "Cleaning up build artifacts and caches..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

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
	@echo "  make lambda_layer - Create a ZIP file with dependencies for AWS Lambda layer."

# Default target (runs when typing just 'make')
default: help
