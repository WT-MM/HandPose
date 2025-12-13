.PHONY: install install-dev format lint test clean run

install:
	@if [ ! -d ".venv" ]; then uv venv; else echo "Virtual environment already exists, skipping uv venv"; fi
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

format:
	black handpose examples tests
	ruff check --fix handpose examples tests

lint:
	black --check handpose examples tests
	ruff check handpose examples tests
	mypy handpose

test:
	pytest tests

py-files := $(shell find handpose examples tests -name '*.py')

static-checks:
	@black --diff --check $(py-files)
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint

ik:
	.venv/bin/mjpython examples/live_demo_ik.py --model orca_hand_fixed.mjcf --scale 1.0

ik-dual:
	.venv/bin/mjpython examples/live_demo_ik.py --model orca_hand_fixed.mjcf --scale 1.0 --dual

manual:
	.venv/bin/mjpython examples/live_demo.py --dual

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache

SHELL := /bin/bash
PYTHON := python
PIP := python -m pip

# 1. Install prerequisites required to compile C extensions
install-build-deps:
	@echo "Installing build dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install "numpy" "cython" "torch" "torchvision"

# 2. Install xtcocotools manually without build isolation
install-xtcocotools:
	@echo "Installing xtcocotools (no build isolation)..."
	$(PIP) install xtcocotools==1.14.3 --no-build-isolation

install-detectron2:
	@echo "Installing Detectron2..."
	$(PIP) install "git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation

# 4. Install HaMeR in editable mode
install-hamer: install-build-deps install-xtcocotools install-detectron2
	@echo "Installing HaMeR..."
	cd third_party/hamer && $(PIP) install . --no-build-isolation
