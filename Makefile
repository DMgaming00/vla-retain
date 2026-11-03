PYTHON ?= python

.PHONY: test lint figures figures-demo

test:
	"$(PYTHON)" -m pytest

lint:
	"$(PYTHON)" -m ruff check src tests scripts

figures:
	"$(PYTHON)" scripts/aggregate.py --results results --out paper_assets

figures-demo:
	"$(PYTHON)" scripts/aggregate.py --demo --out paper_assets/demo
