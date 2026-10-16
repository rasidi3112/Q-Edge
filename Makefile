# Reproduce every number in docs/results.md and README.md.
# `make reproduce` took about 62 minutes on a laptop CPU, 55 of them for the
# ZNE runs of experiments/quantum_head.py. Needs q-edge[fl,report] and, for
# the secagg benchmark as configured, liboqs.

PYTHON ?= python

.PHONY: reproduce experiments report test lint

reproduce: experiments report

experiments:
	$(PYTHON) -m experiments.run_all

report:
	$(PYTHON) -m experiments.report

test:
	$(PYTHON) -m pytest -m "not slow"

lint:
	ruff check .
	ruff format --check .
	mypy src/qedge
