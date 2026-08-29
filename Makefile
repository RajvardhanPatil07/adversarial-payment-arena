# Adversarial Payment Arena -- reproducible evidence pipeline.
#
# Every headline claim in this repository is produced by one of these targets
# and lands in artifacts/ as JSON with a provenance stamp (git sha, seeds,
# python version, command). If a number is not reachable from `make reproduce`,
# it should not be claimed.

PY ?= python
BACKEND := backend
ARTIFACTS := artifacts

.DEFAULT_GOAL := help
.PHONY: help install reproduce calibration fidelity transfer zero-day artifacts clean-artifacts models serve ui test check closed-loop coverage latency

help: ## Show available targets
	@echo "Adversarial Payment Arena -- make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install backend dependencies
	$(PY) -m pip install -r $(BACKEND)/requirements.txt

models: ## Train + serialize the defense models (xgb + iForest) into backend/models/
	$(PY) $(BACKEND)/data/corpus_builder.py

reproduce: calibration fidelity transfer closed-loop coverage latency ## Regenerate the complete evidence set
	@echo ""
	@echo "evidence set regenerated:"
	@ls -1 $(ARTIFACTS)/*.json 2>/dev/null || echo "  (none -- a stage failed)"

calibration: ## Threshold provenance + prevalence + INR cost sweep
	$(PY) $(BACKEND)/experiments/run_calibration_audit.py

fidelity: ## Five fidelity measures per attack generator
	$(PY) $(BACKEND)/experiments/run_fidelity.py

transfer: ## HEADLINE: three-arm fidelity-vs-transfer ablation
	$(PY) $(BACKEND)/experiments/run_transfer_ablation.py

closed-loop: ## HEADLINE: gated vs ungated closed loop (the fidelity scissor)
	$(PY) $(BACKEND)/experiments/run_closed_loop.py

coverage: ## Per-family recall + leave-one-family-out zero-day + layer attribution
	$(PY) $(BACKEND)/experiments/run_family_coverage.py

latency: ## Measured inline decision latency percentiles (p50/p95/p99)
	$(PY) $(BACKEND)/experiments/run_latency.py

zero-day: ## Existing zero-day holdout experiment
	$(PY) $(BACKEND)/experiments/zero_day_holdout.py

artifacts: ## List the current evidence set
	@ls -1 $(ARTIFACTS)/*.json 2>/dev/null || echo "no artifacts yet -- run 'make reproduce'"

clean-artifacts: ## Delete generated artifacts (forces a real regeneration)
	rm -f $(ARTIFACTS)/*.json

serve: ## Run the backend API
	cd $(BACKEND) && uvicorn main:app --reload --port 8000

ui: ## Run the prototype UI
	cd frontend && pnpm dev

test: ## Run the test suite
	cd $(BACKEND) && pytest tests/ -q

check: ## Byte-compile every backend module (fast sanity check)
	$(PY) -m compileall -q $(BACKEND) && echo "compile OK"
