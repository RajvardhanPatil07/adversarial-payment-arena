# Adversarial Payment Arena -- reproducible evidence pipeline.
#
# Every headline claim in this repository is produced by one of these targets
# and lands in artifacts/ as JSON with a provenance stamp. If a number is not
# reachable from `make reproduce`, it should not be claimed.

PY ?= python
BACKEND := backend
ARTIFACTS := artifacts

.DEFAULT_GOAL := help
.PHONY: help install reproduce calibration fidelity transfer behavioural privacy policy smoke-evidence zero-day artifacts clean-artifacts serve ui test check

help: ## Show available targets
	@echo "Adversarial Payment Arena -- make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install backend dependencies
	$(PY) -m pip install -r $(BACKEND)/requirements.txt

reproduce: calibration fidelity transfer behavioural privacy policy ## Regenerate the complete evidence set
	@echo ""
	@echo "evidence set regenerated:"
	@ls -1 $(ARTIFACTS)/*.json 2>/dev/null || echo "  (none -- a stage failed)"

calibration: ## Threshold provenance + prevalence + INR cost sweep
	$(PY) $(BACKEND)/experiments/run_calibration_audit.py

fidelity: ## Five fidelity measures per attack generator
	$(PY) $(BACKEND)/experiments/run_fidelity.py

transfer: ## HEADLINE: three-arm fidelity-vs-transfer ablation
	$(PY) $(BACKEND)/experiments/run_transfer_ablation.py

behavioural: ## Held-out temporal/graph fidelity for lightweight generators
	$(PY) $(BACKEND)/experiments/run_behavioural_fidelity.py

privacy: ## Membership/duplication/attribute-inference audit
	$(PY) $(BACKEND)/experiments/run_privacy_audit.py

policy: ## Validation-selected four-action policy, frozen on held-out test traffic
	$(PY) $(BACKEND)/experiments/run_action_policy.py

smoke-evidence: ## Fast deterministic regression run for new evidence modules
	$(PY) $(BACKEND)/tests/run_tests.py
	$(PY) $(BACKEND)/experiments/run_behavioural_fidelity.py --smoke
	$(PY) $(BACKEND)/experiments/run_privacy_audit.py --smoke
	$(PY) $(BACKEND)/experiments/run_action_policy.py --smoke

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
