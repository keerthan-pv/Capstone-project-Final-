# Convenience targets. Uses the in-tree .venv if present, else the active python.
#
# The eval targets are split by cost on purpose:
#   eval-retrieval  — LLM-free, graded against human labels, safe to run anytime
#   eval-full       — spends OpenRouter credits (generator + judge), run deliberately

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
API_PORT ?= 8000

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dependencies into the current environment
	$(PY) -m pip install -r requirements.txt

.PHONY: test
test: ## Run the full test suite
	$(PY) -m pytest -q

.PHONY: lint
lint: ## Run the CI lint checks (ruff)
	$(PY) -m ruff check --select E,F,W --ignore E501 .

.PHONY: eval-retrieval
eval-retrieval: ## LLM-free retrieval eval, n=500, graded vs human labels (no API cost)
	TOKENIZERS_PARALLELISM=false $(PY) main.py --mode eval --retrieval-only --llm-free --samples 500 --store full
	$(PY) analyze_results.py

.PHONY: eval-full
eval-full: ## FULL eval incl. LLM-judged metrics — SPENDS OpenRouter credits (Run 2)
	@echo ">>> This calls the generator + judge models and costs money."
	@echo ">>> Starting with a bounded sample; edit --samples to scale."
	TOKENIZERS_PARALLELISM=false $(PY) main.py --mode eval --samples 50 --store full
	$(PY) main.py --mode visualize
	$(PY) analyze_results.py

.PHONY: analyze
analyze: ## Re-run the post-hoc analysis (PΩ validity, miss rate, latency)
	$(PY) analyze_results.py

.PHONY: plots
plots: ## Regenerate figures + report from saved CSVs
	$(PY) main.py --mode visualize

.PHONY: api
api: ## Run the API locally (generates an API key if none is set)
	@API_KEY=$${API_KEY:-$$(openssl rand -hex 24)} PORT=$(API_PORT) $(PY) api.py

.PHONY: loadtest
loadtest: ## Load test a running API: make loadtest KEY=... [ARCH=naive]
	$(PY) load_test.py --url http://127.0.0.1:$(API_PORT) --key $(KEY) --arch $${ARCH:-naive} --concurrency 16 --requests 500

.PHONY: build-store
build-store: ## Build the validation vector store + synthetic dataset
	$(PY) preprocess_full.py

.PHONY: clean-pyc
clean-pyc: ## Remove caches and compiled artefacts
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

# ─── rag_lab: benchmarking / diagnosis / routing research layer ───────────────
.PHONY: lab-test lab-demo lab-fetch lab-squad lab-msmarco lab-beir lab-list
lab-test: ## Offline tests for rag_lab (no network, no LangChain needed)
	$(PY) -m unittest discover -s tests -p "test_lab_*.py" -v

lab-demo: ## Full study on the OFFLINE fixture (pipeline check, not research results)
	$(PY) -m rag_lab run --dataset fixture

lab-fetch: ## Download + cache SQuAD 2.0, MS MARCO v2.1 and BEIR/scifact (needs internet + `datasets`)
	$(PY) scripts/fetch_datasets.py squad msmarco beir:scifact --max-queries 2000

lab-squad: ## Real benchmark on SQuAD 2.0 (answers + unanswerable questions)
	$(PY) -m rag_lab run --dataset squad --max-queries 2000 --name squad-v2

lab-msmarco: ## Real benchmark on MS MARCO v2.1 (human relevance labels)
	$(PY) -m rag_lab run --dataset msmarco --max-queries 2000 --name msmarco-v2.1

lab-beir: ## Multi-domain generalisation on BEIR/scifact (retrieval-only)
	$(PY) -m rag_lab run --dataset beir:scifact --max-queries 300 --name beir-scifact

lab-list: ## List stored experiments
	$(PY) -m rag_lab list
