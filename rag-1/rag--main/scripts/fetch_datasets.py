#!/usr/bin/env python
"""Download and cache benchmark datasets from Hugging Face (needs internet + `pip install datasets`).

    python scripts/fetch_datasets.py squad msmarco beir:scifact beir:nfcorpus --max-queries 2000

Cached under data/lab/*.json.gz; then run e.g. `python -m rag_lab run --dataset squad`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["fetch", *sys.argv[1:]]))
