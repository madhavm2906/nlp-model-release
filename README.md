# NLP Inference and Model Release Platform

I am building a system that can train a model, serve it, and decide when a new version is ready to ship. The model scores how well a product matches a shopping query. The project starts with a keyword baseline so every model has a score to beat.

## Run the baseline

Use Python 3.12 and curl from the project folder. The first download is about 1.2 GB.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python esci.py download
.venv/bin/python esci.py prepare
.venv/bin/python baseline.py
```

The scripts fetch a fixed copy of the [Amazon ESCI data](https://github.com/amazon-science/esci-data) and keep the US part of the small set. Queries with the same normalized text stay together. BM25 learns word frequencies from training product titles and scores the labeled candidates on validation. The test split is saved for a later release evaluation.

The dataset summary is in `data/esci-us-small/manifest.json`. The validation score is in `artifacts/bm25-validation.json`. Use nDCG at 10 to see whether the most relevant products are near the top.

The first run scored **0.683883 nDCG at 10** across 3,071 validation queries. That is the score for the next model to beat.

## What comes next

Train a compact model on the same data and measure what improves. Then add a Go service that batches requests, tracks latency, and checks a candidate model before it replaces the current version.
