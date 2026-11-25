# NLP Inference and Model Release Platform

I am building a system that can train a model, serve it, and decide when a new version is ready to ship. The model scores how well a product matches a shopping query. The project starts with a keyword baseline so every model has a score to beat.

## Run the baseline

Use Python 3.12 and curl from the project folder. The first download is about 1.2 GB.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p test_ranking.py -v
.venv/bin/python esci.py download
.venv/bin/python esci.py prepare
.venv/bin/python baseline.py
```

The scripts fetch a fixed copy of the [Amazon ESCI data](https://github.com/amazon-science/esci-data) and keep the US part of the small set. Queries with the same normalized text stay together. BM25 learns word frequencies from training product titles and scores the labeled candidates on validation. The test split is saved for a later release evaluation.

The dataset summary is in `data/esci-us-small/manifest.json`. The validation score is in `artifacts/bm25-validation.json`. Use nDCG at 10 to see whether the most relevant products are near the top.

The first run scored **0.683883 nDCG at 10** across 3,071 validation queries. That is the score for the next model to beat.

## Train the ranker

The ranker reads the query and product title together. It starts from a small [TinyBERT ranking model](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2). During training it sees two titles for the same query and learns to give the more relevant one a higher score.

```bash
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-ranker.txt
.venv/bin/python -m unittest discover -s tests -p 'test*rank*.py' -v
HF_HUB_DISABLE_XET=1 .venv/bin/python -u train_ranker.py --device cpu
```

Look in `artifacts/esci-tinybert-v1` for the saved model and `metrics.json`. The report compares BM25, the pretrained model, and the tuned model on the same validation queries. It also counts how many queries improved or got worse.

On the first CPU run, the pretrained model scored **0.706055 nDCG at 10**. One epoch on 15,000 comparisons raised it to **0.717436**. The final test results are still pending.

## Export and measure inference

```bash
.venv/bin/python -m pip install -r requirements-onnx.txt
.venv/bin/python -m unittest discover -s tests -p test_onnx.py -v
.venv/bin/python -u export_onnx.py
```

The export saves the model and tokenizer in `artifacts/esci-tinybert-v1-onnx`. It checks the scores and ranking quality against PyTorch on the full validation split. The benchmark records CPU latency and throughput for batches of 1, 8, and 32 pairs. It measures model execution after warmup. Tokenization and network time are outside that measurement.

## Serve the model

Install the worker dependencies and run the tests with Go 1.22 or newer.

```bash
.venv/bin/python -m pip install -r requirements-serving.txt
.venv/bin/python -m unittest discover -s tests -v
go test ./...
```

Start the worker in one terminal and run `go run ./cmd/gateway` in another.

```bash
.venv/bin/python serve_ranker.py
```

The gateway ranks candidates at `POST /rank` on port 8082. It combines small requests into one worker call, limits how many requests can wait, and skips expired work before scoring. The worker checks the model files when it starts. The response includes the model version, the total pairs scored in the batch, and the time spent waiting and scoring.

```bash
.venv/bin/python bench_gateway.py --requests 160 --concurrency 16
```

The benchmark uses shopping queries from validation and reports HTTP latency, queue time, throughput, and errors. To measure the same workload without batching, restart the gateway with `-batch-wait=0`.

## What comes next

Compare two model versions under the same traffic, then add the checks for promoting and rolling back a release.
