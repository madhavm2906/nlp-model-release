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

On the first CPU run, the pretrained model scored **0.706055 nDCG at 10**. One epoch on 15,000 comparisons raised it to **0.717436** on validation.

## Export and measure inference

```bash
.venv/bin/python -m pip install -r requirements-onnx.txt
.venv/bin/python -m unittest discover -s tests -p test_onnx.py -v
.venv/bin/python -u export_onnx.py
```

The export saves the model and tokenizer in `artifacts/esci-tinybert-v1-onnx`. It checks the scores and ranking quality against PyTorch on the full validation split. The benchmark records CPU latency and throughput for batches of 1, 8, and 32 pairs. It measures model execution after warmup. Tokenization and network time are outside that measurement.

## Run the release dashboard

After exporting the model, start the stack from the project folder with Docker Compose installed.

```bash
bash scripts/release.sh start
```

Open [the release dashboard](http://127.0.0.1:8084/). It shows the model serving requests, the candidate check, the held out results, and the release history. Use **Rank products** to send a query to the Go gateway. The gateway combines requests into batches and reports queue time and worker time separately.

Use `bash scripts/release.sh status` to check the services, `bash scripts/release.sh logs` to see recent output, and `bash scripts/release.sh stop` when you are done.

## Check and manage a release

Install the release dependencies, set the local database connection, and compare the current model with the INT8 candidate.

```bash
.venv/bin/python -m pip install -r requirements-release.txt
export RELEASE_DATABASE_URL="postgresql://ranker:$(cat artifacts/release-db-password)@127.0.0.1:5544/model_release"
.venv/bin/python make_int8.py
.venv/bin/python release_check.py
```

The check writes `artifacts/int8-release-check.json`. A candidate can advance if its ranking score falls by at most 0.003, its median HTTP p95 grows by at most 10%, and all requests succeed. On my WSL run, INT8 passed the quality check but missed the latency limit, so FP32 stayed active.

Run `model_release.py status` to see the selected version. Use `model_release.py promote --bundle PATH --report REPORT` with a passing report and `model_release.py rollback` to restore the previous version. Run these commands with `.venv/bin/python`. PostgreSQL records each change and the worker checks the selected bundle before loading it.

## Evaluate the final release

With the release database running and `RELEASE_DATABASE_URL` set, run the official ESCI test comparison.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python -u evaluate_test.py
```

The command checks the dataset and selected model, then compares the current release with BM25 and pretrained TinyBERT. It writes `artifacts/final-test-report.json` with ranking scores, how often an exact match ranks first, and a confidence interval for each gain. These results describe ranking the products provided for each query.

On my WSL run over 8,956 test queries, BM25 scored **0.690637**, pretrained TinyBERT scored **0.710407**, and the FP32 release scored **0.718221** in nDCG at 10. The release put an exact match first on **66.85%** of queries that had one.

## Run the benchmark in Kubernetes

After exporting the model and installing the release dependencies, run this with Docker, kind, and kubectl available.

```bash
bash scripts/kube-benchmark.sh
```

The script loads a local image into kind and runs the Go gateway and ONNX worker inside one Job pod. It checks the model and validation data before replaying three rounds of 160 requests. The report and logs are saved in `artifacts/` with the Job name. The timings describe requests inside that pod.

In my WSL run, all 480 measured requests succeeded. The median p95 response time across the three rounds was **71.40 ms**. To save the report from an existing Job, run `bash scripts/kube-benchmark.sh --collect JOB_NAME` with its name.
