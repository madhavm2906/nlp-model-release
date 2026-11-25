import json
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from bench_gateway import run

ROOT = Path(__file__).resolve().parent
THREADS, REQUESTS, CLIENTS, REPEATS = 4, 160, 16, 3
WORKER_PORT, GATEWAY_PORT = 18001, 18082
MAX_NDCG_DROP = 0.003
MAX_LATENCY_RATIO = 1.10


def decision(current, candidate):
    reasons = []
    drop = current["ndcg_at_10"] - candidate["ndcg_at_10"]
    if drop > MAX_NDCG_DROP:
        reasons.append(f"nDCG fell by {drop:.6f}; limit is {MAX_NDCG_DROP:.6f}.")

    for name, row in (("Current", current), ("Candidate", candidate)):
        reasons.extend(f"{name}: {problem}" for problem in row["http"]["problems"])

    current_ms = current["http"]["median_http_p95_ms"]
    candidate_ms = candidate["http"]["median_http_p95_ms"]
    if current_ms is None or candidate_ms is None:
        reasons.append("Both versions need successful HTTP replays.")
    elif candidate_ms > current_ms * MAX_LATENCY_RATIO:
        reasons.append(
            f"Candidate p95 {candidate_ms:.2f} ms exceeds "
            f"{current_ms * MAX_LATENCY_RATIO:.2f} ms."
        )
    return {"status": "blocked" if reasons else "eligible", "reasons": reasons}


def check_ports():
    for port in (WORKER_PORT, GATEWAY_PORT):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ValueError(f"Release replay port {port} is already in use.") from exc


@contextmanager
def server(command, health, expected_version=None):
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        deadline = monotonic() + 15
        while monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited before startup: {health}")
            try:
                with urlopen(health, timeout=0.5) as response:
                    status = json.load(response)
            except (URLError, TimeoutError):
                sleep(0.05)
                continue

            if expected_version is not None and status.get("model_version") != expected_version:
                raise RuntimeError(f"Worker at {health} loaded an unexpected model.")
            break
        else:
            raise RuntimeError(f"Server was not ready after 15 seconds: {health}")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def replay(binary, bundle, version, payloads):
    check_ports()
    worker_url = f"http://127.0.0.1:{WORKER_PORT}"
    gateway_url = f"http://127.0.0.1:{GATEWAY_PORT}"
    worker = [
        sys.executable, "serve_ranker.py", "--bundle", str(bundle.resolve()),
        "--port", str(WORKER_PORT), "--threads", str(THREADS),
    ]
    gateway = [
        str(binary), "-listen", f"127.0.0.1:{GATEWAY_PORT}",
        "-worker", worker_url + "/score",
    ]

    runs = []
    problems = []
    with server(worker, worker_url + "/healthz", version):
        with server(gateway, gateway_url + "/healthz"):
            for _ in range(REPEATS):
                try:
                    row = run(gateway_url + "/rank", payloads, REQUESTS, CLIENTS)
                except ValueError as exc:
                    problems.append(str(exc))
                    continue
                runs.append(row)
                if row["status_counts"] != {"200": REQUESTS}:
                    problems.append(f"Replay returned {row['status_counts']}.")
                if row.get("model_versions") != [version]:
                    problems.append("Replay responses did not all identify the expected model.")

    return {
        "runs": runs,
        "problems": problems,
        "median_http_p95_ms": None if problems else median(row["request_p95_ms"] for row in runs),
    }
