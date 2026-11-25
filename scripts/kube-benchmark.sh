#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "$#" in
    0) job= ;;
    2)
        if [[ "$1" != --collect || ! "$2" =~ ^nlp-ranking-benchmark-[a-z0-9]+$ ]]; then
            printf 'Usage: %s [--collect JOB]\n' "$0" >&2
            exit 2
        fi
        job="$2"
        ;;
    *) printf 'Usage: %s [--collect JOB]\n' "$0" >&2; exit 2 ;;
esac

for tool in docker kind kubectl; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Install %s before running the benchmark.\n' "$tool" >&2
        exit 1
    fi
done
if [[ ! -x .venv/bin/python ]]; then
    echo 'Create the project virtual environment and install the release requirements first.' >&2
    exit 1
fi
docker info --format '{{.ServerVersion}}' >/dev/null

benchmark_dir="$(mktemp -d)"
trap 'rm -rf -- "$benchmark_dir"' EXIT
export KUBECONFIG="$benchmark_dir/kubeconfig"
export KIND_EXPERIMENTAL_PROVIDER=docker
cluster=nlp-release-bench
image=nlp-release-benchmark:local

create_job() {
    .venv/bin/python - "$benchmark_dir/inputs" <<'PY'
import sys
from pathlib import Path
from shutil import copyfile

from serve_ranker import bundle_manifest

inputs = Path(sys.argv[1])
bundle = Path("artifacts/esci-tinybert-v1-onnx")
manifest, _ = bundle_manifest(bundle)
snapshot = Path("data/esci-us-small")

(inputs / "bundle").mkdir(parents=True)
for name in (*manifest["files"], "manifest.json"):
    copyfile(bundle / name, inputs / "bundle" / name)

(inputs / "snapshot").mkdir()
for name in ("manifest.json", "validation.parquet"):
    copyfile(snapshot / name, inputs / "snapshot" / name)
PY

    docker build --target benchmark -f deploy/Dockerfile \
        --build-context "benchmark_inputs=$benchmark_dir/inputs" -t "$image" .

    local clusters
    clusters="$(kind get clusters)"
    if [[ $'\n'"$clusters"$'\n' == *$'\n'"$cluster"$'\n'* ]]; then
        kind export kubeconfig --name "$cluster"
    else
        kind create cluster --name "$cluster" --wait 120s
    fi
    kind load docker-image "$image" --name "$cluster"
    job="$(kubectl create -f deploy/benchmark-job.yaml -o jsonpath='{.metadata.name}')"
}

if [[ -n "$job" ]]; then
    kind export kubeconfig --name "$cluster"
else
    create_job
fi

mkdir -p artifacts
log="artifacts/$job.log"
report="artifacts/$job.json"
printf 'Waiting for %s. Logs will be saved to %s.\n' "$job" "$log"

deadline=$((SECONDS + 330))
job_state=
while (( SECONDS < deadline )); do
    conditions="$(kubectl get "job/$job" --request-timeout=10s \
        -o 'jsonpath={range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}')"
    case $'\n'"$conditions"$'\n' in
        *$'\nComplete\n'*) job_state=complete; break ;;
        *$'\nFailed\n'*) job_state=failed; break ;;
    esac
    sleep 2
done

if [[ -z "$job_state" ]]; then
    kubectl describe "job/$job" >"$log" 2>&1 || true
    kubectl describe pods -l "batch.kubernetes.io/job-name=$job" >>"$log" 2>&1 || true
    printf 'Timed out waiting for the Job. See %s.\n' "$log" >&2
    exit 1
fi

if ! kubectl logs --request-timeout=20s "job/$job" >"$benchmark_dir/job.log" 2>&1; then
    cat "$benchmark_dir/job.log" >&2
    printf 'Could not collect the Job logs. See %s.\n' "$log" >&2
    kubectl describe "job/$job" >"$log" 2>&1 || true
    kubectl describe pods -l "batch.kubernetes.io/job-name=$job" >>"$log" 2>&1 || true
    exit 1
fi
mv -- "$benchmark_dir/job.log" "$log"

.venv/bin/python - "$log" "$report" <<'PY'
import json
import sys
from pathlib import Path

log, output = map(Path, sys.argv[1:])
prefix = "BENCHMARK_REPORT "
lines = log.read_text(encoding="utf-8").splitlines()
reports = [line[len(prefix):] for line in lines if line.startswith(prefix)]
if len(reports) != 1:
    raise SystemExit(f"Expected one benchmark report. See {log}.")

report = json.loads(reports[0])
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if report["status"] != "passed":
    raise SystemExit(f"Benchmark failed. See {output}.")

print(f"Median HTTP p95 across runs: {report['http']['median_http_p95_ms']:.2f} ms")
PY
if [[ "$job_state" != complete ]]; then
    printf 'Kubernetes marked %s as failed. See %s.\n' "$job" "$log" >&2
    exit 1
fi
printf 'Job %s completed.\n' "$job"
printf 'Saved benchmark report to %s\n' "$report"
