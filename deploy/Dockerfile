FROM python:3.12-slim AS worker
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt requirements-serving.txt requirements-release.txt ./
RUN python -m pip install --no-cache-dir -r requirements-release.txt
COPY release_dashboard.py release_store.py serve_release.py serve_ranker.py ./

FROM golang:1.25-alpine AS go-build
WORKDIR /src
COPY go.mod ./
COPY cmd/gateway/ ./cmd/gateway/
RUN CGO_ENABLED=0 go build -buildvcs=false -o /gateway ./cmd/gateway

FROM alpine:3.22 AS gateway
COPY --from=go-build /gateway /gateway
USER 65532:65532
ENTRYPOINT ["/gateway"]

FROM worker AS benchmark
COPY --from=go-build /gateway /usr/local/bin/gateway
COPY benchmark_job.py bench_gateway.py release_replay.py esci.py esci_source.json ./
COPY --from=benchmark_inputs --chown=65532:65532 . /inputs/
USER 65532:65532
ENTRYPOINT ["python", "benchmark_job.py", "--gateway", "/usr/local/bin/gateway", "--bundle", "/inputs/bundle", "--snapshot", "/inputs/snapshot"]

FROM node:24-bookworm-slim AS frontend-build
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM nginx:1.28-alpine AS dashboard
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /src/dist/ /usr/share/nginx/html/
