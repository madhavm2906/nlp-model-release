#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-start}"
case "$action" in
    start|status|logs|stop) ;;
    *) printf '%s\n' 'Usage: bash scripts/release.sh {start|status|logs|stop}' >&2; exit 2 ;;
esac

if [[ "$action" == start ]]; then
    mkdir -p artifacts
    if [[ ! -s artifacts/release-db-password ]]; then
        (umask 077; python3 -c 'import secrets; print(secrets.token_hex(24))' > artifacts/release-db-password)
    fi
fi
RELEASE_ARTIFACTS_DIR="$(realpath artifacts)"
RELEASE_DB_PASSWORD="$(cat artifacts/release-db-password)"
export RELEASE_ARTIFACTS_DIR RELEASE_DB_PASSWORD
compose=(docker compose -f compose.release.yaml -f compose.dashboard.yaml)

case "$action" in
    start)
        "${compose[@]}" up -d --build --wait --wait-timeout 90
        curl -fsS -o /dev/null http://127.0.0.1:8084/api/release
        printf '\nRelease dashboard ready at http://127.0.0.1:8084/\n'
        ;;
    status) exec "${compose[@]}" ps --all ;;
    logs) exec "${compose[@]}" logs --tail 50 ;;
    stop) exec "${compose[@]}" stop ;;
esac
