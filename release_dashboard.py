import json

import psycopg
from fastapi import APIRouter, HTTPException, Response

from release_store import connect


def read_report(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid report object: {path.name}")
    return value


def dashboard_router(artifacts, current):
    router = APIRouter()

    @router.get("/api/release")
    def summary(response: Response):
        try:
            with connect() as db:
                events = db.execute("""
                    SELECT id, action, version, prior_version, recorded_at
                    FROM release_events ORDER BY id DESC LIMIT 20
                """).fetchall()
            release_check = read_report(artifacts / "int8-release-check.json")
            heldout = read_report(artifacts / "final-test-report.json")
        except (OSError, ValueError, psycopg.Error) as exc:
            raise HTTPException(503, "Release history or reports are unavailable.") from exc
        if not events:
            raise HTTPException(503, "Initialize a model release first.")

        response.headers["Cache-Control"] = "no-store"
        return {
            "desired": events[0], "serving_version": current().version,
            "events": events, "release_check": release_check, "heldout": heldout,
        }

    return router
