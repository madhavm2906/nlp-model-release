import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from serve_ranker import bundle_manifest

LOCK_KEY = 8064707121


def connect():
    url = os.environ.get("RELEASE_DATABASE_URL")
    if not url:
        raise ValueError("Set RELEASE_DATABASE_URL before using the release database.")
    return psycopg.connect(url, connect_timeout=3, options="-c statement_timeout=4000", row_factory=dict_row)


def lock_releases(db):
    db.execute("SELECT pg_advisory_xact_lock(%s)", [LOCK_KEY])


def latest(db):
    return db.execute("""
        SELECT id, action, version, bundle, prior_version, prior_bundle, recorded_at
        FROM release_events ORDER BY id DESC LIMIT 1
    """).fetchone()


def active_release(db):
    row = latest(db)
    if row is None:
        raise ValueError("Initialize a model release first.")
    return row


def record(db, action, version, bundle, prior=None, evidence=None):
    previous = (prior["version"], prior["bundle"]) if prior is not None else (None, None)
    db.execute("""
        INSERT INTO release_events (action, version, bundle, prior_version, prior_bundle, evidence)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (action, version, str(bundle), *previous, Jsonb(evidence) if evidence is not None else None))
    return active_release(db)


def initialize(db, bundle, *, keep_existing=False):
    bundle = bundle.resolve()
    lock_releases(db)
    db.execute("""
        CREATE TABLE IF NOT EXISTS release_events (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            action text NOT NULL CHECK (action IN ('init', 'promote', 'rollback')),
            version text NOT NULL,
            bundle text NOT NULL,
            prior_version text,
            prior_bundle text,
            evidence jsonb,
            recorded_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    current = latest(db)
    if current is not None and keep_existing:
        return current
    _, version = bundle_manifest(bundle)
    if current is not None:
        if current["version"] != version:
            raise ValueError("A different model is already active. Use a release report to promote.")
        return current
    return record(db, "init", version, bundle)


def rollback(db):
    lock_releases(db)
    current = active_release(db)
    if current["action"] == "rollback":
        return current
    if current["prior_bundle"] is None:
        raise ValueError("There is no previous release to restore.")
    bundle = Path(current["prior_bundle"])
    _, version = bundle_manifest(bundle)
    if version != current["prior_version"]:
        raise ValueError("Previous bundle no longer matches the recorded version.")
    return record(db, "rollback", version, bundle, current)
