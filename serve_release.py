import argparse
import logging
from pathlib import Path
from threading import Event, Lock, Thread

import uvicorn

from release_dashboard import dashboard_router
from release_store import active_release, connect, initialize
from serve_ranker import Ranker, create_app

logger = logging.getLogger("uvicorn.error")


def desired_release():
    with connect() as db:
        return active_release(db)


def load_ranker(release, threads):
    ranker = Ranker(Path(release["bundle"]), threads)
    if ranker.version != release["version"]:
        raise ValueError("Bundle no longer matches the recorded release version.")
    return ranker


class LiveRanker:
    def __init__(self, threads):
        self.threads = threads
        self.lock = Lock()
        self.stopped = Event()
        self.release = desired_release()
        self.ranker = load_ranker(self.release, threads)
        self.thread = Thread(target=self.poll, daemon=True)

    def current(self):
        with self.lock:
            return self.ranker

    def poll(self):
        while not self.stopped.wait(1):
            try:
                release = desired_release()
                if release["id"] == self.release["id"]:
                    continue
                ranker = load_ranker(release, self.threads)
                if desired_release()["id"] != release["id"]:
                    continue
                with self.lock:
                    self.release, self.ranker = release, ranker
                logger.info("Serving release %s (%s)", release["id"], ranker.version)
            except Exception as exc:
                logger.warning("Release refresh failed; keeping the loaded model: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Serve the current PostgreSQL model release.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--bootstrap-bundle", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.bootstrap_bundle is not None:
        with connect() as db:
            initialize(db, args.bootstrap_bundle, keep_existing=True)
    live = LiveRanker(args.threads)
    app = create_app(live.current)
    app.include_router(dashboard_router(args.artifacts, live.current))
    live.thread.start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    finally:
        live.stopped.set()
        live.thread.join(timeout=2)


if __name__ == "__main__":
    main()
