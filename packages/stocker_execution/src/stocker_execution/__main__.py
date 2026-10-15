"""`stocker paper-run`: the CFD and QQQ paper record and its dashboard. It never places orders."""

import argparse
import asyncio
import logging
from pathlib import Path


def paper_run(config: Path, database: Path, host: str, port: int) -> None:
    import uvicorn

    from stocker_dashboard.app import create_dashboard_app
    from stocker_execution.config import load
    from stocker_execution.runtime import LOG_FORMAT, Runtime
    from stocker_execution.store import Store

    # Runtime errors reach the service journal as well as the rotating runtime log.
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

    async def serve() -> None:
        runtime = Runtime(load(config), Store(database))
        app = create_dashboard_app(runtime)
        # Security authenticates the actual loopback proxy peer, not X-Forwarded-For.
        server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, proxy_headers=False, access_log=False)
        )
        worker = asyncio.create_task(runtime.run())

        async def dashboard() -> None:
            runtime.web_health = "RUNNING"
            try:
                await server.serve()
                if not server.started:
                    raise RuntimeError("Dashboard stopped before startup completed")
                if not server.should_exit:
                    raise RuntimeError("Dashboard stopped unexpectedly")
            except (Exception, SystemExit) as exc:
                runtime.web_health = "FAILED"
                runtime.report_failure("dashboard", exc)
                await asyncio.Event().wait()

        web = asyncio.create_task(dashboard())
        try:
            done, _ = await asyncio.wait({worker, web}, return_when=asyncio.FIRST_COMPLETED)
            worker_error = worker.exception() if worker in done and not worker.cancelled() else None
            if worker in done and (worker_error is not None or web not in done):
                runtime.worker_health = "FAILED"
                error = worker_error or RuntimeError("paper record worker terminated unexpectedly")
                runtime.report_failure("worker", error)
                server.should_exit = True
                raise error
            await web
        finally:
            await runtime.stop()
            for task in (worker, web):
                task.cancel()
            await asyncio.gather(worker, web, return_exceptions=True)

    asyncio.run(serve())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stocker", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("paper-run", help="run the paper record and its dashboard")
    run.add_argument("--config", type=Path, default=Path("configs/paper.example.yaml"))
    run.add_argument("--database", type=Path, default=Path(".stocker/paper-record.sqlite3"))
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args(argv)
    paper_run(arguments.config, arguments.database, arguments.host, arguments.port)


if __name__ == "__main__":
    main()
