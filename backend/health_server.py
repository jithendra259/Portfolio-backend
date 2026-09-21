"""Simple health‑check HTTP server using aiohttp.

This runs alongside the LiveKit ``AgentServer`` without pulling in FastAPI.
It listens on a configurable host/port (default ``0.0.0.0:8001``) and
responds to ``GET /health`` with a tiny JSON payload.  The endpoint can be
used by external monitors (cron‑job.org, GitHub Actions, etc.) to verify the
service is up.
"""

from __future__ import annotations

import asyncio
from aiohttp import web

async def _health(_: web.Request) -> web.Response:
    """Return a minimal JSON health payload.

    The response body is only a few bytes, keeping external monitors well
    under any size limits.
    """
    return web.json_response({"status": "ok"})


async def start_health_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Create and start the aiohttp health server.

    The function returns only when the server has been started; it does not
    block the event loop.  Call it with ``await`` from the main async entry
    point (e.g., ``server.py``) to run the health endpoint concurrently with
    the LiveKit ``AgentServer``.
    """
    app = web.Application()
    app.router.add_get("/health", _health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    # Keep a reference so the runner is not garbage‑collected.
    # The server will continue running for the lifetime of the process.
    return None
