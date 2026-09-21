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
import uuid

async def _health(_: web.Request) -> web.Response:
    """Return a minimal JSON health payload.

    The response body is only a few bytes, keeping external monitors well
    under any size limits.
    """
    return web.json_response({"status": "ok"})


async def _create_room(request: web.Request) -> web.Response:
    """Create a new unique room identifier for a user.

    The client can POST to ``/create_room`` (no body required) and receive a
    JSON payload like ``{"room": "<uuid>"}``.  The LiveKit client can then join
    that room using the normal LiveKit SDK.  This keeps the backend stateless
    and works for any number of concurrent users.
    """
    room_id = str(uuid.uuid4())
    return web.json_response({"room": room_id})


async def start_health_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Create and start the aiohttp health server.

    The function returns only when the server has been started; it does not
    block the event loop.  Call it with ``await`` from the main async entry
    point (e.g., ``server.py``) to run the health endpoint concurrently with
    the LiveKit ``AgentServer``.
    """
    app = web.Application()
    app.router.add_get("/health", _health)
    app.router.add_post("/create_room", _create_room)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    # Keep a reference so the runner is not garbage‑collected.
    # The server will continue running for the lifetime of the process.
    return None
