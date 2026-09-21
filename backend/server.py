"""
LiveKit Agent Server and RTC Session Lifecycle Orchestrator.
Manages room connections, voice pipeline startup, greeting utterance, and clean session shutdown.
"""

import json
import uuid
from aiohttp import web

from livekit import agents
from livekit.agents import AgentServer, room_io
from livekit.agents.worker import http_server as _http_server_module

from agent import create_multi_agent_system, log_session_start
from config import settings
from voice import create_voice_session, prewarm_voice_pipeline

# ── Custom HTTP routes for health checks and multi-user room creation ──

async def _health_handler(request: web.Request) -> web.Response:
    """Simple health check endpoint."""
    return web.json_response({"status": "ok", "service": "portfolio-backend"})

async def _create_room_handler(request: web.Request) -> web.Response:
    """Generate a unique room UUID for multi-user session isolation."""
    room_id = str(uuid.uuid4())
    return web.json_response({"room": room_id})

# ── Monkey-patch HttpServer to inject our custom routes ────────────────────
# The AgentServer creates its HttpServer inside run(). By patching HttpServer.__init__
# before AgentServer.run() is called, our routes get registered on the internal
# aiohttp app before the HTTP server starts (on the exposed port 10000).

_original_http_server_init = _http_server_module.HttpServer.__init__

def _patched_http_server_init(self, *args, **kwargs):
    _original_http_server_init(self, *args, **kwargs)
    # The HttpServer creates its internal aiohttp Application as self.app
    if hasattr(self, 'app') and hasattr(self.app, 'router'):
        self.app.router.add_get("/health", _health_handler)
        self.app.router.add_post("/create_room", _create_room_handler)
        print("--> [Server] Injected /health and /create_room routes into AgentServer HTTP interface.")

# Apply the monkey-patch before AgentServer is instantiated
_http_server_module.HttpServer.__init__ = _patched_http_server_init

# Configure AgentServer with thread executor, generous init timeout, and prewarm routine
server = AgentServer(
    port=settings.PORT,
    host=settings.HOST,
    load_threshold=float("inf"),
    load_fnc=lambda *args: 0.0,
    num_idle_processes=1,
    job_executor_type=agents.JobExecutorType.THREAD,
    initialize_process_timeout=120.0,
    shutdown_process_timeout=30.0,
    setup_fnc=prewarm_voice_pipeline,
)

# No separate aiohttp health server needed — routes are injected into AgentServer's HTTP interface
# which runs on the exposed Render port (10000).


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext) -> None:
    """Entry point for each incoming WebRTC voice session."""
    # 1. Connect immediately so LiveKit signals to browser that agent joined (<50ms)
    await ctx.connect()
    print("--> [Server] Worker connected to LiveKit room.")

    # 2. Instantiate isolated voice session with fresh STT, TTS, and dual-LLM pipeline
    session = create_voice_session(ctx)
    session_id = getattr(ctx.room, "name", "portfolio_session")

    # 3. Instantiate Multi-Agent Specialist Cluster (Greeter, Research, Engineering, Booking)
    greeter, userdata = create_multi_agent_system(
        session_id=session_id,
        get_room=lambda: ctx.room,
        get_session=lambda: session,
    )
    session.userdata = userdata

    # Async non-blocking session recording to Supabase
    log_session_start(
        session_id=session_id,
        visitor_name="Anonymous Visitor",
        initial_screen="/",
    )

    @ctx.room.on("data_received")
    def on_data_received(packet) -> None:
        topic = getattr(packet, "topic", None)
        if topic not in ("client_context", "page_context"):
            return
        try:
            payload = json.loads(packet.data.decode("utf-8"))
            if payload.get("type") == "page_context" or "pathname" in payload:
                pathname = (payload.get("pathname") or "/").strip()
                title = payload.get("title", "")
                userdata.active_screen = pathname
                userdata.active_title = title
                userdata.screen_context = payload
                print(f"--> [Server] Visitor page context: {pathname} (title: {title})")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
            print(f"--> [Server Warning] Invalid client page context: {error}")

    await session.start(
        room=ctx.room,
        agent=greeter,
        room_options=room_io.RoomOptions(
            # Disable local Rust AudioProcessingModule (auto_gain_control=False).
            # LiveKit's default auto_gain_control=True instantiates rtc.AudioProcessingModule,
            # which synchronously calls Rust FFI apm.process_stream() on EVERY audio frame,
            # blocking the event loop for ~182ms on Render 0.1 vCPU and causing VAD delays.
            # Audio quality is already handled in browser WebRTC (AEC/AGC/NS) and Deepgram nova-3.
            audio_input=room_io.AudioInputOptions(
                auto_gain_control=False,
                noise_cancellation=None,
            ),
            text_output=room_io.TextOutputOptions(
                sync_transcription=False,
            ),
        ),
    )
    print("-->[Server] Assistant session started. Audio: WebRTC browser-side + Deepgram nova-3.")


    # 4. Inactivity & Lifecycle Watchdog (user_away_timeout)
    idle_disconnect_task: asyncio.Task | None = None

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        nonlocal idle_disconnect_task
        state = getattr(ev, "new_state", None)
        if state == "away":
            print("--> [Server Watchdog] Visitor away detected. Sending check-in prompt.")
            session.say(
                "Still there? Let me know if you'd like to explore any of Jithendra's research papers, engineering projects, or resume.",
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )

            async def _idle_shutdown():
                try:
                    await asyncio.sleep(settings.IDLE_DISCONNECT_TIMEOUT)
                    print("--> [Server Watchdog] Inactivity timeout reached without reply. Gracefully shutting down.")
                    ctx.shutdown(reason="inactivity timeout")
                except asyncio.CancelledError:
                    pass

            if idle_disconnect_task is None or idle_disconnect_task.done():
                idle_disconnect_task = asyncio.create_task(_idle_shutdown())
        elif state in ("speaking", "listening"):
            if idle_disconnect_task and not idle_disconnect_task.done():
                print("--> [Server Watchdog] Visitor activity resumed. Resetting idle watchdog.")
                idle_disconnect_task.cancel()
                idle_disconnect_task = None

    # 5. Cleanly shutdown the job runner as soon as the user disconnects
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        if len(ctx.room.remote_participants) == 0:
            print("--> [Server] Remote participant left room, shutting down job cleanly.")
            if idle_disconnect_task and not idle_disconnect_task.done():
                idle_disconnect_task.cancel()
            ctx.shutdown(reason="remote participant left")