"""
Supabase Async Background Logger.
Provides non-blocking, zero-latency persistence for LiveKit voice conversation turns,
routing analytics, visitor sessions, and recruiter booking leads.
All network I/O executes off the main asyncio event loop via asyncio.create_task.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from supabase import Client, create_client
    _SUPABASE_INSTALLED = True
except ImportError:
    _SUPABASE_INSTALLED = False

from config import settings

logger = logging.getLogger("supabase_logger")

_client: Optional[Any] = None


def get_supabase_client() -> Optional[Any]:
    """Returns singleton Supabase client or None if unconfigured."""
    global _client
    if _client is not None:
        return _client

    if not _SUPABASE_INSTALLED:
        logger.warning("[Supabase] supabase-py package not installed. Skipping remote logging.")
        return None

    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY
    if not url or not key:
        logger.warning("[Supabase] SUPABASE_URL or SUPABASE_KEY not configured. Skipping remote logging.")
        return None

    try:
        _client = create_client(url, key)
        logger.info("[Supabase] Client successfully initialized.")
        return _client
    except Exception as e:
        logger.warning(f"[Supabase] Failed to initialize client: {e}")
        return None


async def log_session_start_async(
    session_id: str,
    visitor_name: str = "Anonymous Visitor",
    initial_screen: str = "/",
    metadata: Optional[dict] = None,
) -> None:
    """Non-blocking record of incoming LiveKit WebRTC session."""
    client = get_supabase_client()
    if not client:
        return

    def _sync_insert():
        try:
            payload = {
                "session_id": session_id,
                "visitor_name": visitor_name,
                "initial_screen": initial_screen,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
            try:
                client.table("portfolio_sessions").insert(payload).execute()
            except Exception:
                client.table("sessions").insert(payload).execute()
        except Exception as e:
            logger.debug(f"[Supabase Session Insert Skip] {e}")

    try:
        await asyncio.to_thread(_sync_insert)
    except Exception as err:
        logger.debug(f"[Supabase Thread Error] {err}")


def log_session_start(
    session_id: str,
    visitor_name: str = "Anonymous Visitor",
    initial_screen: str = "/",
    metadata: Optional[dict] = None,
) -> None:
    """Fire-and-forget wrapper for session initialization."""
    try:
        asyncio.create_task(
            log_session_start_async(session_id, visitor_name, initial_screen, metadata)
        )
    except RuntimeError:
        pass


async def log_turn_async(
    session_id: str,
    role: str,
    content: str,
    latency_ms: float = 0.0,
    route: str = "direct",
    target_screen: str = "",
    active_agent: str = "greeter",
) -> None:
    """Non-blocking record of individual dialogue turn with latency metrics."""
    client = get_supabase_client()
    if not client:
        return

    def _sync_insert():
        try:
            payload = {
                "session_id": session_id,
                "role": role,
                "content": content[:2000] if content else "",
                "latency_ms": round(latency_ms, 2),
                "route": route,
                "target_screen": target_screen,
                "active_agent": active_agent,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                client.table("portfolio_turns").insert(payload).execute()
            except Exception:
                client.table("turns").insert(payload).execute()
        except Exception as e:
            logger.debug(f"[Supabase Turn Insert Skip] {e}")

    try:
        await asyncio.to_thread(_sync_insert)
    except Exception as err:
        logger.debug(f"[Supabase Thread Error] {err}")


def log_turn(
    session_id: str,
    role: str,
    content: str,
    latency_ms: float = 0.0,
    route: str = "direct",
    target_screen: str = "",
    active_agent: str = "greeter",
) -> None:
    """Fire-and-forget wrapper for turn logging."""
    try:
        asyncio.create_task(
            log_turn_async(
                session_id=session_id,
                role=role,
                content=content,
                latency_ms=latency_ms,
                route=route,
                target_screen=target_screen,
                active_agent=active_agent,
            )
        )
    except RuntimeError:
        pass


async def log_booking_lead_async(
    visitor_name: str,
    email: str,
    topic: str,
    preferred_date: str = "",
    preferred_time: str = "",
    notes: str = "",
    session_id: str = "",
) -> None:
    """Non-blocking record of recruiter/collaborator meeting lead."""
    client = get_supabase_client()
    if not client:
        return

    def _sync_insert():
        try:
            payload = {
                "visitor_name": visitor_name,
                "email": email,
                "topic": topic,
                "preferred_date": preferred_date,
                "preferred_time": preferred_time,
                "notes": notes,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                client.table("portfolio_leads").insert(payload).execute()
            except Exception:
                try:
                    client.table("bookings").insert(payload).execute()
                except Exception:
                    client.table("leads").insert(payload).execute()
            print(f"--> [Supabase Lead Logged] Recruiter booking: {visitor_name} ({email}) - {topic}")
        except Exception as e:
            logger.warning(f"[Supabase Lead Insert Skip] {e}")

    try:
        await asyncio.to_thread(_sync_insert)
    except Exception as err:
        logger.debug(f"[Supabase Thread Error] {err}")


def log_booking_lead(
    visitor_name: str,
    email: str,
    topic: str,
    preferred_date: str = "",
    preferred_time: str = "",
    notes: str = "",
    session_id: str = "",
) -> None:
    """Fire-and-forget wrapper for booking lead capture."""
    try:
        asyncio.create_task(
            log_booking_lead_async(
                visitor_name=visitor_name,
                email=email,
                topic=topic,
                preferred_date=preferred_date,
                preferred_time=preferred_time,
                notes=notes,
                session_id=session_id,
            )
        )
    except RuntimeError:
        pass
