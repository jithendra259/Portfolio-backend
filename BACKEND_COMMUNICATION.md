# Backend Communication Protocol

**Service:** `portfolio-backend` (LiveKit Voice Agent Worker)  
**Deploy:** Render (HTTP port 10000) + LiveKit Cloud (WebRTC)  
**Repo:** `Portfolio-backend` (`backend/` subdirectory)  
**Entry:** `backend/server.py`

---

## 1. Architecture Overview

```
Browser (Frontend) ──w──> LiveKit Cloud (WebRTC signaling + media)
                                    │
                                    ▼
              LiveKit AgentWorker (backend/agent_worker.py)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
   server.py (RTC)       agent/ (LLM graph)        api/ (STT/TTS)
   - Room lifecycle      - 5-node graph agent    - Deepgram STT
   - Data channels       - 4 specialist agents      - ElevenLabs TTS
   - HTTP routes         - RAG (TF-IDF)              - Shared HTTP client
   - Supabase logging    - Tools (nav, theme,      - Groq LLM (via api/llm.py)
```

### Environment Variables (`config/settings.py`)

| Key | Value (production) | Purpose |
|---|---|---|
| `LIVEKIT_URL` | `wss://portfolio-jezy7ize.livekit.cloud` | LiveKit Cloud project WebSocket |
| `LIVEKIT_API_KEY` | `APIbRAUawrxisqw` | LiveKit API key |
| `LIVEKIT_API_SECRET` | (secret) | LiveKit API secret |
| `PORT` | `10000` | Render HTTP port |
| `NEXT_PUBLIC_RENDER_BACKEND_URL` | `https://portfolio-backend-ljlv.onrender.com` | Backend public URL (health ping / room creation) |
| `GROQ_API_KEY` | (secret) | Groq LLM inference |
| `ELEVENLABS_API_KEY` | (secret) | Text-to-speech |
| `SUPABASE_URL` | `https://tlyqtsbyxckovzdssdeg.supabase.co` | Analytics database |
| `SUPABASE_KEY` | (secret) | Supabase service key |
| `ENABLE_DENSE_RAG` | `false` | TF-IDF is the active retrieval algorithm |

---

## 2. HTTP Routes (`server.py`)

Routes are monkey-patched into the LiveKit `AgentServer`'s internal aiohttp app (port 10000).

| Method | Path | Response | Purpose |
|---|---|---|---|
| `GET` | `/` | `{"status":"ok","service":"portfolio-backend"}` | Health check + Render warmup |
| `GET` | `/health` | `{"status":"ok","service":"portfolio-backend"}` | Health check (duplicate) |
| `POST` | `/create_room` | `{"room":"<uuid>"}` | Generate unique room UUID for multi-user session isolation |

---

## 3. LiveKit Agent Dispatch

Registered in `server.py:67`:

```python
@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext) -> None:
```

- **Agent name:** `my-agent` (must match `AGENT_NAME` / `NEXT_PUBLIC_AGENT_NAME` in frontend env)
- When a participant joins a room, LiveKit dispatches `my-agent` automatically (explicit dispatch via frontend `token/route.ts` RoomAgentDispatch also works)

---

## 4. Data Channel Protocol

All data channel communication uses LiveKit `publishData` / `data_received` events on the room's `localParticipant`.

### 4.1 Frontend → Backend

| Topic | Payload Schema | Handled In |
|---|---|---|
| `client_context` | `{ "type": "page_context", "pathname": string, "title": string, "url": string, "hash": string, "timestamp": number }` | `server.py:93-108` (on_data_received) |
| `page_context` | Same payload | `server.py:93-108` (same handler) |

Backend handler (`server.py:94-108`):
- Filters: must have `topic in ("client_context", "page_context")`
- Parses JSON, checks `payload.get("type") == "page_context" or "pathname" in payload`
- Stores into `userdata.active_screen` (pathname), `userdata.active_title`, `userdata.screen_context` (full payload)

The frontend re-publishes on both topics (`client_context` + `page_context`) immediately on session start, again after 600ms, and whenever it receives a `query_page` ping from the backend.

### 4.2 Backend → Frontend

| Topic | Payload | Published By | Handled In (Frontend) |
|---|---|---|---|
| `navigation` | `{ "type": "navigate", "target": "<target>" }` or raw string | `tools.py:17-18`, `booking.py:139-140` | `useVoiceAutoNavigation.ts:411-427` |
| `assistant_action` | `{ "type": "download", "url": string, "filename": string }` | `tools.py:181-182` | `useVoiceAutoNavigation.ts:384-392` |
| `assistant_action` | `{ "type": "theme", "theme": "dark" \| "light" }` | `tools.py:208-212` | `useVoiceAutoNavigation.ts:393-396` |
| `assistant_action` | `{ "type": "booking_confirmed", "booking": {...} }` | `booking.py:108-112` | `useVoiceAutoNavigation.ts:397-403` |
| `query_page` | (any) | — | `app.tsx:194` (listens, re-publishes on `client_context` + `page_context`) |

**Note on `query_page`:** The backend does NOT publish to `query_page`. Instead, the `get_current_page_context` LLM tool reads cached `userdata.screen_context` directly. The frontend's `query_page` listener is a safety-net fallback that is effectively unused.

---

## 5. LLM Graph & Agent Orchestration

### Graph Nodes (`agent/graph/graph.py`)
1. Entry → 2. Research Reasoner → 3. Engineering Analysis → 4. Greeter Handoff → 5. Exit

### Specialist Agents (`agent/specialists/`)
| Agent | File | Responsibilities |
|---|---|---|
| Greeter | `greeter.py` | First contact, intent routing, transfers to Research/Engineering/Booking |
| Research Specialist | `research.py` | Paper deep-dives, TF-IDF knowledge search, mathematical formulations |
| Engineering Specialist | `engineering.py` | Technical architecture explanations, project demos |
| Booking Specialist | `booking.py` | Recruiter scheduling, lead capture, Supabase persistence |

### Toolsets (`agent/tools.py`)
- `NavigationToolset` — publishes `navigation` topic (targets: `contact`, `skills`, `projects`, `research`, etc.)
- `ResearchToolset` — triggers `ResearchReasoner` / TF-IDF search
- `ResourceToolset` — publishes `assistant_action` with `type: "download"` (resume, research, certificates, AQI report, swarm report)
- `ThemeToolset` — publishes `assistant_action` with `type: "theme"` (`dark` or `light`)
- `BookingSpecialist.confirm_booking` — publishes `assistant_action` with `type: "booking_confirmed"` + calls `supabase_logger.log_booking_lead`

---

## 6. RAG / Knowledge Base

- **Algorithm:** TF-IDF (sparse retrieval only when `ENABLE_DENSE_RAG=false`)
- **Source:** `agent/rag/retriever.py`, `chunker.py`, `corpus.py`, `pdf_loader.py`
- **Cache:** `cache/chunks.json` (378 chunks from 6 PDFs + 3 extracted transcripts)
- **Vector store:** `cache/vector_store.npz` (TF-IDF sparse matrix)
- **Search function:** `search_knowledge_base(query, top_k=...)`

---

## 7. Supabase Integration

All persistence is non-blocking via `asyncio.to_thread` / `asyncio.create_task`.

| Table | Purpose | Schema (partial) | Writer |
|---|---|---|---|
| `portfolio_sessions` | Session start | `session_id, visitor_name, initial_screen, created_at, metadata` | `supabase_logger.py:72` (fallback: `sessions`) |
| `portfolio_turns` | Conversation turns | `session_id, role, content, latency_ms, route, target_screen, active_agent, created_at` | `supabase_logger.py:126` (fallback: `turns`) |
| `portfolio_leads` | Booking leads | `visitor_name, email, topic, preferred_date, preferred_time, notes, session_id, created_at` | `supabase_logger.py:191` (fallback: `bookings` → `leads`) |

**Schema note:** The frontend's `schedule-appointment/route.ts` writes to a `bookings` table directly (with `meet_url` column), while the backend's `supabase_logger.log_booking_lead` writes to `portfolio_leads` (with `session_id` column) with a fallback chain: `portfolio_leads` → `bookings` → `leads`. Column schemas differ slightly between the two writers.

---

## 8. Voice Pipeline

### WebRTC Session (`server.py:67-127`)
- Connects immediately (`ctx.connect()`) for <50ms join
- Audio processing: browser-side WebRTC AEC/AGC/NS + Deepgram nova-3 STT (no Rust APM to avoid 182ms event-loop blocking)
- Idle watchdog: 60-second away check-in → `IDLE_DISCONNECT_TIMEOUT` (default 300s) inactivity shutdown
- Clean shutdown on `participant_disconnected` (no remote participants)

### Voice Session (`voice/session.py`)
- STT: Deepgram (nova-3)
- TTS: ElevenLabs
- LLM: Dual-LLM pipeline (primary + response policy guardrail)
- VAD: Silero
- Turn detection: GPT-Paced

---

## 9. Deployment Notes

- **Backend:** Render (static service on port 10000) — must stay warm via frontend health-ping loop
- **Frontend:** Vercel (Next.js 15)
- **LiveKit Cloud:** `wss://portfolio-jezy7ize.livekit.cloud`
- **Render URL:** `https://portfolio-backend-ljlv.onrender.com` (used in `NEXT_PUBLIC_RENDER_BACKEND_URL`)
