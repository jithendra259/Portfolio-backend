"""
Base Specialist Agent for Multi-Agent Portfolio Architecture.
Implements context preservation, conversation history tracking, screen awareness,
non-blocking Supabase turn logging, and professional response coherence.
"""

from collections.abc import AsyncIterable
import time
from typing import Any, Callable, Optional

from livekit import rtc
from livekit.agents import (
    Agent,
    FlushSentinel,
    ModelSettings,
    StopResponse,
    UserTurnExceededEvent,
    llm,
)
from livekit.agents.voice.agent import NOT_GIVEN

from agent.graph import route_portfolio_query
from agent.supabase_logger import log_turn
from .userdata import PortfolioUserData
from prompts import (
    get_full_system_prompt,
    get_specialist_instructions,
    build_section_context_prompt,
    build_publication_context_prompt,
    build_project_context_prompt,
)


class PortfolioBaseAgent(Agent):
    """
    Base class for all portfolio specialist agents.
    Provides conversation history tracking, screen awareness, context preservation,
    and professional response coherence across turns.
    """

    def __init__(
        self,
        agent_name: str,
        instructions: str,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        self.agent_name = agent_name
        self.userdata = userdata
        self._get_room = get_room
        self.last_user_query: str = ""
        self.last_assistant_response: str = ""
        self.conversation_summary: str = ""

        super().__init__(
            instructions=instructions,
            tools=tools,
        )

    async def on_enter(self) -> None:
        """
        Lifecycle hook called upon agent handoff or initial entry.
        Executes bounded context migration and updates Supabase session state.
        Injects dynamic specialist prompt with active screen context.
        """
        print(f"--> [Agent Lifecycle] Entered '{self.agent_name}' specialist.")

        handoff = self.userdata.pending_handoff
        if handoff and handoff.target == self.agent_name:
            self.userdata.pending_handoff = None
            new_ctx = self.chat_ctx.copy()

            # Use dynamic specialist prompt with active screen context
            specialist_prompt = self.get_specialist_prompt()
            full_context = f"[Task Directive: {specialist_prompt}\n\nAddress the visitor's specific query: '{handoff.reason}'. Active screen: {handoff.active_screen}.]"

            new_ctx.add_message(role="system", content=full_context)

            if handoff.last_user_query:
                new_ctx.add_message(role="user", content=handoff.last_user_query)
            await self.update_chat_ctx(new_ctx)
        else:
            # Continuity: Carry forward relevant context from previous agent
            prev_agent = self.userdata.prev_agent
            if prev_agent and hasattr(prev_agent, "chat_ctx") and prev_agent.chat_ctx:
                try:
                    copied_ctx = prev_agent.chat_ctx.copy(
                        exclude_handoff=True,
                        exclude_config_update=True,
                        exclude_instructions=True,
                    ).truncate(max_items=4)
                    new_ctx = self.chat_ctx.copy()
                    for item in copied_ctx.items:
                        if item not in new_ctx.items:
                            new_ctx.items.append(item)
                    await self.update_chat_ctx(new_ctx)
                except Exception as err:
                    print(f"--> [Context Preservation Warning] {err}")

        self._clean_extra()

        # Async non-blocking record of agent entry to Supabase
        log_turn(
            session_id=self.userdata.session_id,
            role="system",
            content=f"Switched active agent to: {self.agent_name}",
            latency_ms=0.0,
            route="handoff",
            target_screen=self.userdata.active_screen,
            active_agent=self.agent_name,
        )

    def get_formatted_page_context(self) -> str:
        """Returns concise, 1-line summary of what the visitor is currently viewing."""
        from prompts.knowledge import PAGE_KNOWLEDGE
        path = (self.userdata.active_screen or "/").strip()
        data = PAGE_KNOWLEDGE.get(path)
        if not data:
            for k, v in PAGE_KNOWLEDGE.items():
                if k != "/" and k in path:
                    data = v
                    break
        if data:
            title = data.get("title", path)
            summary = data.get("summary", "")[:120]
            return f"Viewing '{title}' ({path}): {summary}"
        return f"Viewing page '{path}'."

    def get_dynamic_system_prompt(self) -> str:
        """
        Returns a dynamic, context-aware system prompt based on:
        - Active screen/section
        - Current specialist role
        - Recent conversation topics
        """
        active_screen = self.userdata.active_screen or "/"
        active_title = self.userdata.active_title or ""
        
        # Get recent topics from conversation summary
        recent_topics = []
        if self.conversation_summary:
            # Extract potential topics from recent exchanges
            for word in self.conversation_summary.split():
                if word.startswith("case_study_") or word in ["research", "projects", "skills", "experience"]:
                    recent_topics.append(word)
        
        # Build full dynamic prompt
        return get_full_system_prompt(
            active_screen=active_screen,
            active_title=active_title,
            recent_topics=recent_topics[-3:] if recent_topics else None
        )

    def get_specialist_prompt(self) -> str:
        """Returns specialist-specific dynamic prompt with active screen context."""
        active_screen = self.userdata.active_screen or "/"
        return get_specialist_instructions(self.agent_name, active_screen)

    def _clean_extra(self, ctx: Optional[llm.ChatContext] = None) -> None:
        """Strips extra metadata dictionaries so LiveKit serializer never produces extra_content."""
        target_ctx = ctx or self.chat_ctx
        if not target_ctx or not hasattr(target_ctx, "items"):
            return
        for item in target_ctx.items:
            if hasattr(item, "extra") and isinstance(item.extra, dict):
                item.extra.clear()

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """
        Lifecycle hook called when user finishes speaking.
        Injects screen context and routing grounding with minimal token footprint.
        """
        if not new_message.text_content or not new_message.text_content.strip():
            raise StopResponse()

        user_text = new_message.text_content.strip()
        self.last_user_query = user_text

        # LOG: User speech
        print(f"\n{'='*60}")
        print(f"👤 USER ({self.agent_name}): {user_text}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Clean any extra content from previous turns
        self._clean_extra(turn_ctx)

        # Run fast LangGraph query routing & grounding with active screen context
        result = await route_portfolio_query(
            user_text,
            screen_context=self.userdata.screen_context or {"pathname": self.userdata.active_screen},
        )
        user_intent = result.get("user_intent", "")
        user_exp = result.get("user_expectation", "")
        mode = result.get("description_mode", "short")
        grounding = result.get("grounding", "")

        if user_intent or user_exp:
            print(f"--> [{self.agent_name} Thinking] Intent: '{user_intent}' | Mode: '{mode}' | Expects: '{user_exp}'")

        # Lean context injection to keep token count strictly under Groq's 8000 TPM limit
        context_injection_parts = []

        # 1. Screen awareness
        screen_context = self.get_formatted_page_context()
        context_injection_parts.append(f"[Current Screen: {screen_context}]")

        # 2. Routing grounding (specific facts for this query, if any)
        if grounding:
            context_injection_parts.append(grounding)

        # 3. Rich case study context injection when on case study pages
        active_screen = (self.userdata.active_screen or "/").strip().lower().replace("#", "").replace("/", "")
        case_study_targets = [
            "case_study_adaptive_governance",
            "case_study_regime_supervisory",
            "case_study_supervisory_xai",
            "case_study_voice_architecture",
            "case_study_aqi",
            "case_study_swarm_robotics",
        ]
        
        if active_screen in case_study_targets:
            # Inject detailed case study knowledge for perfect answers
            case_study_context = build_publication_context_prompt(active_screen)
            project_context = build_project_context_prompt(active_screen)
            section_context = build_section_context_prompt(active_screen)
            
            if case_study_context:
                context_injection_parts.append(case_study_context)
            if project_context:
                context_injection_parts.append(project_context)
            if section_context:
                context_injection_parts.append(section_context)

        # 4. Navigation hint when section or navigation intent is detected
        if "screen navigation to" in user_exp.lower() or "navigate" in user_intent.lower() or "referencing portfolio section" in user_intent.lower():
            context_injection_parts.append("[If showing this section, call navigate_portfolio(target)]")

        # 5. Mode guidance
        context_injection_parts.append(f"[Response mode: {mode} — {'under 20 words' if mode == 'short' else '45-70 words, key result first'}]")

        full_context = "\n".join(context_injection_parts)

        # Strip prior ephemeral turn context injections so system prompts don't accumulate in chat history
        turn_ctx.items = [
            item for item in turn_ctx.items
            if not (
                hasattr(item, "role") and item.role == "system" and
                hasattr(item, "text_content") and "[Current Screen:" in (item.text_content or "")
            )
        ]
        turn_ctx.add_message(role="system", content=full_context)

        self._clean_extra(turn_ctx)

        # Keep context bounded (4 items = 2 exchanges) to eliminate token rate limit issues
        if len(turn_ctx.items) > 4:
            turn_ctx.truncate(max_items=4)

        elapsed_ms = (time.time() - start_time) * 1000.0

        log_turn(
            session_id=self.userdata.session_id,
            role="user",
            content=user_text,
            latency_ms=elapsed_ms,
            route=result.get("route", "direct"),
            target_screen=self.userdata.active_screen,
            active_agent=self.agent_name,
        )

    def update_conversation_summary(self, user_query: str, assistant_response: str) -> None:
        """Update rolling conversation summary for context coherence."""
        self.last_assistant_response = assistant_response
        exchange = f"Q: {user_query[:80]}... A: {assistant_response[:80]}..."
        if self.conversation_summary:
            self.conversation_summary = f"{self.conversation_summary} | {exchange}"
        else:
            self.conversation_summary = exchange
        if len(self.conversation_summary) > 300:
            self.conversation_summary = self.conversation_summary[-300:]

    async def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        """Polite interrupt handling for prolonged visitor turns."""
        print(f"--> [{self.agent_name} Watchdog] Turn exceeded: words={ev.accumulated_word_count}")
        if hasattr(self, "session") and self.session:
            await self.session.say(
                "Pardon the interruption — which specific area should we focus on?",
                allow_interruptions=True,
            )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        """Provides zero-latency speech cues during tool executions and captures response for context."""
        called_tools: list[llm.FunctionToolCall] = []
        has_text = False
        response_text = ""

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, llm.ChatChunk) and chunk.delta:
                if chunk.delta.content:
                    has_text = True
                    response_text += chunk.delta.content
                if chunk.delta.tool_calls:
                    called_tools.extend(chunk.delta.tool_calls)
            yield chunk

        # Update conversation summary with this exchange
        if response_text.strip() and self.last_user_query:
            self.update_conversation_summary(self.last_user_query, response_text.strip())

            # LOG: Agent response
            print(f"\n{'='*60}")
            print(f"🤖 AGENT ({self.agent_name}): {response_text.strip()}")
            print(f"{'='*60}\n")

        tool_names = [tool.name for tool in called_tools]
        if not has_text:
            if any("navigate" in t for t in tool_names):
                yield "Navigating your screen now. "
                yield FlushSentinel()
            elif any("research" in t for t in tool_names):
                yield "Retrieving verified research data now. "
                yield FlushSentinel()
            elif any("booking" in t or "schedule" in t for t in tool_names):
                yield "Opening the meeting scheduler now. "
                yield FlushSentinel()

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        """Sanitizes caption stream before sending over LiveKit data channel."""
        async for chunk in text:
            cleaned = (
                chunk.replace("**", "")
                .replace("###", "")
                .replace("##", "")
                .replace("`", "")
            )
            yield cleaned
