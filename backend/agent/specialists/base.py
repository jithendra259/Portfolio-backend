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

from agent.graph import route_portfolio_query
from agent.supabase_logger import log_turn
from .userdata import PortfolioUserData


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
        Lifecycle hook invoked when this agent becomes active.
        Preserves relevant conversation context while keeping it focused.
        """
        print(f"--> [Agent Lifecycle] Entered '{self.agent_name}' specialist.")

        # 1. Scoped Need-to-Know Handoff: Receive specific task + conversation summary
        if self.userdata.pending_handoff and self.userdata.pending_handoff.target == self.agent_name:
            handoff = self.userdata.pending_handoff
            self.userdata.pending_handoff = None

            new_ctx = self.chat_ctx.copy()
            new_ctx.items.clear()
            
            # Build context with conversation summary + current task
            context_parts = [
                f"[Task Directive: You are the {self.agent_name.capitalize()} Specialist. "
                f"Address the visitor's specific query: '{handoff.reason}'. "
                f"Active screen: {handoff.active_screen}.]",
            ]
            
            if self.conversation_summary:
                context_parts.append(f"[Conversation so far: {self.conversation_summary}]")
            
            if handoff.last_user_query:
                context_parts.append(f"Visitor just asked: '{handoff.last_user_query}'")

            new_ctx.add_message(role="system", content="\n".join(context_parts))

            if handoff.last_user_query:
                new_ctx.add_message(role="user", content=handoff.last_user_query)
            self.update_chat_ctx(new_ctx)
        else:
            # 2. Continuity: Carry forward relevant context from previous agent
            prev_agent = self.userdata.prev_agent
            if prev_agent and hasattr(prev_agent, "chat_ctx") and prev_agent.chat_ctx:
                try:
                    # Keep last 6 items (3 exchanges) for continuity
                    copied_ctx = prev_agent.chat_ctx.copy(
                        exclude_handoff=True,
                        exclude_config_update=True,
                        exclude_instructions=True,
                    ).truncate(max_items=6)
                    new_ctx = self.chat_ctx.copy()
                    for item in copied_ctx.items:
                        if item not in new_ctx.items:
                            new_ctx.items.append(item)
                    self.update_chat_ctx(new_ctx)
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
        Injects screen context, conversation history, and routing grounding.
        """
        if not new_message.text_content or not new_message.text_content.strip():
            raise StopResponse()

        user_text = new_message.text_content.strip()
        self.last_user_query = user_text
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

        # Build comprehensive context injection
        context_injection_parts = []
        
        # 1. Screen awareness (always relevant)
        screen_context = self.get_formatted_page_context()
        context_injection_parts.append(f"[Current Screen: {screen_context}]")

        # 2. Conversation summary for coherence
        if self.conversation_summary:
            context_injection_parts.append(f"[Previous conversation: {self.conversation_summary}]")

        # 3. Last assistant response to avoid repetition
        if self.last_assistant_response:
            context_injection_parts.append(f"[You previously said: {self.last_assistant_response[:150]}]")

        # 4. Routing grounding (specific facts for this query)
        if grounding:
            context_injection_parts.append(grounding)

        # 5. Mode guidance
        context_injection_parts.append(f"[Response mode: {mode} — {'under 20 words' if mode == 'short' else '45-70 words, key result first'}]")

        full_context = "\n".join(context_injection_parts)
        turn_ctx.add_message(role="system", content=full_context)

        self._clean_extra(turn_ctx)

        # Keep more context for conversation continuity (8 items = 4 exchanges)
        if len(turn_ctx.items) > 8:
            turn_ctx.truncate(max_items=8)

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
        # Simple rolling summary - in production could use LLM to summarize
        exchange = f"Q: {user_query[:80]}... A: {assistant_response[:80]}..."
        if self.conversation_summary:
            self.conversation_summary = f"{self.conversation_summary} | {exchange}"
        else:
            self.conversation_summary = exchange
        # Keep summary bounded
        if len(self.conversation_summary) > 500:
            self.conversation_summary = self.conversation_summary[-500:]

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
