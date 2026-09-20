"""
Booking Specialist Agent.
Facilitates recruiter scheduling, technical interview booking, and collaboration intake.
Persists verified leads directly into Supabase via non-blocking background workers.
"""

import json
from typing import Annotated, Callable, Optional
from livekit import rtc
from livekit.agents import llm

from .base import PortfolioBaseAgent
from .userdata import PortfolioUserData
from prompts.response_policy import COMMON_RESPONSE_POLICY
from agent.supabase_logger import log_booking_lead

BOOKING_INSTRUCTIONS = f"""You are Kandula Jithendra Subramanyam's Recruiter Relations & Booking Specialist.
Your mission is to make scheduling an interview, coffee chat, or research discussion completely effortless.

### OBJECTIVES:
1. Warmly confirm the visitor's intent to connect with Jithendra.
2. Ensure the visitor's screen is viewing the booking interface (`navigate_portfolio(target='book_appointment')`).
3. Collect the essential details in a natural conversational flow:
   - Full Name
   - Email address
   - Topic of discussion (e.g., Senior AI / Quant Researcher role, project inquiry, speaking invitation)
   - Preferred date or time
4. Once provided, call `confirm_booking` to log the appointment into the database.
5. If the visitor wants to review more portfolio work first, call `transfer_to_greeter`.
\n{COMMON_RESPONSE_POLICY}
"""


class BookingSpecialist(PortfolioBaseAgent):
    """Specialist agent managing recruiter bookings and lead capture into Supabase."""

    def __init__(
        self,
        tools: list[llm.Tool | llm.Toolset],
        userdata: PortfolioUserData,
        get_room: Callable[[], Optional[rtc.Room]],
    ) -> None:
        @llm.function_tool(
            description=(
                "Confirm and save a recruiter meeting, job interview, or collaboration inquiry. "
                "Persists the lead into Supabase and updates the frontend appointment status."
            )
        )
        async def confirm_booking(
            name: Annotated[str, "Visitor's full name."],
            email: Annotated[str, "Visitor's email address."],
            topic: Annotated[str, "Meeting topic or job role opportunity."],
            preferred_date: Annotated[str, "Preferred date for the meeting."] = "Flexible",
            preferred_time: Annotated[str, "Preferred time slot."] = "Flexible",
            notes: Annotated[str, "Any special notes or job requirements."] = "",
        ) -> str:
            """Records recruiter appointment into Supabase and broadcasts confirmation."""
            userdata.set_visitor(name=name, email=email)
            userdata.booking_details = {
                "name": name,
                "email": email,
                "topic": topic,
                "date": preferred_date,
                "time": preferred_time,
                "notes": notes,
            }

            # Asynchronous non-blocking write to Supabase
            log_booking_lead(
                visitor_name=name,
                email=email,
                topic=topic,
                preferred_date=preferred_date,
                preferred_time=preferred_time,
                notes=notes,
                session_id=userdata.session_id,
            )

            # Broadcast event to frontend over data channel
            room = get_room()
            if room and room.local_participant:
                try:
                    payload = json.dumps({
                        "type": "booking_confirmed",
                        "booking": userdata.booking_details,
                    })
                    await room.local_participant.publish_data(payload.encode("utf-8"), topic="assistant_action")
                except Exception as e:
                    print(f"--> [Booking Broadcast Warning] {e}")

            return (
                f"Booking confirmed for {name} ({email}) regarding '{topic}' on {preferred_date} at {preferred_time}. "
                "The lead has been recorded, and Jithendra will follow up promptly."
            )

        all_tools = list(tools) + [confirm_booking]

        super().__init__(
            agent_name="booking",
            instructions=BOOKING_INSTRUCTIONS,
            tools=all_tools,
            userdata=userdata,
            get_room=get_room,
        )

    async def on_enter(self) -> None:
        """Navigates to booking page and offers help upon handoff."""
        await super().on_enter()
        try:
            # Auto-navigate screen to booking page
            room = self._get_room()
            if room and room.local_participant:
                payload = json.dumps({"type": "navigate", "target": "book_appointment"})
                await room.local_participant.publish_data(payload.encode("utf-8"), topic="navigation")

            if hasattr(self, "session") and self.session:
                name_prefix = f" {self.userdata.visitor_name}" if self.userdata.visitor_name else ""
                self.session.say(
                    f"I'm Jithendra's Booking Specialist. I can lock in a meeting or interview slot for you{name_prefix}. What date, time, and topic work best?",
                    allow_interruptions=True,
                )
        except Exception as e:
            print(f"--> [Booking Entry Warning] {e}")
