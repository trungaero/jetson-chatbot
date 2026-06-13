"""
Agent module — LangGraph ReAct agent using Gemma 4 via llama.cpp server.

Architecture:
  - ChatOpenAI client → llama.cpp OpenAI-compatible API
  - LangGraph create_react_agent for tool-calling loop
  - Conversation history managed manually
  - Reasoning blocks (<think>...</think>) stripped before TTS
  - Vision: when look_around_with_camera tool fires, the captured image is
    sent directly to the local llama-server (gemma4-e2b supports vision via
    the OpenAI multimodal image_url format). No external API key required.
"""

import re
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import config
from tools import get_tools

# ── Constants ─────────────────────────────────────────────────────────────────

REMINDER_SYSTEM_PROMPT = (
    "You are a reminder assistant. Your only job is to notify the user that "
    "a previously scheduled reminder has arrived. "
    "Speak in a natural, friendly, direct tone — one or two sentences maximum. "
    "Do not ask follow-up questions. Do not use markdown, bullet points, or lists. "
    "Example: 'Hey, just a heads-up — it's time to take your medication.'"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from the model's response."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Chat Agent ────────────────────────────────────────────────────────────────


class ChatAgent:
    """LangGraph ReAct agent wrapping llama.cpp server, with camera vision support."""

    def __init__(self):
        print(
            f"Connecting to LLM at {config.LLAMA_SERVER_URL} (model: {config.LLAMA_MODEL})..."
        )

        self.llm = ChatOpenAI(
            base_url=f"{config.LLAMA_SERVER_URL}/v1",
            api_key="not-needed",  # llama.cpp doesn't require auth
            model=config.LLAMA_MODEL,
            max_tokens=config.LLAMA_MAX_TOKENS,
            temperature=config.LLAMA_TEMPERATURE,
        )

        self.tools = get_tools()
        self.checkpointer = InMemorySaver()
        self.history: list[BaseMessage] = []
        self._thread_id = "main"

        tool_names = [t.name for t in self.tools]
        print(f"✓ Agent ready. Tools: {tool_names}")

        # Build agent — system prompt injected via prompt parameter
        self._agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=config.LLAMA_SYSTEM_PROMPT,
            # checkpointer=self.checkpointer,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_text: str) -> tuple[str, str]:
        """
        Send user input through the ReAct agent and return the response.

        When the agent invokes look_around_with_camera, the returned image is
        sent directly to the local llama-server vision endpoint for description.
        That description is injected back so the agent can craft a natural reply.

        Returns:
            (tts_text, full_text) where:
              - tts_text: reasoning stripped, safe for Piper TTS
              - full_text: complete raw response for logging/display
        """
        if not user_text.strip():
            return "", ""

        # Append new message and keep only the last 10
        self.history.append(HumanMessage(content=user_text))
        self.history = self.history[-config.MAX_CONVERSATION_HISTORY :]

        # Run the agent
        print("   ⚙ Agent running...")
        try:
            result = self._agent.invoke(
                {"messages": self.history},
                # config={"configurable": {"thread_id": self._thread_id}},
            )
        except Exception as e:
            err = f"Agent error: {e}"
            print(f"   ✗ {err}")
            return "I encountered an error. Please try again.", err
        # ── Extract final response ────────────────────────────────────────────
        messages_out = result.get("messages", [])

        ai_messages = [
            m for m in messages_out if isinstance(m, AIMessage) and m.content
        ]
        full_text = ai_messages[-1].content.strip() if ai_messages else ""

        if not full_text:
            return "I didn't get a response. Please try again.", ""

        # Strip reasoning for TTS
        tts_text = strip_reasoning(full_text)

        # Log reasoning separately if present
        thinking_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            print(
                f"   💭 Reasoning: {thinking[:200]}{'...' if len(thinking) > 200 else ''}"
            )

        print(f'   🤖 Response: "{tts_text}"')

        # Persist AI response in history for future turns
        self.history.append(AIMessage(content=full_text))

        return tts_text, full_text

    def reset_history(self):
        """Clear conversation history."""
        self.history = []
        print("   🔄 Conversation history cleared.")


# ── Reminder Agent ────────────────────────────────────────────────────────────


class ReminderAgent:
    """
    Lightweight agent with no tools, dedicated to announcing reminders.
    No conversation history is maintained.
    """

    def __init__(self):
        print("Initializing ReminderAgent...")
        self._llm = ChatOpenAI(
            base_url=f"{config.LLAMA_SERVER_URL}/v1",
            api_key="not-needed",
            model=config.LLAMA_MODEL,
            max_tokens=80,
            temperature=0.5,
        )
        print("✓ ReminderAgent ready.")

    def announce(self, task: str) -> str:
        """Generate a spoken reminder announcement for the given task."""
        messages = [
            {"role": "system", "content": REMINDER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Remind the user about: {task}"},
        ]
        try:
            response = self._llm.invoke(messages)
            text = strip_reasoning(response.content.strip())
            print(f'   🔔 Reminder announcement: "{text}"')
            return text
        except Exception as e:
            print(f"   ✗ ReminderAgent error: {e}")
            return f"Reminder: {task}"
