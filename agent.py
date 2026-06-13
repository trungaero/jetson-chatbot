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

import requests as _http

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

_VISION_SYSTEM_PROMPT = (
    "You are the vision module of a voice assistant running on a Jetson device. "
    "Describe what you see in the image in 2-4 natural, conversational sentences. "
    "Focus on the most relevant objects, people, or scene. "
    "Do not use markdown, bullet points, or lists."
)

_VISION_MAX_TOKENS = 300


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from the model's response."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _describe_image_with_local_llm(base64_jpeg: str, user_question: str = "") -> str:
    """
    Send a base64 JPEG to the local llama-server (gemma4-e2b) for visual
    description using the OpenAI multimodal image_url format.

    llama-server accepts images as:
        { "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,<data>" } }

    Args:
        base64_jpeg: Base64-encoded JPEG string (no data-URI prefix).
        user_question: Optional context from the user's original query.

    Returns:
        A plain-text description of the image, or an error message.
    """
    data_uri = f"data:image/jpeg;base64,{base64_jpeg}"

    prompt_text = (
        f"The user asked: \"{user_question}\"\nDescribe what you see in this image."
        if user_question
        else "Describe what you see in this image."
    )

    payload = {
        "model": config.LLAMA_MODEL,
        "max_tokens": _VISION_MAX_TOKENS,
        "temperature": 0.5,
        "stream": False,
        "messages": [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
    }

    try:
        resp = _http.post(
            f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text or "I could not generate a description."
    except _http.exceptions.Timeout:
        return "Vision request timed out. Please try again."
    except _http.exceptions.ConnectionError:
        return (
            f"Could not reach the llama-server at {config.LLAMA_SERVER_URL}. "
            "Make sure it is running."
        )
    except _http.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        return f"llama-server vision error ({e}): {body}"
    except (KeyError, IndexError) as e:
        return f"Unexpected response format from llama-server: {e}"
    except Exception as e:
        return f"Unexpected error during vision analysis: {e}"


# ── Chat Agent ────────────────────────────────────────────────────────────────

class ChatAgent:
    """LangGraph ReAct agent wrapping llama.cpp server, with camera vision support."""

    def __init__(self):
        print(f"Connecting to LLM at {config.LLAMA_SERVER_URL} (model: {config.LLAMA_MODEL})...")

        self.llm = ChatOpenAI(
            base_url=f"{config.LLAMA_SERVER_URL}/v1",
            api_key="not-needed",           # llama.cpp doesn't require auth
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
            checkpointer=self.checkpointer,
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
        self.history = self.history[-config.MAX_CONVERSATION_HISTORY:]

        # Run the agent
        print("   ⚙ Agent running...")
        try:
            result = self._agent.invoke(
                {"messages": self.history},
                config={"configurable": {"thread_id": self._thread_id}},
            )
        except Exception as e:
            err = f"Agent error: {e}"
            print(f"   ✗ {err}")
            return "I encountered an error. Please try again.", err

        # ── Vision post-processing ────────────────────────────────────────────
        # If the agent called look_around_with_camera, intercept the tool
        # result, send the image to the local llama-server vision endpoint,
        # replace the raw base64 payload with a human-readable description,
        # then re-invoke the agent so it can produce a proper spoken reply.
        messages_out = result.get("messages", [])
        vision_description = self._extract_vision_description(messages_out, user_text)

        if vision_description:
            # Inject the description as a new human message and re-run
            inject = HumanMessage(
                content=(
                    f"[Camera vision result]: {vision_description}\n\n"
                    f"Now answer the user's original question: \"{user_text}\""
                )
            )
            try:
                result = self._agent.invoke(
                    {"messages": self.history[:-1] + [inject]},
                    config={"configurable": {"thread_id": self._thread_id + "-vision"}},
                )
                messages_out = result.get("messages", [])
            except Exception as e:
                print(f"   ✗ Vision re-invoke error: {e}")

        # ── Extract final response ────────────────────────────────────────────
        ai_messages = [
            m for m in messages_out
            if isinstance(m, AIMessage) and m.content
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
            print(f"   💭 Reasoning: {thinking[:200]}{'...' if len(thinking) > 200 else ''}")

        print(f"   🤖 Response: \"{tts_text}\"")

        # Persist AI response in history for future turns
        self.history.append(AIMessage(content=full_text))

        return tts_text, full_text

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_vision_description(
        self, messages: list[BaseMessage], user_text: str
    ) -> str | None:
        """
        Scan agent messages for a look_around_with_camera tool result that
        contains a base64_jpeg payload. If found, call the local llama-server
        vision endpoint and return a plain-text description. Returns None otherwise.
        """
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue

            content = msg.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    continue

            if not isinstance(content, dict):
                continue

            err = content.get("error")
            if err:
                print(f"   ✗ Camera tool error: {err}")
                return f"The camera tool reported an error: {err}"

            b64 = content.get("base64_jpeg")
            if b64:
                print("   🔍 Sending captured image to local llama-server for vision analysis...")
                description = _describe_image_with_local_llm(b64, user_text)
                print(f"   👁  Vision description: \"{description[:120]}{'...' if len(description) > 120 else ''}\"")
                return description

        return None

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
            {"role": "user",   "content": f"Remind the user about: {task}"},
        ]
        try:
            response = self._llm.invoke(messages)
            text = strip_reasoning(response.content.strip())
            print(f"   🔔 Reminder announcement: \"{text}\"")
            return text
        except Exception as e:
            print(f"   ✗ ReminderAgent error: {e}")
            return f"Reminder: {task}"
