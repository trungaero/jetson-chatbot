"""
LLM Client module — Communicates with llama.cpp server via OpenAI-compatible API.
"""

import requests

import config


class LLMClient:
    """HTTP client for llama.cpp server running Qwen3:1.7B."""

    def __init__(self):
        self.server_url = config.LLAMA_SERVER_URL
        self.max_tokens = config.LLAMA_MAX_TOKENS
        self.temperature = config.LLAMA_TEMPERATURE
        self.system_prompt = config.LLAMA_SYSTEM_PROMPT
        self.max_history = config.MAX_CONVERSATION_HISTORY

        # Conversation history: list of {"role": ..., "content": ...}
        self.history: list[dict] = []

        # Verify server is reachable
        self._check_server()

    def _check_server(self):
        """Check if llama.cpp server is running."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"✓ LLM server connected at {self.server_url}")
            else:
                print(f"⚠ LLM server responded with status {resp.status_code}")
        except requests.ConnectionError:
            print(f"⚠ Cannot reach LLM server at {self.server_url}")
            print("  Make sure llama-server is running!")

    def _build_messages(self, user_text: str) -> list[dict]:
        """Build the full message array with system prompt and history."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _trim_history(self):
        """Keep conversation history within the configured limit."""
        # Each exchange is 2 messages (user + assistant)
        max_messages = self.max_history * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def chat(self, user_text: str) -> str:
        """
        Send user text to LLM and return the response.

        Args:
            user_text: The transcribed user speech.

        Returns:
            Assistant's response text.
        """
        if not user_text.strip():
            return ""

        messages = self._build_messages(user_text)

        payload = {
            "model": config.LLAMA_MODEL,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=60,  # LLM generation can take time on Jetson
            )
            resp.raise_for_status()
            data = resp.json()

            assistant_text = data["choices"][0]["message"]["content"].strip()

            # Update conversation history
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": assistant_text})
            self._trim_history()

            print(f'   🤖 Response: "{assistant_text}"')
            return assistant_text

        except requests.Timeout:
            print("   ✗ LLM request timed out.")
            return "Sorry, I took too long to respond. Please try again."
        except requests.ConnectionError:
            print("   ✗ Cannot connect to LLM server.")
            return "I'm having trouble connecting to my brain. Is the server running?"
        except (KeyError, IndexError) as e:
            print(f"   ✗ Unexpected LLM response format: {e}")
            return "I received an unexpected response. Please try again."
        except requests.HTTPError as e:
            print(f"   ✗ LLM server error: {e}")
            return "The language model encountered an error."

    def reset_history(self):
        """Clear conversation history for a fresh start."""
        self.history = []
        print("   🔄 Conversation history cleared.")
