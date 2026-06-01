"""
Configuration for the Jetson Orin Nano Voice Chatbot.
All tunable parameters are defined here.
"""

import os as _os

# ==============================================================================
# LLM Server (llama.cpp)
# ==============================================================================
LLAMA_SERVER_URL = "http://localhost:8080"
LLAMA_MODEL = "gemma-4-e2b"       # model name reported by llama.cpp server
LLAMA_MAX_TOKENS = 512
LLAMA_TEMPERATURE = 0.7

# Load system prompt from file; fall back to default if file is missing
_SYSTEM_PROMPT_FILE = _os.path.join(_os.path.dirname(__file__), "system_prompt.txt")
if _os.path.isfile(_SYSTEM_PROMPT_FILE):
    with open(_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as _f:
        LLAMA_SYSTEM_PROMPT = _f.read().strip()
else:
    LLAMA_SYSTEM_PROMPT = (
        "You are a helpful voice assistant running on a Jetson Orin Nano. "
        "Keep your responses concise, conversational, and under 3 sentences "
        "unless the user asks for more detail."
    )

# ==============================================================================
# Speech-to-Text (faster-whisper)
# ==============================================================================
WHISPER_MODEL_SIZE = "base"          # Options: tiny, base, small, medium
WHISPER_DEVICE = "cpu"              # "cuda" for GPU, "cpu" for CPU-only
WHISPER_COMPUTE_TYPE = "int8"     # float16 for GPU, int8 for CPU
WHISPER_LANGUAGE = "en"              # Set to None for auto-detect
WHISPER_BEAM_SIZE = 5

# ==============================================================================
# Text-to-Speech (Piper)
# ==============================================================================
PIPER_EXECUTABLE = "./piper/piper"
PIPER_MODEL = "./piper/models/en_US-lessac-medium.onnx"
PIPER_OUTPUT_DIR = "/tmp"

# ==============================================================================
# Audio Device (Jabra Link 380 — card 2, device 0)
# ==============================================================================
AUDIO_DEVICE_INDEX = None       # PyAudio device index (None = default/auto-detect)
AUDIO_CARD = 2                  # ALSA card number
AUDIO_DEVICE = 0                # ALSA device number
AUDIO_ALSA_DEVICE = "plughw:2,0"  # ALSA device string for aplay (plughw handles format conversion)

# ==============================================================================
# Audio Recording
# ==============================================================================
AUDIO_SAMPLE_RATE = 16000       # 16kHz required by Whisper
AUDIO_CHANNELS = 1              # Mono
AUDIO_FORMAT_WIDTH = 2          # 16-bit (2 bytes per sample)
AUDIO_CHUNK_SIZE = 1024         # Frames per buffer

# ==============================================================================
# Conversation
# ==============================================================================
MAX_CONVERSATION_HISTORY = 10   # Max number of message pairs to keep in context
