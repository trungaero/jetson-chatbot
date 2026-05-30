"""
Jetson Orin Nano Voice Chatbot — Main Entry Point

Pipeline:
  1. Push-to-talk audio recording
  2. Speech-to-text via faster-whisper (GPU)
  3. LLM inference via llama.cpp server (Qwen3:1.7B)
  4. Text-to-speech via Piper TTS
  5. Audio playback

Usage:
  python main.py
"""

import sys
import signal

from audio_recorder import AudioRecorder
from stt import SpeechToText
from llm_client import LLMClient
from tts import TextToSpeech
from audio_player import AudioPlayer


def print_banner():
    """Print startup banner."""
    print("=" * 60)
    print("  🤖 Jetson Orin Nano Voice Chatbot")
    print("  Model: Qwen3:1.7B via llama.cpp")
    print("  STT:   faster-whisper (GPU)")
    print("  TTS:   Piper")
    print("=" * 60)
    print()


def main():
    print_banner()

    # ── Initialize components ────────────────────────────────────────────────
    print("[1/4] Initializing Speech-to-Text...")
    stt = SpeechToText()
    print()

    print("[2/4] Connecting to LLM server...")
    llm = LLMClient()
    print()

    print("[3/4] Initializing Text-to-Speech...")
    tts = TextToSpeech()
    print()

    print("[4/4] Initializing Audio I/O...")
    recorder = AudioRecorder()
    player = AudioPlayer(use_aplay=True)
    print("✓ Audio I/O ready.")
    print()

    print("─" * 60)
    print("Ready! Press [ENTER] to start recording, [ENTER] again to stop.")
    print("Press Ctrl+C to quit.")
    print("─" * 60)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    def shutdown(signum=None, frame=None):
        print("\n\nShutting down...")
        recorder.cleanup()
        player.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Main conversation loop ───────────────────────────────────────────────
    turn = 0
    while True:
        try:
            turn += 1

            # Step 1: Record audio (push-to-talk)
            audio_data = recorder.record()
            if len(audio_data) == 0:
                print("   (No audio captured, try again)")
                continue

            # Check for very short recordings (likely accidental)
            duration = len(audio_data) / 16000
            if duration < 0.3:
                print("   (Recording too short, try again)")
                continue

            # Step 2: Transcribe speech to text
            print("   ⏳ Transcribing...")
            user_text = stt.transcribe(audio_data)
            if not user_text:
                print("   (Could not understand speech, try again)")
                continue

            # Step 3: Get LLM response
            print("   ⏳ Thinking...")
            response_text = llm.chat(user_text)
            if not response_text:
                continue

            # Step 4: Synthesize speech
            print("   ⏳ Synthesizing speech...")
            wav_path = tts.synthesize(response_text)

            # Step 5: Play response audio
            if wav_path:
                player.play(wav_path)
                tts.cleanup_file(wav_path)
            else:
                print("   (TTS failed, text response shown above)")

        except KeyboardInterrupt:
            shutdown()
        except Exception as e:
            print(f"\n   ✗ Error: {e}")
            print("   Continuing...\n")
            continue


if __name__ == "__main__":
    main()
