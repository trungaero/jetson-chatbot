"""
Jetson Orin Nano Voice Chatbot — Main Entry Point

Pipeline:
  1. Push-to-talk audio recording
  2. Speech-to-text via faster-whisper
  3. LangGraph ReAct agent (Gemma 4 E2B) with tool calling
  4. Text-to-speech via Piper TTS  (reasoning stripped)
  5. Audio playback

Usage:
  python main.py
"""

import sys
import signal
import queue

from audio_recorder import AudioRecorder
from stt import SpeechToText
from agent import ChatAgent, ReminderAgent
from tts import TextToSpeech
from audio_player import AudioPlayer
from reminder_scheduler import ReminderScheduler


def print_banner():
    """Print startup banner."""
    print("=" * 60)
    print("  🤖 Jetson Orin Nano Voice Chatbot")
    print("  Model: Gemma 4 E2B via llama.cpp")
    print("  Agent: LangGraph ReAct + search + reminders")
    print("  STT:   faster-whisper")
    print("  TTS:   Piper")
    print("=" * 60)
    print()


def main():
    print_banner()
    # ── Shared input queue (recording thread + reminder scheduler → main loop) ───
    input_queue: queue.Queue = queue.Queue()
    # ── Initialize components ────────────────────────────────────────────────
    print("[1/4] Initializing Speech-to-Text...")
    stt = SpeechToText()
    print()

    print("[2/4] Initializing ReAct Agent...")
    agent = ChatAgent()
    print()

    reminder_agent = ReminderAgent()
    print()

    print("[2b]  Starting reminder scheduler...")
    scheduler = ReminderScheduler(input_queue)
    scheduler.start()
    print()

    print("[3/4] Initializing Text-to-Speech...")
    tts = TextToSpeech()
    print()

    print("[4/4] Initializing Audio I/O...")
    recorder = AudioRecorder()
    player = AudioPlayer(use_aplay=True)
    print("✓ Audio I/O ready.")
    print()

    # Start background recording loop — feeds ("user", text) into input_queue
    recorder.start_loop(input_queue, stt)

    print("─" * 60)
    print("Ready! Press [ENTER] to start recording, [ENTER] again to stop.")
    print("Reminders will interrupt automatically when due.")
    print("Press Ctrl+C to quit.")
    print("─" * 60)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    def shutdown(signum=None, frame=None):
        print("\n\nShutting down...")
        scheduler.stop()
        recorder.cleanup()
        player.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Main loop — drain input_queue (user speech + reminders) ─────────────
    while True:
        try:
            # Block until either a user utterance or a reminder arrives
            kind, text = input_queue.get()

            if kind == "reminder":
                # Extract task name from the trigger string
                task = text.replace("[REMINDER] Please remind the user about: ", "").strip()
                print(f"\n   🔔 Reminder triggered: {task}")
                tts_text = reminder_agent.announce(task)
                if tts_text:
                    print("   ⏳ Synthesizing reminder speech...")
                    wav_path = tts.synthesize(tts_text)
                    if wav_path:
                        player.play(wav_path)
                        tts.cleanup_file(wav_path)
                continue

            # Normal user input → full ReAct agent
            print(f"   🗣 User: \"{text}\"")
            print("   ⏳ Agent thinking...")
            tts_text, full_text = agent.chat(text)
            if not tts_text:
                continue

            print("   ⏳ Synthesizing speech...")
            wav_path = tts.synthesize(tts_text)
            if wav_path:
                player.play(wav_path)
                tts.cleanup_file(wav_path)
            else:
                print(f"   (TTS failed — response: {tts_text})")

        except KeyboardInterrupt:
            shutdown()
        except Exception as e:
            print(f"\n   ✗ Error: {e}")
            print("   Continuing...\n")
            continue


if __name__ == "__main__":
    main()
