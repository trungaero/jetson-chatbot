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
from agent import ChatAgent
from tts import TextToSpeech
from audio_player import AudioPlayer
from reminder_scheduler import ReminderScheduler, reminder_queue


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

    # ── Initialize components ────────────────────────────────────────────────
    print("[1/4] Initializing Speech-to-Text...")
    stt = SpeechToText()
    print()

    print("[2/4] Initializing ReAct Agent...")
    agent = ChatAgent()
    print()

    print("[2b]  Starting reminder scheduler...")
    scheduler = ReminderScheduler()
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

    print("─" * 60)
    print("Ready! Press [ENTER] to start recording, [ENTER] again to stop.")
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

    # ── Main conversation loop ───────────────────────────────────────────────
    turn = 0
    while True:
        try:
            # ── Check for fired reminders before waiting for user input ──────
            try:
                trigger = reminder_queue.get_nowait()
                print(f"\n   🔔 Processing reminder...")
                tts_text, _ = agent.chat(trigger)
                if tts_text:
                    print("   ⏳ Synthesizing reminder speech...")
                    wav_path = tts.synthesize(tts_text)
                    if wav_path:
                        player.play(wav_path)
                        tts.cleanup_file(wav_path)
                continue
            except queue.Empty:
                pass  # No reminder due, proceed to normal recording

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

            # Step 3: Run ReAct agent (may invoke tools)
            print("   ⏳ Agent thinking...")
            tts_text, full_text = agent.chat(user_text)
            if not tts_text:
                continue

            # Step 4: Synthesize speech (reasoning already stripped)
            print("   ⏳ Synthesizing speech...")
            wav_path = tts.synthesize(tts_text)

            # Step 5: Play response audio
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
