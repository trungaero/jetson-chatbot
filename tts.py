"""
Text-to-Speech module using Piper TTS.
Converts text to speech via the Piper binary (subprocess).
"""

import os
import subprocess
import tempfile
import time

import config


class TextToSpeech:
    """Wraps Piper TTS for local text-to-speech synthesis."""

    def __init__(self):
        self.piper_exe = config.PIPER_EXECUTABLE
        self.model_path = config.PIPER_MODEL
        self.output_dir = config.PIPER_OUTPUT_DIR

        # Verify Piper binary exists
        if not os.path.isfile(self.piper_exe):
            print(f"⚠ Piper executable not found at: {self.piper_exe}")
            print("  Download from: https://github.com/rhasspy/piper/releases")
        else:
            print("✓ Piper TTS ready.")

        # Verify model exists
        if not os.path.isfile(self.model_path):
            print(f"⚠ Piper voice model not found at: {self.model_path}")
            print("  Download from: https://huggingface.co/rhasspy/piper-voices")

    def synthesize(self, text: str) -> str:
        """
        Convert text to speech and save as WAV file.

        Args:
            text: Text to synthesize.

        Returns:
            Path to the generated WAV file, or empty string on failure.
        """
        if not text.strip():
            return ""

        # Generate unique output filename
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(self.output_dir, f"response_{timestamp}.wav")

        try:
            # Pipe text into Piper via stdin
            process = subprocess.run(
                [
                    self.piper_exe,
                    "--model",
                    self.model_path,
                    "--output_file",
                    output_path,
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if process.returncode != 0:
                print(f"   ✗ Piper TTS error: {process.stderr.strip()}")
                return ""

            if os.path.isfile(output_path):
                file_size = os.path.getsize(output_path)
                print(f"   🔊 Synthesized audio ({file_size // 1024}KB): {output_path}")
                return output_path
            else:
                print("   ✗ Piper produced no output file.")
                return ""

        except subprocess.TimeoutExpired:
            print("   ✗ Piper TTS timed out.")
            return ""
        except FileNotFoundError:
            print(f"   ✗ Piper binary not found: {self.piper_exe}")
            return ""

    def synthesize_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences and synthesize each separately.
        Useful for streaming-style playback of long responses.

        Args:
            text: Full response text.

        Returns:
            List of paths to WAV files (one per sentence).
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s.strip()]

        wav_files = []
        for sentence in sentences:
            wav_path = self.synthesize(sentence)
            if wav_path:
                wav_files.append(wav_path)

        return wav_files

    def cleanup_file(self, wav_path: str):
        """Remove a generated WAV file after playback."""
        try:
            if os.path.isfile(wav_path):
                os.remove(wav_path)
        except OSError:
            pass
