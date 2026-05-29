"""
Speech-to-Text module using faster-whisper.
GPU-accelerated transcription on Jetson Orin Nano.
"""

import numpy as np
from faster_whisper import WhisperModel

import config


class SpeechToText:
    """Wraps faster-whisper for real-time transcription."""

    def __init__(self):
        print(f"Loading Whisper model '{config.WHISPER_MODEL_SIZE}' "
              f"on {config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE_TYPE})...")
        
        self.model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        print("✓ Whisper model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio numpy array to text.

        Args:
            audio: Float32 numpy array, mono, 16kHz sample rate.

        Returns:
            Transcribed text string.
        """
        if len(audio) == 0:
            return ""

        segments, info = self.model.transcribe(
            audio,
            beam_size=config.WHISPER_BEAM_SIZE,
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,          # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts).strip()

        if full_text:
            print(f"   📝 Transcribed: \"{full_text}\"")
            print(f"      (language: {info.language}, probability: {info.language_probability:.2f})")

        return full_text

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            Transcribed text string.
        """
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=config.WHISPER_BEAM_SIZE,
            language=config.WHISPER_LANGUAGE,
            vad_filter=True,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()
