"""
Audio Recorder module — Push-to-Talk recording via PyAudio.
Hold the configured key to record, release to stop.
"""

import io
import wave
import threading
import numpy as np
import pyaudio
from pynput import keyboard

import config


class AudioRecorder:
    """Records audio while the push-to-talk key is held down."""

    def __init__(self):
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.channels = config.AUDIO_CHANNELS
        self.chunk_size = config.AUDIO_CHUNK_SIZE
        self.format = pyaudio.paInt16
        self.ptt_key = config.PTT_KEY

        self._audio = pyaudio.PyAudio()
        self._stream = None
        self._frames: list[bytes] = []
        self._recording = False
        self._done_event = threading.Event()

    def _get_ptt_key(self):
        """Map config key string to pynput Key object."""
        key_map = {
            "space": keyboard.Key.space,
            "ctrl": keyboard.Key.ctrl_l,
            "shift": keyboard.Key.shift,
            "alt": keyboard.Key.alt_l,
        }
        return key_map.get(self.ptt_key, keyboard.Key.space)

    def record(self) -> np.ndarray:
        """
        Block until push-to-talk cycle completes.
        Returns recorded audio as a numpy float32 array normalized to [-1, 1].
        """
        self._frames = []
        self._done_event.clear()
        target_key = self._get_ptt_key()

        print(f"\n🎤 Hold [{self.ptt_key.upper()}] to speak...")

        def on_press(key):
            if key == target_key and not self._recording:
                self._recording = True
                print("   ● Recording... (release to stop)")

        def on_release(key):
            if key == target_key and self._recording:
                self._recording = False
                self._done_event.set()
                return False  # Stop listener

        # Start keyboard listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # Wait for key press to start recording
        while not self._recording and not self._done_event.is_set():
            self._done_event.wait(timeout=0.05)

        # Open audio stream and record
        if self._recording:
            self._stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            while self._recording:
                try:
                    data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    self._frames.append(data)
                except OSError:
                    break

            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        listener.join(timeout=1.0)

        if not self._frames:
            return np.array([], dtype=np.float32)

        # Convert raw bytes to numpy float32 array
        audio_data = np.frombuffer(b"".join(self._frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        duration = len(audio_float) / self.sample_rate
        print(f"   ✓ Recorded {duration:.1f}s of audio")

        return audio_float

    def record_to_wav_bytes(self) -> bytes:
        """
        Record audio and return as WAV file bytes (in-memory).
        Useful for passing directly to whisper.
        """
        audio_float = self.record()
        if len(audio_float) == 0:
            return b""

        # Convert back to int16 for WAV
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Write to in-memory WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(config.AUDIO_FORMAT_WIDTH)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    def cleanup(self):
        """Release PyAudio resources."""
        if self._stream:
            self._stream.close()
        self._audio.terminate()
