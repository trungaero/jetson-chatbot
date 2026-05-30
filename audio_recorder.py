"""
Audio Recorder module — Push-to-Talk recording via PyAudio.
Press Enter to start recording, press Enter again to stop.
Works on headless Linux (no X server required).
"""

import io
import sys
import wave
import threading
import numpy as np
import pyaudio

import config


class AudioRecorder:
    """Records audio with Enter-key start/stop (works without X11)."""

    def __init__(self):
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.channels = config.AUDIO_CHANNELS
        self.chunk_size = config.AUDIO_CHUNK_SIZE
        self.format = pyaudio.paInt16

        self._audio = pyaudio.PyAudio()
        self._device_index = self._find_device()
        self._stream = None
        self._frames: list[bytes] = []
        self._recording = False
        self._stop_event = threading.Event()

    def _find_device(self) -> int | None:
        """Find the input device index matching config, or use default."""
        if config.AUDIO_DEVICE_INDEX is not None:
            return config.AUDIO_DEVICE_INDEX

        # Auto-detect Jabra or specified card
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and "jabra" in info["name"].lower():
                print(f"   Found input device: [{i}] {info['name']}")
                return i

        # Fallback to default
        return None

    def _wait_for_enter(self):
        """Wait for Enter key press in a background thread."""
        sys.stdin.readline()
        self._stop_event.set()

    def record(self) -> np.ndarray:
        """
        Block until recording cycle completes.
        Press Enter to start, press Enter again to stop.
        Returns recorded audio as a numpy float32 array normalized to [-1, 1].
        """
        self._frames = []
        self._stop_event.clear()
        self._recording = False

        print("\n🎤 Press [ENTER] to start recording...")
        sys.stdin.readline()  # Wait for first Enter

        # Start recording
        self._recording = True
        print("   ● Recording... Press [ENTER] to stop.")

        # Start a thread waiting for the stop signal
        stop_thread = threading.Thread(target=self._wait_for_enter, daemon=True)
        stop_thread.start()

        # Open audio stream and record
        stream_kwargs = dict(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        if self._device_index is not None:
            stream_kwargs["input_device_index"] = self._device_index

        self._stream = self._audio.open(**stream_kwargs)

        while not self._stop_event.is_set():
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                self._frames.append(data)
            except OSError:
                break

        self._recording = False
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None

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
