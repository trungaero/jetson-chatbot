"""
VAD Recorder — automatically detects speech and records until silence.
Uses Silero VAD model (runs on CPU or CUDA).
"""

import io
import wave
import threading
import collections
import numpy as np
import pyaudio
import torch
import config


class VADRecorder:
    """Records audio automatically using Voice Activity Detection."""

    def __init__(self):
        self.sample_rate = 16000  # Silero VAD requires 16kHz
        self.channels = 1
        self.chunk_size = 512
        self.format = pyaudio.paInt16

        self._audio = pyaudio.PyAudio()
        self._device_index = self._find_device()

        print("🧠 Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model.eval()
        print("✓ VAD model loaded")

    def _find_device(self) -> int | None:
        if config.AUDIO_DEVICE_INDEX is not None:
            return config.AUDIO_DEVICE_INDEX
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and "jabra" in info["name"].lower():
                return i
        return None

    def _get_speech_prob(self, chunk_bytes: bytes) -> float:
        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        audio_float = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
        with torch.no_grad():
            return self._model(audio_float, self.sample_rate).item()

    def record(self) -> np.ndarray:
        """Blocking: listen, detect speech, record until silence, return audio."""
        threshold = config.VAD_THRESHOLD
        silence_limit = int(
            config.VAD_SILENCE_DURATION * self.sample_rate / self.chunk_size
        )
        min_speech_chunks = int(
            config.VAD_MIN_SPEECH_DURATION * self.sample_rate / self.chunk_size
        )
        pre_roll = collections.deque(maxlen=config.VAD_PRE_ROLL_CHUNKS)

        stream_kwargs = dict(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        if self._device_index is not None:
            stream_kwargs["input_device_index"] = self._device_index

        stream = self._audio.open(**stream_kwargs)
        print("\n👂 Listening... (speak to start)")

        frames, silence_chunks, speech_chunks, in_speech = [], 0, 0, False

        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                prob = self._get_speech_prob(data)

                if not in_speech:
                    pre_roll.append(data)
                    if prob >= threshold:
                        in_speech = True
                        speech_chunks = 1
                        silence_chunks = 0
                        frames = list(pre_roll)
                        print("● Recording...")
                else:
                    frames.append(data)
                    if prob >= threshold:
                        speech_chunks += 1
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            if speech_chunks >= min_speech_chunks:
                                break
                            else:
                                print("  (too short, ignoring)")
                                frames, speech_chunks, silence_chunks, in_speech = (
                                    [],
                                    0,
                                    0,
                                    False,
                                )
                                pre_roll.clear()
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return np.array([], dtype=np.float32)

        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        print(f"✓ Captured {len(audio_float)/self.sample_rate:.1f}s of speech")
        return audio_float

    def record_to_wav_bytes(self) -> bytes:
        audio_float = self.record()
        if len(audio_float) == 0:
            return b""
        audio_int16 = (audio_float * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    def start_loop(self, input_queue, stt):
        def _loop():
            while True:
                try:
                    audio_float = self.record()
                    if len(audio_float) == 0:
                        continue
                    text = stt.transcribe(audio_float)
                    if text and text.strip():
                        input_queue.put(("user", text.strip()))
                except Exception as e:
                    print(f"[VADRecorder] error: {e}")

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def cleanup(self):
        self._audio.terminate()
