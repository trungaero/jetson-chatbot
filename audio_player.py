"""
Audio Player module — Plays WAV files through the system speaker.
Uses PyAudio for cross-platform playback.
"""

import os
import wave
import subprocess
import pyaudio


class AudioPlayer:
    """Plays WAV audio files."""

    def __init__(self, use_aplay: bool = True):
        """
        Args:
            use_aplay: If True, use aplay (Linux/ALSA) for playback.
                       If False, use PyAudio for playback.
        """
        self.use_aplay = use_aplay
        if not use_aplay:
            self._audio = pyaudio.PyAudio()
        else:
            self._audio = None

    def play(self, wav_path: str):
        """
        Play a WAV file.

        Args:
            wav_path: Path to the WAV file to play.
        """
        if not wav_path or not os.path.isfile(wav_path):
            print("   ✗ No audio file to play.")
            return

        if self.use_aplay:
            self._play_aplay(wav_path)
        else:
            self._play_pyaudio(wav_path)

    def _play_aplay(self, wav_path: str):
        """Play using aplay (ALSA, Linux) on configured device."""
        import config
        try:
            subprocess.run(
                ["aplay", "-D", config.AUDIO_ALSA_DEVICE, "-q", wav_path],
                check=True,
                timeout=60,
            )
        except FileNotFoundError:
            print("   ✗ 'aplay' not found. Install alsa-utils or set use_aplay=False.")
            # Fallback to PyAudio
            self._play_pyaudio(wav_path)
        except subprocess.TimeoutExpired:
            print("   ✗ Audio playback timed out.")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ aplay error: {e}")

    def _play_pyaudio(self, wav_path: str):
        """Play using PyAudio."""
        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        try:
            wf = wave.open(wav_path, "rb")
        except wave.Error as e:
            print(f"   ✗ Cannot open WAV file: {e}")
            return

        stream = self._audio.open(
            format=self._audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        chunk_size = 1024
        data = wf.readframes(chunk_size)
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)

        stream.stop_stream()
        stream.close()
        wf.close()

    def play_multiple(self, wav_paths: list[str]):
        """Play multiple WAV files sequentially (for sentence-by-sentence playback)."""
        for path in wav_paths:
            self.play(path)

    def cleanup(self):
        """Release PyAudio resources."""
        if self._audio:
            self._audio.terminate()
            self._audio = None
