# Jetson Orin Nano Voice Chatbot

A fully local, offline voice chatbot running on NVIDIA Jetson Orin Nano. Speak to it, and it responds with voice — no cloud services required.

## Architecture

```
[Microphone] → [PyAudio Recording] → [faster-whisper STT]
    → [llama.cpp / Qwen3:1.7B] → [Piper TTS] → [Speaker]
```

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Audio Capture | PyAudio + pynput | Push-to-talk recording |
| Speech-to-Text | faster-whisper (GPU) | Transcribe speech to text |
| LLM | llama.cpp server + Qwen3:1.7B | Generate conversational responses |
| Text-to-Speech | Piper TTS | Convert response text to audio |
| Audio Playback | aplay / PyAudio | Play synthesized speech |

## Prerequisites

- **Hardware:** NVIDIA Jetson Orin Nano (8GB)
- **OS:** JetPack 6.x (Ubuntu 22.04, L4T R36.x)
- **Peripherals:** USB microphone, speaker (USB or 3.5mm)
- **CUDA:** Verified with `nvcc --version`

## Setup Instructions

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev \
    ffmpeg cmake git alsa-utils wget
```

### 2. Verify Audio Hardware

```bash
# List recording devices
arecord -l

# List playback devices
aplay -l

# Test recording (5 seconds)
arecord -d 5 -f S16_LE -r 16000 -c 1 test.wav

# Test playback
aplay test.wav
```

### 3. Build llama.cpp with CUDA

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

### 4. Download Qwen3:1.7B Model

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-1.7B-GGUF \
    qwen3-1.7b-q4_k_m.gguf \
    --local-dir ~/llama.cpp/models
```

### 5. Start llama.cpp Server

```bash
cd ~/llama.cpp
./build/bin/llama-server \
    -m ./models/qwen3-1.7b-q4_k_m.gguf \
    -c 2048 \
    -ngl 99 \
    --host 0.0.0.0 \
    --port 8080
```

Verify: `curl http://localhost:8080/health` should return `{"status":"ok"}`

### 6. Install Piper TTS

```bash
# Download Piper binary (aarch64)
cd ~
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
mv piper ~/piper

# Download voice model
mkdir -p ~/piper/models
cd ~/piper/models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# Test
echo "Hello, I am your voice assistant." | ~/piper/piper \
    --model ~/piper/models/en_US-lessac-medium.onnx \
    --output_file /tmp/test_tts.wav
aplay /tmp/test_tts.wav
```

### 7. Install Python Application

```bash
cd ~/chatbot-jetson
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If `pyaudio` fails to install, ensure `portaudio19-dev` is installed first.

### 8. Configure

Edit `config.py` to match your paths:

```python
PIPER_EXECUTABLE = "/home/jetson/piper/piper"
PIPER_MODEL = "/home/jetson/piper/models/en_US-lessac-medium.onnx"
```

Adjust whisper model size if needed:
- `tiny` — fastest, lowest accuracy
- `base` — good balance (default)
- `small` — better accuracy, ~2x slower

### 9. Run

Make sure llama.cpp server is running in another terminal, then:

```bash
source venv/bin/activate
python main.py
```

## Usage

1. The chatbot starts and loads all models
2. Hold **SPACE** to speak
3. Release **SPACE** — your speech is transcribed and sent to the LLM
4. The response is spoken back to you
5. Press **Ctrl+C** to exit

## Performance Expectations

| Stage | Expected Time |
|-------|--------------|
| Whisper transcription (5s audio) | ~1-2s |
| LLM generation (short reply) | ~3-8s |
| Piper TTS synthesis | <1s |
| **Total round-trip** | **~5-11s** |

## Troubleshooting

### "Cannot reach LLM server"
- Ensure llama-server is running: `curl http://localhost:8080/health`
- Check port matches `config.py`

### "Piper executable not found"
- Verify path in `config.py` matches actual install location
- Ensure binary is executable: `chmod +x ~/piper/piper`

### No audio input detected
- Check microphone: `arecord -l`
- Set default device: `export AUDIODEV=hw:1,0` (adjust card number)

### CUDA out of memory
- Use smaller Whisper model: set `WHISPER_MODEL_SIZE = "tiny"` in config
- Reduce LLM context: change `-c 2048` to `-c 1024`
- Ensure no other GPU processes are running

### PyAudio installation fails
```bash
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

## File Structure

```
chatbot-jetson/
├── main.py              # Entry point — orchestrates the pipeline
├── audio_recorder.py    # Push-to-talk microphone recording
├── stt.py               # faster-whisper speech-to-text
├── llm_client.py        # HTTP client to llama.cpp server
├── tts.py               # Piper TTS text-to-speech
├── audio_player.py      # WAV audio playback
├── config.py            # All configurable parameters
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## License

MIT
