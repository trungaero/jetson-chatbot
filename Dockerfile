# ============================================================
# Jetson Orin Nano Voice Chatbot
# Base: NVIDIA L4T PyTorch (JetPack 6.x, Ubuntu 22.04, aarch64)
# ============================================================
FROM dustynv/l4t-pytorch:r36.2.0

# -- Labels --------------------------------------------------
LABEL maintainer="trungaero"
LABEL description="Offline voice chatbot: faster-whisper + llama.cpp + Piper TTS"
LABEL org.opencontainers.image.source="https://github.com/trungaero/jetson-chatbot"

# -- Environment ---------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Piper paths (can be overridden at runtime via -e)
    PIPER_HOME=/opt/piper \
    PIPER_MODELS=/opt/piper/models \
    # HuggingFace cache — faster-whisper models are stored here.
    # Mount your host HF cache with: -v ~/.cache/huggingface:/root/.cache/huggingface
    HF_HOME=/root/.cache/huggingface \
    # llama.cpp server endpoint (override with -e if running externally)
    LLAMA_SERVER_URL=http://localhost:8080

# -- System dependencies -------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Audio
        portaudio19-dev \
        libasound2-dev \
        alsa-utils \
        # Build tools (needed by llama.cpp & pyaudio wheel)
        cmake \
        ninja-build \
        build-essential \
        git \
        wget \
        curl \
        # Misc
        ffmpeg \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# -- Python dependencies -------------------------------------
WORKDIR /app
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip --index-url https://pypi.org/simple && \
    pip3 install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://pypi.ngc.nvidia.com \
        -r requirements.txt

# -- Piper TTS binary (aarch64) ------------------------------
RUN mkdir -p ${PIPER_HOME} ${PIPER_MODELS} && \
    wget -q -O /tmp/piper.tar.gz \
        https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz && \
    tar -xzf /tmp/piper.tar.gz -C ${PIPER_HOME} --strip-components=1 && \
    chmod +x ${PIPER_HOME}/piper && \
    rm /tmp/piper.tar.gz

# -- Piper voice models are mounted from the host at runtime -
# Models directory is intentionally left empty in the image.
# Mount your host models folder with:
#   -v /path/on/host/piper/models:/opt/piper/models

# -- Copy application source ---------------------------------
COPY . .

# -- Declare mount points ------------------------------------
# These directories are expected to be bind-mounted from the host.
VOLUME ["/opt/piper/models", "/root/.cache/huggingface"]

# -- Runtime configuration -----------------------------------
# Audio device access:    --device /dev/snd
# llama.cpp server:       -e LLAMA_SERVER_URL=http://host.docker.internal:8080
# Piper models (mount):   -v ~/piper/models:/opt/piper/models
# HuggingFace cache:      -v ~/.cache/huggingface:/root/.cache/huggingface
#
# Full example:
#   docker run --rm -it --runtime nvidia \
#     --device /dev/snd \
#     -e LLAMA_SERVER_URL=http://host.docker.internal:8080 \
#     -v ~/piper/models:/opt/piper/models \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     jetson-chatbot

CMD ["python3", "main.py"]