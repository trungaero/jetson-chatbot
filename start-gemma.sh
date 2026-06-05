#!/bin/bash
# ============================================================
# Gemma Server Startup Script
# ============================================================

CONTAINER_NAME="gemma-server"
IMAGE_NAME="ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin"
HF_CACHE="${HOME}/.cache/huggingface"
MODEL_PATH="/root/.cache/huggingface/hub/gemma-4-E2B-it-Q4_K_M.gguf"

echo "[$(date)] Starting $CONTAINER_NAME..."

# -- Stop and remove any existing container ------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[$(date)] Removing existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null
    docker rm "$CONTAINER_NAME" 2>/dev/null
fi

# -- Start the gemma server container ------------------------
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --runtime nvidia \
    --network host \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "$IMAGE_NAME" \
    llama-server \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8080 \
        --n-gpu-layers 99 \
        --ctx-size 2048 \
        --reasoning off

echo "[$(date)] $CONTAINER_NAME started, waiting for server to be ready..."

# -- Wait until llama-server is accepting requests -----------
RETRIES=60
until curl -sf http://localhost:8080/health > /dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        echo "[$(date)] ERROR: gemma-server not reachable after 60 attempts. Aborting."
        exit 1
    fi
    echo "[$(date)] gemma-server not ready yet, retrying in 2s... ($RETRIES attempts left)"
    sleep 2
done

echo "[$(date)] gemma-server is ready on port 8080."