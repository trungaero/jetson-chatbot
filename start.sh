#!/bin/bash
# ============================================================
# Jetson Chatbot Startup Script
# ============================================================

CONTAINER_NAME="jetson-chatbot"
IMAGE_NAME="jetson-chatbot"
PIPER_MODELS="${HOME}/piper/models"
HF_CACHE="${HOME}/.cache/huggingface"
TORCH_CACHE="${HOME}/chatbot-jetson/jetson-chatbot/data/models/torch/hub"

echo "[$(date)] Starting $CONTAINER_NAME..."

# -- Stop and remove any existing container ------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[$(date)] Removing existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null
    docker rm "$CONTAINER_NAME" 2>/dev/null
fi

# -- Wait for llama-server to be ready -----------------------
echo "[$(date)] Waiting for llama-server on port 8080..."
RETRIES=30
until curl -sf http://localhost:8080/health > /dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        echo "[$(date)] ERROR: llama-server not reachable after 30 attempts. Aborting."
        exit 1
    fi
    echo "[$(date)] llama-server not ready yet, retrying in 2s... ($RETRIES attempts left)"
    sleep 2
done
echo "[$(date)] llama-server is ready."

# -- Detect camera devices on the host -----------------------
CAMERA_DEVICES=""
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        CAMERA_DEVICES="$CAMERA_DEVICES --device $dev"
        echo "[$(date)] Found camera device: $dev"
    fi
done

if [ -z "$CAMERA_DEVICES" ]; then
    echo "[$(date)] WARNING: No /dev/video* devices found. Camera tool will not work."
fi

# -- Start the container -------------------------------------
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --runtime nvidia \
    --network host \
    --device /dev/snd \
    $CAMERA_DEVICES \
    --group-add video \
    -v "${PIPER_MODELS}:/opt/piper/models" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${TORCH_CACHE}:/data/models/torch/hub" \
    "$IMAGE_NAME"

echo "[$(date)] $CONTAINER_NAME started."