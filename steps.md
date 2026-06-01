# 1. download hf cli
```
curl -LsSf https://hf.co/cli/install.sh | bash
```

# 2. download qwen3
```
 env PATH="/home/trung/.local/bin:$PATH" HF_DEBUG=1 hf  download unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_K_M.gguf
``` 

# 3. run qwen3
```
docker run -it --rm --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-orin llama-server -m /root/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B-GGUF/snapshots/d7f544eead698dbd1f15126ef60b45a1e1933222/Qwen3-1.7B-Q4_K_M.gguf --host 0.0.0.0 --port 8080
```

```
arecord -D hw:2,0 -d 5 -f S16_LE -r 16000 -c 1 /tmp/test.wav
```
```
aplay -D plughw:2,0 /tmp/test.wav
```