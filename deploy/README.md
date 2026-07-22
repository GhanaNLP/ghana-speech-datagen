# Deploying the VoxCPM2 TTS server

The datagen (`ghana-speech-datagen`) is a **pure HTTP client**. The TTS model —
[`ghananlpcommunity/VoxCPM2-Ghana`](https://huggingface.co/ghananlpcommunity/VoxCPM2-Ghana),
a 2B VoxCPM2 fine-tune covering 40+ Ghanaian languages — runs as a standalone
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) server on a GPU. Deploy
it once, then generate as many datasets as you like against its API.

## 1. Prerequisites

- Linux + an NVIDIA GPU supported by vLLM (works on anything from a 24 GB L4 up
  to an H200; the config right-sizes the KV cache to fit small cards).
- Python 3.10+ and the `uv` package manager (or plain `pip`).

## 2. Install vLLM-Omni

```bash
uv pip install "vllm==0.24.0" --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni && uv pip install -e .
```

> Prefer the latest `main` if the pinned version drifts — the project moves fast.

## 3. Start the server

From this `deploy/` directory:

```bash
# Simplest — pulls the model from HF on first run, open (no auth), port 8000
bash serve.sh

# With an API key + a specific port / GPU budget
API_KEY=my-secret PORT=8000 GPU_MEM=0.9 bash serve.sh

# Serve a local checkpoint instead of the HF model
MODEL=/path/to/VoxCPM2-Ghana bash serve.sh
```

First launch downloads ~10 GB of weights (set `HF_TOKEN` for faster, rate-limit-free
downloads) and takes a couple of minutes to load onto the GPU. When ready:

```bash
curl -s http://localhost:8000/health         # -> 200
```

## 4. Point the datagen at it

```bash
export TTS_SERVER_URL=http://localhost:8000    # or a remote host / tunnel URL
export TTS_API_KEY=my-secret                   # only if you set API_KEY above

ghana-speech-datagen tts --lang ewe --hours 5
# or per-invocation:
ghana-speech-datagen asr --lang ewe --hours 5 \
    --server-url http://gpu-host:8000 --api-key my-secret
```

## API (OpenAI-compatible speech endpoint)

`serve.sh` exposes the standard vLLM-Omni speech API — usable directly, too:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "voxcpm2",
        "input": "Akwaaba, wo ho te sɛn?",
        "ref_audio": "data:audio/wav;base64,<base64 wav>",
        "ref_text": "reference transcript",
        "response_format": "wav"
      }' --output out.wav
```

- **Voice cloning** is inline per request via `ref_audio` (a base64 `data:` URI,
  an `http(s)` URL, or a `file:` path) + `ref_text`. The server caches resolved
  references by hash, so reusing a small reference pool across many texts is cheap.
- **Language** is inferred from the text's script and the reference voice — there
  is no `<|lang:|>` tag (that was the old VoxCPM v1 mechanism).
