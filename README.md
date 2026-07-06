# Ghana Speech Datagen

Generate synthetic ASR training data using **VoxCPM.cpp** (GGML/CUDA/CPU) — no
PyTorch needed. Feeds texts from an HF dataset or file through the Ghana NLP
Community VoxCPM GGUF model, voice-cloning from a pool of reference audio.

> **GPU recommended** for usable speed. VoxCPM.cpp with CUDA achieves ~3×
> real-time (RTF 0.33) on an H200. Works on CPU too (`--backend cpu`) but is
> much slower.

## Supported languages

The `ghana-tts-36k` model supports **41+ Ghanaian languages**. See the model card
at [hf.co/ghananlpcommunity/ghana-tts-36k](https://huggingface.co/ghananlpcommunity/ghana-tts-36k)
for the full list.

## Setup

### Prerequisites

- Linux with GCC/G++
- CUDA 12+ (optional — for GPU backend)
- CMake ≥ 3.22
- Python 3.10+

### 1. Install VoxCPM.cpp

```bash
git clone https://github.com/your-org/voxcpm-cpp.git
cd voxcpm-cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON   # omit -DGGML_CUDA=ON for CPU-only
make -j$(nproc) voxcpm-server voxcpm_tts
```

### 2. Download the GGUF model

```bash
# Q8_0 quantized model with F16 AudioVAE (~850 MB)
cd voxcpm-cpp
mkdir -p models
wget -O models/ghana-tts-36k-q8_0-audiovae-f16.gguf \
  https://huggingface.co/walusungungulube/ghana-tts-36k-gguf/resolve/main/ghana-tts-36k-q8_0-audiovae-f16.gguf
```

### 3. Install ghana-speech-datagen

```bash
git clone https://github.com/ghananlpcommunity/ghana-speech-datagen.git
cd ghana-speech-datagen
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set the `VOXCPM_SERVER_BIN` environment variable so the tool can find the server:

```bash
export VOXCPM_SERVER_BIN=/path/to/voxcpm-cpp/build/examples/voxcpm-server
```

## Quickstart — ASR

Synthesise speech texts using reference audio for voice cloning. Provide
**texts to synthesise** and a **reference audio source** (HF dataset or local
directory) — the model speaks each text in the voice of a randomly-selected
reference.

```bash
# Texts from HF dataset, ref audio from another HF dataset
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --hours 5

# Texts from a .txt file, ref audio from local dir + metadata
ghana-speech-datagen asr --text-file sentences.txt \
    --ref-audio-dir my_refs/ --ref-metadata refs.csv \
    --hours 2

# Sub-sample texts, use CPU backend
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --max-samples 2000 --backend cpu

# Push result to a new HF dataset repo
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --hours 10 --push my-asr-repo
```

## Output

```
data/<name>/
  wavs/<id>.wav            mono 16-bit PCM, at --sample-rate (default 22050)
  manifest.jsonl           full info: id, file, text, duration, ref_audio, ref_text
  metadata.jsonl           ASR manifest:  {"audio":"...","text":"..."}
```

## Options

| flag | meaning |
|------|---------|
| `--dataset ID` / `--text COL` | source: an HF dataset with text to synthesise |
| `--text-file PATH` | source: a .txt file with text to synthesise |
| `--config` / `--split` | dataset config / split (default `train`) |
| `--ref-dataset ID` | HF dataset with reference audio+transcript columns |
| `--audio-column COL` | column with reference audio (default `audio`) |
| `--ref-text-column COL` | column with reference transcripts (default `text`) |
| `--ref-config` / `--ref-split` | ref dataset config / split |
| `--ref-audio-dir DIR` | local dir with ref audio (use with `--ref-metadata`) |
| `--ref-metadata PATH` | CSV/JSONL mapping ref audio filenames to transcripts |
| `--hours H` | target hours of audio to generate |
| `--min-duration` / `--max-duration` | drop generated clips outside this range (seconds) |
| `--max-samples N` | randomly pick at most this many texts |
| `--min-samples N` | minimum valid samples required (default 50) |
| `--sample-rate HZ` | output WAV rate (default 22050) |
| `--cfg` | CFG value passed to the server (default 2.0) |
| `--backend cuda\|cpu` | inference backend (default `cuda`) |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO` | upload the finished run to an HF dataset repo (public) |
| `--private` | make the pushed repo private instead |
| `--token` | HF token — for gated datasets/models |

## Use as a library

```python
from ghana_speech_datagen.generator import generate_asr

pairs = [
    ("Hello world", "/refs/spk1.wav", "Prompt text one"),
    ("How are you", "/refs/spk2.wav", "Prompt text two"),
]

summary = generate_asr(
    out_dir="data/my-run",
    pairs=pairs,
    target_seconds=7200,
    sample_rate=16000,
    backend="cuda",
    on_clip=lambda dur: print(f"Generated {dur:.1f}s"),
)
# {'rows': ..., 'hours': ..., 'skipped': ..., 'duration_dropped': ...}
```

For direct server control:

```python
from ghana_speech_datagen.voxcpm_cpp import VoxCPMCppServer

with VoxCPMCppServer(voice_dir=".voices", backend="cuda") as server:
    server.wait_until_ready()
    server.register_voice("spk1", "/refs/spk1.wav", "Prompt text")
    wav_bytes = server.synthesize("spk1", "Text to speak", response_format="wav")
```

## Performance

- **Backend** — VoxCPM.cpp with GGML CUDA kernels. On an H200, RTF is ~0.33
  (~55 steps/s, 11.4 s audio generated in 3.7 s wall time).
- **CPU fallback** — set `--backend cpu` for CPU-only environments (much slower).
- **Sample rate.** The model synthesises at **16 kHz**; output is resampled to
  `--sample-rate`. Upsampling beyond 16 kHz doesn't add acoustic bandwidth.
- **Single server process.** A single `voxcpm-server` handles all synthesis
  requests. No parallel model instances needed — the GGUF model is already
  highly optimized.

## Tests

```bash
pip install pytest
pytest tests/
```

## Project layout

```
ghana_speech_datagen/
  cli.py             the `ghana-speech-datagen` command
  generator.py       ASR generation loop (generate_asr)
  voxcpm_cpp.py      VoxCPMCppServer wrapper (manages server subprocess)
  speakers/          built-in male/female reference wav + text
examples/
  ghana_speech_datagen.ipynb   Colab runner (currently PyTorch-based, WIP)
  modal_run.py                Modal runner (WIP)
tests/
```

## License

CC-BY-4.0
