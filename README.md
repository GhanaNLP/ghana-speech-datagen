# Ghana Speech Datagen

Generate synthetic speech training data using **VoxCPM.cpp** (GGML/CUDA/CPU) — no
PyTorch needed. Feeds text through the Ghana NLP Community VoxCPM GGUF model,
voice-cloning from reference audio.

**You don't have to bring your own text.** Just pass `--lang` and the tool pulls
default text (and reference voices) for that language automatically:

```bash
ghana-speech-datagen tts --lang ewe --hours 5   # → TTS dataset (LJSpeech)
ghana-speech-datagen asr --lang ewe --hours 5   # → ASR dataset (JSONL manifest)
```

This uses the model's language tag (`<|lang:ewe|> …`, exactly as the model was
trained) so pronunciation matches the language. You can still bring your own text
(`--dataset`/`--text-file`) and reference audio when you want to.

## Two modes

Both modes synthesise speech from text; they differ in the **reference voices**
they use and the **output format** they write — each ready for its use case.

| | `tts` | `asr` |
|---|---|---|
| **Voices** | a small speaker set (the packaged male/female voices by default, or your own) | a large, diverse pool of reference audio (min. `--min-samples`, default 50) |
| **Best for** | building a TTS voice/dataset with consistent speakers | building ASR training data with many speakers for robustness |
| **Reference** | `--voices`, `--speaker-dir`, `--speaker` | `--ref-dataset`, `--ref-audio-dir`, or in-language default pool |
| **Output** | LJSpeech: `wavs/` + `metadata.csv` (`id\|text\|text`) | `wavs/` + `metadata.jsonl` (`{"audio","text"}`) |
| **Default rate** | 22050 Hz (TTS standard) | 22050 Hz |

Every clip is also recorded in `manifest.jsonl` (full record, including
`speaker` for `tts`).

> **GPU recommended** for usable speed. VoxCPM.cpp with CUDA achieves ~3×
> real-time (RTF 0.33) on an H200. Works on CPU too (`--backend cpu`) but is
> much slower.

## Supported languages

The `ghana-tts-36k` model supports **41+ Ghanaian languages** (plus English). Every
language ships with a built-in default text source, so `--lang <code>` is all you
need. List them with:

```bash
ghana-speech-datagen asr --list-langs
```

Codes are the same tags the model was trained with — e.g. `ewe`, `fat`, `dag`,
`twi-asante`, `twi-akuapem`, `en`. `--lang` also accepts a full config name
(`Ewe_ewe`) or display name (`Asante Twi`).

### Adding more text sources

Default text comes from [`ghananlpcommunity/ghana-speech`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech).
To give a language extra text, add one line to `_EXTRA_SOURCES` in
[`ghana_speech_datagen/text_sources.py`](ghana_speech_datagen/text_sources.py).
Twi, for example, also draws from a 500-hour health corpus:

```python
_EXTRA_SOURCES = {
    "twi-asante": [
        TextSource("ghananlpcommunity/twi-health-asr-gemini-500hrs",
                   text_column="transcription"),
    ],
    # "ewe": [TextSource("your-org/your-ewe-text", text_column="text")],
}
```

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

## Quickstart — TTS

Synthesise a TTS dataset voiced by a small, consistent speaker set. By default it
uses the packaged male/female voices — no reference audio needed.

```bash
# Simplest: default text for a language, packaged male + female voices
ghana-speech-datagen tts --lang ewe --hours 5

# One voice only
ghana-speech-datagen tts --lang ewe --voices female --hours 5

# Your own speakers: a dir of NAME.wav + NAME.txt (prompt) pairs
ghana-speech-datagen tts --lang ewe --speaker-dir my_voices/ --hours 5

# A single custom speaker
ghana-speech-datagen tts --lang ewe \
    --speaker ref.wav --speaker-text "the reference transcript" --hours 2

# Your own text file, tagged with a language for correct pronunciation
ghana-speech-datagen tts --text-file sentences.txt --lang ewe --hours 2
```

Output is **LJSpeech** format (`wavs/` + `metadata.csv`), ready for most TTS
trainers (Coqui TTS, VITS, Tacotron, …).

## Quickstart — ASR

The model speaks each text in the voice of a randomly-selected reference clip
from a large pool. Both the text and the reference audio have sensible defaults
per language.

```bash
# Simplest: default text + in-language reference voices for a language
ghana-speech-datagen asr --lang ewe --hours 5

# Default text for Twi (ghana-speech + health corpus), your own reference voices
ghana-speech-datagen asr --lang twi-asante \
    --ref-dataset org/ref-audio-ds --ref-text-column text --hours 5

# List every supported language and its text sources
ghana-speech-datagen asr --list-langs
```

You can also bring your own text and/or reference audio:

```bash
# Texts from HF dataset, ref audio from another HF dataset
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --hours 5

# Your own text file, but tag it with a language so pronunciation is correct
ghana-speech-datagen asr --text-file sentences.txt --lang ewe \
    --ref-dataset org/ref-audio-ds --hours 2

# Texts from a .txt file, ref audio from local dir + metadata
ghana-speech-datagen asr --text-file sentences.txt \
    --ref-audio-dir my_refs/ --ref-metadata refs.csv \
    --hours 2

# Sub-sample texts, use CPU backend
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --max-samples 2000 --backend cpu

# Send it to a specific HF dataset repo (instead of the auto-named one)
ghana-speech-datagen asr --dataset org/text-ds --text text \
    --ref-dataset org/ref-audio-ds --ref-text-column text \
    --hours 10 --push my-asr-repo
```

## Uploading to Hugging Face

**Both modes auto-push to the Hub by default**, incrementally, as clips are
generated — so a long run keeps a live copy on HF even if it's interrupted.

- The repo is auto-named `you/ghana-speech-synth-<name>`; override it with `--push REPO_ID`.
- `--save-every N` controls how often the partial dataset is flushed and pushed (default 200 clips).
- `--private` makes the repo private.
- **`--no-push` disables uploading** — generate locally only (no HF token needed).

```bash
# Local only, nothing uploaded
ghana-speech-datagen tts --lang ewe --hours 5 --no-push

# Push to a private repo, uploading every 500 clips
ghana-speech-datagen asr --lang ewe --hours 20 --private --save-every 500
```

## Output

```
data/<name>/
  wavs/<id>.wav            mono 16-bit PCM, at --sample-rate (default 22050)
  manifest.jsonl           full record per clip: id, file, text, duration (+ speaker for tts)

  # tts writes (LJSpeech, the standard TTS layout):
  metadata.csv             id|text|normalized_text

  # asr writes:
  metadata.jsonl           {"audio":"wavs/...","text":"..."}
```

The transcript in the manifests is the **clean spoken text** — the `<|lang:…|>`
tag is only a synthesis control signal and is never written to disk.

## Options

| flag | meaning |
|------|---------|
| `--lang CODE` | use built-in default text for a language; also sets the model's `<|lang:CODE|>` tag and defaults the reference pool to in-language audio |
| `--list-langs` | list supported languages and their default text sources |
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
| `--backend cuda\|cpu` | inference backend (auto-detected if omitted) |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO` | override the auto-named HF dataset repo to push to |
| `--no-push` | disable the default auto-push; generate locally only |
| `--save-every N` | flush + push every N clips as they're generated (default 200) |
| `--private` | make the pushed repo private instead |
| `--token` | HF token — for gated datasets/models and pushing |

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
    on_clip=lambda dur: print(f"Generated {dur:.1f}s"),
)
# {'rows': ..., 'hours': ..., 'skipped': ..., 'duration_dropped': ...}
```

For direct server control:

```python
from ghana_speech_datagen.voxcpm_cpp import VoxCPMCppServer

with VoxCPMCppServer(voice_dir=".voices") as server:
    server.wait_until_ready()
    server.register_voice("spk1", "/refs/spk1.wav", "Prompt text")
    wav_bytes = server.synthesize("spk1", "Text to speak", response_format="wav")
```

## Performance

- **Backend** — VoxCPM.cpp with GGML CUDA kernels. On an H200, RTF is ~0.33
  (~55 steps/s, 11.4 s audio generated in 3.7 s wall time).
- **CPU fallback** — backend is auto-detected (runs `nvidia-smi`); override with `--backend cpu`.
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
