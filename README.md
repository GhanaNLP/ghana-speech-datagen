# Ghana TTS DataGen — synthetic speech data generator

Turn a **text dataset** into **synthetic TTS training audio** — streamed through
the Ghana NLP Community VoxCPM model (`ghana-tts-36k`), voice-cloning built-in
male/female reference speakers. It writes WAVs (resampled to your target rate) +
a manifest locally, runs multiple model instances in parallel, and resumes where
it left off.

> **A GPU is required** for usable speed (VoxCPM is a neural TTS model; ~4.5 GB
> VRAM per instance). A T4 works; bigger GPUs run more instances in parallel.

## Supported languages

The `ghana-tts-36k` model supports **41+ Ghanaian languages**. See the model card
at [hf.co/ghananlpcommunity/ghana-tts-36k](https://huggingface.co/ghananlpcommunity/ghana-tts-36k)
for the full list.

## Run in the cloud (free T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ghananlpcommunity/ghana-tts-datagen/blob/main/notebooks/ghana_tts_datagen.ipynb)

Pick a **GPU** runtime (Colab: `Runtime → Change runtime type → T4 GPU`).

## Install (local)

> **Local use needs an NVIDIA GPU.** Without one, generation is impractically
> slow — use the **Colab notebook above** (free T4) instead. Clone locally only
> if you have a GPU.

```bash
git clone https://github.com/ghananlpcommunity/ghana-tts-datagen.git
cd ghana-tts-datagen
python3 -m venv .venv && source .venv/bin/activate
sudo apt-get install -y ffmpeg          # system dependency
pip install -e .                        # gives you the `ghana-tts-datagen` command
```

## Quickstart

Source is **either** an HF dataset column **or** a plain text file (one sentence
per line). Output is written in your chosen **TTS format** (`--formats`).

```bash
# Preview 5 clips first (hear it before a big run)
ghana-tts-datagen --dataset ghananlpcommunity/your-text-dataset --text-column text --preview 5

# From an HF dataset → 5 h, LJSpeech layout (default), into data/<name>
ghana-tts-datagen --dataset ghananlpcommunity/your-text-dataset --text-column text \
    --hours 5 --name twi-run --formats ljspeech

# From your own sentences (one per line) → a Piper dataset
ghana-tts-datagen --text-file sentences.txt --hours 2 --formats piper

# Multiple formats at once, then push to your HF repo
ghana-tts-datagen --text-file sentences.txt --hours 2 --formats ljspeech,vits,melo \
    --lang TWI --push you/my-synth

# Resume: re-run the same command — finished rows are skipped
```

## Output — ready for your TTS trainer

Everything lands in `data/<name>/` (override with `--out`):

```
data/twi-run/
  wavs/<id>.wav            mono, silence-trimmed, at --sample-rate (default 22050)
  manifest.jsonl           full info: id, file, text, gender, speaker, duration
  progress.json            resume state (re-run to continue)
  # + the manifest(s) for the formats you asked for:
  metadata.csv             ljspeech  →  id|text|text
  metadata.piper.csv       piper     →  id|speaker|text   (metadata.csv if piper only)
  filelist.txt, speakers.txt   vits  →  wavs/<id>.wav|sid|text
  metadata.list            melo      →  wavs/<id>.wav|speaker|LANG|text
```

Point your framework's data-prep at this folder — `wavs/` + the matching manifest
is exactly what LJSpeech/Coqui, Piper, VITS, and MeloTTS expect.

## Options

| flag | meaning |
|------|---------|
| `--dataset ID` / `--text-column COL` | source: an HF dataset column |
| `--text-file PATH` | source: a .txt file, one sentence per line |
| `--config` / `--split` | dataset config / split (default split `train`) |
| `--hours H` | target hours of audio to generate |
| `--voices custom\|male\|female` | speaker selection (default `custom`) |
| `--male-pct N` | %% male in `custom` mode (deterministic per row) |
| `--max-chars N` | skip rows longer than this (default 400) |
| `--sample-rate HZ` | output WAV rate (default 22050; e.g. 24000 for MeloTTS, 44100) |
| `--precision fp32\|fp16\|bf16` | model precision (default fp32) — see Performance |
| `--instances N` | parallel model instances (default: auto by VRAM) |
| `--cfg` / `--steps` | CFG value / inference timesteps |
| `--formats …` | TTS manifests to write: `ljspeech,piper,vits,melo` (default `ljspeech`) |
| `--lang CODE` | language code for the `melo` manifest |
| `--name` / `--out` | run name (→ `data/<name>`) or explicit output dir |
| `--push REPO [--private]` | upload the finished run to an HF dataset repo |
| `--token` | HF token (else `HF_TOKEN` env) — for gated datasets/models |
| `--preview N` | generate N preview clips and exit |
| `--list-datasets` | list datasets under the `ghananlpcommunity` org |

Resuming is automatic: point `--name`/`--out` at an existing run folder (or just
re-run the same command) and it reads `progress.json` and skips finished rows.

## Performance & GPU

- **Parallel instances.** Several model copies pull rows off a shared queue
  (~4.5 GB VRAM each in fp32). It auto-sizes by VRAM — e.g. **2 instances on a
  T4** (16 GB). Push harder with `--instances 3` if it's stable; a T4 often
  becomes compute-bound past that.
- **Precision** (`--precision`):
  - `fp32` — default, safest, highest quality.
  - `fp16` — ~half the VRAM (so ~2× the instances) and faster on most GPUs, **but
    may degrade quality or NaN** on TTS models; preview before committing.
  - `bf16` — ~half the VRAM, more numerically stable than fp16, **but needs an
    Ampere+ GPU (A100/L4/H100) — not a T4**.
- **Sample rate.** The model synthesises at **16 kHz**; output is resampled to
  `--sample-rate` so files match your framework, but true bandwidth stays ~8 kHz
  (upsampling doesn't add detail).

## Use as a library

```python
from ghana_tts_datagen import generate, export_formats

summary = generate(out_dir="data/run", dataset="org/ds", text_column="text",
                   target_hours=5, voices="custom", male_pct=50)
export_formats("data/run", ["ljspeech", "vits"], lang="TWI")
print(summary)
```

## Tests

```bash
pip install pytest
pytest tests/
```

## Project layout

```
ghana_tts_datagen/
  cli.py             the `ghana-tts-datagen` command
  generator.py       voice-clone, silence-trim, parallel run, resume, TTS-format export
  speakers/          built-in male/female reference wav + text
notebooks/ghana_tts_datagen.ipynb   Colab (GPU) runner
tests/
```

## License

CC-BY-4.0
