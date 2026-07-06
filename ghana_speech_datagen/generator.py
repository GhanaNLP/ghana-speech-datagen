"""Core synthetic-speech generation using VoxCPM.cpp (GGML/CUDA/CPU).

Streams text-reference pairs, synthesises each row with the Ghana NLP Community
VoxCPM GGUF model via VoxCPM.cpp, and writes WAVs at the chosen sample rate +
manifests.  Supports CPU and CUDA backends.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000       # native rate the model synthesises at
DEFAULT_SR = 22050        # default OUTPUT rate (TTS-friendly); override with --sample-rate

_SPEAKER_DIR = Path(__file__).resolve().parent / "speakers"
SPEAKERS: dict[str, dict] = {}
for _g in ("male", "female"):
    _wav = _SPEAKER_DIR / f"{_g}.wav"
    _txt = _SPEAKER_DIR / f"{_g}.txt"
    if _wav.is_file() and _txt.is_file():
        SPEAKERS[_g] = {"wav": str(_wav), "text": _txt.read_text(encoding="utf-8").strip()}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def pick_gender(idx: int, mode: str, male_pct: int) -> str:
    if mode in ("male", "all male"):
        return "male"
    if mode in ("female", "all female"):
        return "female"
    return "male" if (idx * 2654435761) % 100 < male_pct else "female"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-") or "run"


def normalize_audio(audio_input, out_dir: str) -> str:
    """Read audio (HF dict or path), convert to 16 kHz mono 16-bit WAV, return path."""
    os.makedirs(os.path.join(out_dir, "_normalized"), exist_ok=True)

    if isinstance(audio_input, dict):
        arr = audio_input.get("array")
        sr = audio_input.get("sampling_rate", SAMPLE_RATE)
        if arr is not None:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != SAMPLE_RATE:
                arr = librosa.resample(arr, orig_sr=int(sr), target_sr=SAMPLE_RATE)
        else:
            arr, _ = librosa.load(audio_input["path"], sr=SAMPLE_RATE, mono=True)
        h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        src = ("array", arr)
    else:
        p = str(audio_input)
        h = hashlib.sha256(p.encode()).hexdigest()[:16]
        src = ("path", p)

    out_path = os.path.join(out_dir, "_normalized", f"{h}.wav")
    if not os.path.isfile(out_path):
        wav = src[1] if src[0] == "array" else librosa.load(src[1], sr=SAMPLE_RATE, mono=True)[0]
        tmp = out_path + ".tmp.wav"
        sf.write(tmp, wav, SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp, out_path)
    return out_path


def resample(wav, src_sr: int, dst_sr: int):
    wav = np.asarray(wav, dtype=np.float32)
    if src_sr == dst_sr or wav.size == 0:
        return wav
    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr)


# --------------------------------------------------------------------------- #
# ASR generation (VoxCPM.cpp)
# --------------------------------------------------------------------------- #
def generate_asr(
    *,
    out_dir: str,
    pairs: list,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    min_samples: int = 50,
    target_seconds: float = 3600,
    sample_rate: int = DEFAULT_SR,
    cfg_value: float = 2.0,
    on_clip=None,
    progress=None,
    backend: str = "cuda",
) -> dict:
    """Generate synthetic speech from texts using VoxCPM.cpp.

    ``pairs`` is a list of ``(text_to_synthesise, ref_audio, ref_text)`` tuples.
    For each text, a random reference audio is used as the voice prompt.
    Output is ASR format (``wavs/`` + ``manifest.jsonl`` + ``metadata.jsonl``).

    ``backend`` -- ``"cuda"`` (default) or ``"cpu"``.
    """
    from ghana_speech_datagen.voxcpm_cpp import VoxCPMCppServer

    out_dir = str(out_dir)
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # Normalise all reference audio up front
    ref_paths: dict[int, tuple[str, str]] = {}
    for idx, (_, audio_input, ref_text) in enumerate(pairs):
        try:
            path = normalize_audio(audio_input, out_dir)
            ref_paths[idx] = (path, ref_text)
        except Exception:
            pass

    if len(ref_paths) < min_samples:
        raise RuntimeError(
            f"Only {len(ref_paths)} valid reference audios "
            f"(need >={min_samples}). Aborting."
        )

    with VoxCPMCppServer(
        voice_dir=os.path.join(out_dir, ".voxcpm-voices"),
        max_decode_steps=1024,
        backend=backend,
    ) as server:
        server.wait_until_ready(timeout=120.0)

        voice_ids = []
        for idx, (wav_path, ref_text) in ref_paths.items():
            vid = f"ref_{idx}"
            try:
                server.register_voice(vid, wav_path, ref_text)
                voice_ids.append(vid)
            except Exception:
                pass

        if len(voice_ids) < min_samples:
            raise RuntimeError(
                f"Only registered {len(voice_ids)} voices "
                f"(need >={min_samples}). Aborting."
            )

        valid: list[dict] = []
        skipped = 0
        duration_dropped = 0
        total_sec = 0.0

        for idx, (text, _, _) in enumerate(pairs):
            if total_sec >= target_seconds:
                break
            if not text:
                skipped += 1
                continue

            vid = voice_ids[idx % len(voice_ids)]
            try:
                wav_bytes = server.synthesize(vid, text, response_format="wav")
                data, sr = sf.read(io.BytesIO(wav_bytes))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                wav = np.asarray(data, dtype=np.float32)
            except Exception:
                skipped += 1
                continue

            wav = resample(wav, SAMPLE_RATE, int(sample_rate))
            dur = float(len(wav)) / sample_rate

            if dur < min_duration or dur > max_duration:
                duration_dropped += 1
                continue

            uid = f"{idx:07d}_{uuid.uuid4().hex[:8]}"
            rel = f"wavs/{uid}.wav"
            out = os.path.join(wav_dir, f"{uid}.wav")
            tmp = out + ".tmp.wav"
            sf.write(tmp, wav, int(sample_rate), subtype="PCM_16")
            os.replace(tmp, out)

            valid.append({
                "id": uid,
                "file": rel,
                "text": text,
                "duration": round(dur, 3),
            })
            total_sec += dur

            if on_clip:
                on_clip(total_sec)

    if len(valid) < min_samples:
        raise RuntimeError(
            f"Only {len(valid)} valid samples (need >={min_samples}). "
            f"{skipped} skipped, {duration_dropped} dropped by duration. Aborting."
        )

    with open(Path(out_dir) / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(Path(out_dir) / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in valid:
            f.write(
                json.dumps({"audio": r["file"], "text": r["text"]}, ensure_ascii=False)
                + "\n"
            )

    return {
        "rows": len(valid),
        "hours": total_sec / 3600,
        "skipped": skipped,
        "duration_dropped": duration_dropped,
        "out_dir": out_dir,
    }


# --------------------------------------------------------------------------- #
# Export manifests
# --------------------------------------------------------------------------- #
EXPORT_FORMATS = ("ljspeech", "asr")


def _manifest_text(s: str) -> str:
    return s.replace("\r", " ").replace("\n", " ").replace("|", " ").strip()


def export_formats(out_dir: str, formats) -> list[str]:
    out = Path(out_dir)
    rows = []
    with open(out / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    fmts = [f for f in formats if f in EXPORT_FORMATS]
    written: list[str] = []

    def _write(name, lines):
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(str(out / name))

    if "ljspeech" in fmts:
        _write("metadata.csv",
               [f"{r['id']}|{_manifest_text(r['text'])}|{_manifest_text(r['text'])}" for r in rows])
    if "asr" in fmts:
        _write("metadata.jsonl",
               [json.dumps({"audio": r["file"],
                            "text": _manifest_text(r["text"])})
                for r in rows])
    return written
