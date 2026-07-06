"""Command-line interface for Ghana Speech Datagen.

Subcommands:
  asr   Generate synthetic speech from text (needs GPU)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import soundfile as sf

from .generator import DEFAULT_SR, sanitize_name

DATASET_ORG = "ghananlpcommunity"
MIN_ASR_SAMPLES = 50


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

DEFAULT_MIN_REF_DURATION = 1.0
DEFAULT_MAX_REF_DURATION = 15.0


def _validate_ref_duration(duration: float, label: str,
                            min_dur: float | None, max_dur: float | None):
    if min_dur is not None and duration < min_dur:
        sys.exit(f"Reference audio '{label}' is {duration:.1f}s (minimum {min_dur}s). "
                 f"Use a longer clip or lower --min-ref-duration.")
    if max_dur is not None and duration > max_dur:
        sys.exit(f"Reference audio '{label}' is {duration:.1f}s (maximum {max_dur}s). "
                 f"Use a shorter clip or raise --max-ref-duration.")


def _get_audio_duration(audio) -> float:
    """Return duration in seconds from an HF audio dict or file path."""
    if isinstance(audio, dict):
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is not None and sr:
            return float(len(arr)) / float(sr)
        path = audio.get("path", "")
        if path:
            return float(sf.info(path).duration)
        raise ValueError("Cannot determine duration from audio dict")
    return float(sf.info(str(audio)).duration)


def _resolve_token(args) -> str:
    tok = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            import getpass
            tok = getpass.getpass(
                "HF Token (required -- needed to push to your HF account): "
            ).strip()
        except (EOFError, OSError):
            tok = ""
        if not tok:
            sys.exit("No token provided. Set --token or the HF_TOKEN env var.")
    os.environ["HF_TOKEN"] = tok
    return tok


def _push_repo(name: str, token: str, push: str | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, create_repo
    if push:
        repo_id = push
    else:
        who = HfApi(token=token).whoami()
        repo_id = f"{who['name']}/ghana-speech-synth-{name}"
    create_repo(repo_id, repo_type="dataset", token=token, private=private, exist_ok=True)
    return repo_id


def _upload(out_dir: str, repo_id: str, token: str, msg: str = "update"):
    from huggingface_hub import HfApi
    HfApi(token=token).upload_folder(
        folder_path=out_dir,
        path_in_repo=os.path.basename(out_dir.rstrip("/")),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ghana-speech-datagen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- asr ----
    asr = sub.add_parser("asr", help="Generate synthetic speech using reference audio pool (GPU required)",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    asr_txt = asr.add_argument_group("text source (provide one)")
    asr_txt.add_argument("--dataset", help="HF dataset with text to synthesise")
    asr_txt.add_argument("--text", dest="text_column",
                         help="column with text to synthesise (with --dataset)")
    asr_txt.add_argument("--text-file", help="path to a .txt file, one sentence per line")
    asr_txt.add_argument("--config", help="dataset config (optional)")
    asr_txt.add_argument("--split", default="train")

    asr_ref = asr.add_argument_group("reference audio source (provide one)")
    asr_ref.add_argument("--ref-dataset", help="HF dataset id with reference audio+transcript columns")
    asr_ref.add_argument("--audio-column", default="audio",
                         help="column with reference audio (default: audio)")
    asr_ref.add_argument("--ref-text-column", default="text",
                         help="column with reference transcripts (default: text)")
    asr_ref.add_argument("--ref-config", help="ref dataset config (optional)")
    asr_ref.add_argument("--ref-split", default="train")
    asr_ref.add_argument("--ref-audio-dir",
                         help="local dir with reference audio files (use with --ref-metadata)")
    asr_ref.add_argument("--ref-metadata",
                         help="CSV/JSONL mapping ref audio filenames to transcripts")
    asr_ref.add_argument("--min-ref-duration", type=float, default=DEFAULT_MIN_REF_DURATION,
                         help=f"minimum ref audio duration in seconds (default {DEFAULT_MIN_REF_DURATION})")
    asr_ref.add_argument("--max-ref-duration", type=float, default=DEFAULT_MAX_REF_DURATION,
                         help=f"maximum ref audio duration in seconds (default {DEFAULT_MAX_REF_DURATION})")

    asr_val = asr.add_argument_group("generation")
    asr_val.add_argument("--hours", type=float, default=1.0, help="target hours of audio")
    asr_val.add_argument("--min-samples", type=int, default=MIN_ASR_SAMPLES,
                         help=f"minimum valid samples required (default {MIN_ASR_SAMPLES})")
    asr_val.add_argument("--min-duration", type=float, default=1.0,
                         help="drop generated clips shorter than this (seconds)")
    asr_val.add_argument("--max-duration", type=float, default=30.0,
                         help="drop generated clips longer than this (seconds)")
    asr_val.add_argument("--max-samples", type=int,
                         help="randomly pick at most this many texts")

    asr_gen = asr.add_argument_group("model")
    asr_gen.add_argument("--sample-rate", type=int, default=DEFAULT_SR,
                         help=f"output WAV rate (default {DEFAULT_SR})")
    asr_gen.add_argument("--cfg", type=float, default=2.0, dest="cfg_value",
                         help="CFG value")
    asr_gen.add_argument('--backend', choices=['cuda', 'cpu'], default='cuda',
                 help='inference backend (default: cuda)')

    asr_out = asr.add_argument_group("output")
    asr_out.add_argument("--out", help="output directory (default: data/<name>)")
    asr_out.add_argument("--name",
                         help="output name (default: dataset or audio-dir name)")
    asr_out.add_argument("--push", metavar="REPO_ID",
                         help="override auto-generated HF dataset repo")
    asr_out.add_argument("--private", action="store_true",
                         help="make the dataset repo private")
    asr_out.add_argument("--token", help="HF token (for pushing)")

    asr_misc = asr.add_argument_group("misc")
    asr_misc.add_argument("--list-datasets", action="store_true",
                          help=f"list datasets under the {DATASET_ORG} org")

    return p


# --------------------------------------------------------------------------- #
# TTS flow
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# ASR flow  (generate with reference audio pool)
# --------------------------------------------------------------------------- #

def _load_texts(dataset: str | None, text_column: str | None,
                text_file: str | None, config: str | None, split: str,
                max_samples: int | None, token: str | None) -> list[str]:
    if text_file:
        texts = [ln.strip() for ln in open(text_file, encoding="utf-8") if ln.strip()]
    elif dataset and text_column:
        from datasets import load_dataset
        ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
        if max_samples:
            ds = ds.shuffle(seed=42).take(max_samples)
        texts = [ex.get(text_column, "").strip() for ex in ds]
    else:
        sys.exit("Provide --dataset + --text, or --text-file with texts to synthesise.")
    texts = [t for t in texts if 2 <= len(t) <= 400]
    if max_samples and len(texts) > max_samples:
        texts = random.sample(texts, max_samples)
    if not texts:
        sys.exit("No valid texts found (need 2-400 chars each).")
    return texts


def _load_refs_from_dataset(dataset: str, audio_col: str, text_col: str,
                             config: str | None, split: str,
                             max_samples: int | None, token: str,
                             min_dur: float, max_dur: float) -> list:
    from datasets import load_dataset
    ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
    if max_samples:
        ds = ds.shuffle(seed=42).take(max_samples)
    refs = []
    for ex in ds:
        audio = ex.get(audio_col)
        text = ex.get(text_col)
        if audio is None or text is None:
            continue
        text = str(text).strip()
        try:
            dur = _get_audio_duration(audio)
        except Exception:
            continue
        try:
            _validate_ref_duration(dur, f"{dataset}#{ex.get('id', '?')}", min_dur, max_dur)
        except SystemExit:
            continue  # silently skip out-of-range refs
        refs.append((audio, text))
    if not refs:
        sys.exit(f"No valid reference audio+text pairs found in {dataset}.")
    return refs


def _load_refs_from_local(audio_dir: str, metadata_path: str,
                           max_samples: int | None,
                           min_dur: float, max_dur: float) -> list:
    audio_dir = Path(audio_dir)
    meta = Path(metadata_path)
    rows = []
    if meta.suffix == ".jsonl":
        with open(meta, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with open(meta, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    if max_samples and len(rows) > max_samples:
        rows = random.sample(rows, max_samples)
    refs = []
    for row in rows:
        audio_path = row.get("audio") or row.get("file") or row.get("path", "")
        text = row.get("text") or row.get("transcript") or row.get("sentence", "")
        if audio_path and text:
            full_path = str(audio_dir / audio_path)
            if not os.path.isfile(full_path):
                continue
            try:
                dur = sf.info(full_path).duration
            except Exception:
                continue
            try:
                _validate_ref_duration(dur, audio_path, min_dur, max_dur)
            except SystemExit:
                continue
            refs.append((full_path, text.strip()))
    if not refs:
        sys.exit(f"No valid reference audio+text pairs found in {audio_dir}.")
    return refs


def _cmd_asr(args):
    from . import generator

    token = _resolve_token(args) if args.push else (args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))

    texts = _load_texts(args.dataset, args.text_column, args.text_file,
                        args.config, args.split, args.max_samples, token)

    if args.ref_dataset:
        refs = _load_refs_from_dataset(
            args.ref_dataset, args.audio_column, args.ref_text_column,
            args.ref_config, args.ref_split, None, token,
            args.min_ref_duration, args.max_ref_duration,
        )
        default_name = sanitize_name(args.ref_dataset.split("/")[-1])
    elif args.ref_audio_dir and args.ref_metadata:
        refs = _load_refs_from_local(args.ref_audio_dir, args.ref_metadata, None,
                                     args.min_ref_duration, args.max_ref_duration)
        default_name = sanitize_name(os.path.basename(args.ref_audio_dir.rstrip("/")))
    else:
        sys.exit("Provide --ref-dataset or --ref-audio-dir + --ref-metadata.")

    # Pair texts with random refs
    random.shuffle(refs)
    pairs = []
    for i, t in enumerate(texts):
        ref = refs[i % len(refs)]
        pairs.append((t, ref[0], ref[1]))

    name = args.name or default_name
    out_dir = args.out or os.path.join("data", name)

    from tqdm.auto import tqdm
    target_seconds = round(args.hours * 3600)
    bar = tqdm(total=target_seconds, unit="s", unit_scale=False,
               desc="Synthesising ASR clips", file=sys.stderr)
    state = {"last": 0.0}

    def _on_clip(dur):
        delta = dur - state["last"]
        if delta > 0:
            bar.update(delta)
            state["last"] = dur

    summary = generator.generate_asr(
        out_dir=out_dir, pairs=pairs,
        min_duration=args.min_duration, max_duration=args.max_duration,
        min_samples=args.min_samples,
        target_seconds=target_seconds,
        sample_rate=args.sample_rate,
        cfg_value=args.cfg_value,
        backend=args.backend,
        on_clip=_on_clip,
        progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    print(f"\n✅ {summary['rows']} clips · {summary['hours']:.2f} h "
          f"({summary['skipped']} skipped, "
          f"{summary['duration_dropped']} dropped by duration)"
          f" → {summary['out_dir']}", file=sys.stderr)
    print("   wavs/  manifest.jsonl  metadata.jsonl", file=sys.stderr)

    if args.push is not None or args.dataset:
        push_repo = _push_repo(name, token, args.push, args.private)
        push_url = f"https://huggingface.co/datasets/{push_repo}"
        _upload(out_dir, push_repo, token, msg=f"asr data: {summary['rows']} clips / {summary['hours']:.2f}h")
        print(f"   pushed to {push_url}", file=sys.stderr)

    return 0


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_datasets:
        token = os.environ.get("HF_TOKEN") or ""
        from huggingface_hub import HfApi
        ids = sorted(d.id for d in HfApi(token=token).list_datasets(author=DATASET_ORG, limit=500))
        print("\n".join(ids) if ids else f"(no datasets found under {DATASET_ORG})")
        return 0

    elif args.command == "asr":
        return _cmd_asr(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
