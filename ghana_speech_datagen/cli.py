"""Command-line interface for Ghana Speech Datagen.

Subcommands (both need a GPU for usable speed):
  tts   Synthesise a TTS dataset from a small speaker set (LJSpeech output)
  asr   Synthesise an ASR dataset from a large reference-audio pool (ASR output)
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


def _stored_token() -> str | None:
    """Token saved by ``huggingface-cli login`` (HfFolder/get_token)."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _resolve_token(args) -> str:
    tok = (args.token or os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN") or _stored_token())
    if not tok:
        try:
            import getpass
            tok = getpass.getpass(
                "HF Token (required -- needed to push to your HF account): "
            ).strip()
        except (EOFError, OSError):
            tok = ""
        if not tok:
            sys.exit("No token provided. Set --token or the HF_TOKEN env var, "
                     "log in with `huggingface-cli login`, "
                     "or pass --no-push to generate locally without uploading.")
    os.environ["HF_TOKEN"] = tok
    return tok


def _read_token(args) -> str | None:
    """Best-effort token for *reading* datasets (never prompts)."""
    return (args.token or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN") or _stored_token())


def _build_pusher(name: str, token: str, args):
    """Create the target repo and return ``(on_save, push_url)``.

    ``on_save(out_dir)`` uploads the run folder to HF — called repeatedly for
    incremental pushes as data is generated.
    """
    push_repo = _push_repo(name, token, args.push, args.private)
    push_url = f"https://huggingface.co/datasets/{push_repo}"

    def on_save(out_dir: str) -> None:
        _upload(out_dir, push_repo, token, msg="synth data (incremental)")

    return on_save, push_url


def _push_repo(name: str, token: str, push: str | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, create_repo
    if push:
        repo_id = push
    else:
        who = HfApi(token=token).whoami()
        repo_id = f"{who['name']}/ghana-speech-synth-{name}"
    create_repo(repo_id, repo_type="dataset", token=token, private=private, exist_ok=True)
    return repo_id


# Internal working dirs kept under the run folder but never published.
_UPLOAD_IGNORE = [".voxcpm-voices/**", "_normalized/**",
                  "**/.voxcpm-voices/**", "**/_normalized/**"]


def _upload(out_dir: str, repo_id: str, token: str, msg: str = "update"):
    from huggingface_hub import HfApi
    HfApi(token=token).upload_folder(
        folder_path=out_dir,
        path_in_repo=os.path.basename(out_dir.rstrip("/")),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
        ignore_patterns=_UPLOAD_IGNORE,
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

    # ---- tts ----
    tts = sub.add_parser(
        "tts",
        help="Synthesise a TTS dataset from a small speaker set (LJSpeech output)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    tts_txt = tts.add_argument_group("text source (provide one; default: --lang)")
    tts_txt.add_argument("--lang", help="language code — use built-in default text "
                         "for that language (run --list-langs to see codes)")
    tts_txt.add_argument("--dataset", help="HF dataset with text to synthesise")
    tts_txt.add_argument("--text", dest="text_column",
                         help="column with text to synthesise (with --dataset)")
    tts_txt.add_argument("--text-file", help="path to a .txt file, one sentence per line")
    tts_txt.add_argument("--config", help="dataset config (optional)")
    tts_txt.add_argument("--split", default="train")

    tts_spk = tts.add_argument_group("speaker voices (default: packaged male + female)")
    tts_spk.add_argument("--voices", choices=["both", "male", "female"], default="both",
                         help="which packaged voices to use (default: both)")
    tts_spk.add_argument("--speaker-dir",
                         help="dir of custom speakers: each NAME.wav with a NAME.txt "
                              "prompt transcript beside it")
    tts_spk.add_argument("--speaker", metavar="WAV",
                         help="a single custom speaker WAV (use with --speaker-text)")
    tts_spk.add_argument("--speaker-text",
                         help="prompt transcript for --speaker")
    tts_spk.add_argument("--min-ref-duration", type=float, default=DEFAULT_MIN_REF_DURATION,
                         help=f"minimum speaker audio duration (default {DEFAULT_MIN_REF_DURATION})")
    tts_spk.add_argument("--max-ref-duration", type=float, default=DEFAULT_MAX_REF_DURATION,
                         help=f"maximum speaker audio duration (default {DEFAULT_MAX_REF_DURATION})")

    tts_val = tts.add_argument_group("generation")
    tts_val.add_argument("--hours", type=float, default=1.0, help="target hours of audio")
    tts_val.add_argument("--min-samples", type=int, default=1,
                         help="minimum valid samples required (default 1)")
    tts_val.add_argument("--min-duration", type=float, default=1.0,
                         help="drop generated clips shorter than this (seconds)")
    tts_val.add_argument("--max-duration", type=float, default=30.0,
                         help="drop generated clips longer than this (seconds)")
    tts_val.add_argument("--max-samples", type=int,
                         help="randomly pick at most this many texts")

    tts_gen = tts.add_argument_group("model")
    tts_gen.add_argument("--sample-rate", type=int, default=DEFAULT_SR,
                         help=f"output WAV rate (default {DEFAULT_SR}, the TTS standard)")
    tts_gen.add_argument("--cfg", type=float, default=2.0, dest="cfg_value",
                         help="CFG value")
    tts_gen.add_argument("--backend", choices=["cuda", "cpu"], default="cuda",
                         help="inference backend (default: cuda)")

    tts_out = tts.add_argument_group("output")
    tts_out.add_argument("--out", help="output directory (default: data/<name>)")
    tts_out.add_argument("--name", help="output name (default: language or text source)")
    tts_out.add_argument("--push", metavar="REPO_ID",
                         help="override auto-generated HF dataset repo")
    tts_out.add_argument("--no-push", action="store_true",
                         help="generate locally only; do NOT upload to HF")
    tts_out.add_argument("--save-every", type=int, default=200,
                         help="flush + push every N clips as they are generated (default 200)")
    tts_out.add_argument("--private", action="store_true",
                         help="make the dataset repo private")
    tts_out.add_argument("--token", help="HF token (for pushing)")

    tts_misc = tts.add_argument_group("misc")
    tts_misc.add_argument("--list-datasets", action="store_true",
                          help=f"list datasets under the {DATASET_ORG} org")
    tts_misc.add_argument("--list-langs", action="store_true",
                          help="list supported languages and their default text sources")

    # ---- asr ----
    asr = sub.add_parser("asr", help="Generate synthetic speech using reference audio pool (GPU required)",
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    asr_txt = asr.add_argument_group("text source (provide one; default: --lang)")
    asr_txt.add_argument("--lang", help="language code — use built-in default text "
                         "for that language (run --list-langs to see codes)")
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
    asr_ref.add_argument("--max-ref-samples", type=int,
                         help="randomly pick at most this many reference clips (default: all)")

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
    asr_out.add_argument("--no-push", action="store_true",
                         help="generate locally only; do NOT upload to HF")
    asr_out.add_argument("--save-every", type=int, default=200,
                         help="flush + push every N clips as they are generated (default 200)")
    asr_out.add_argument("--private", action="store_true",
                         help="make the dataset repo private")
    asr_out.add_argument("--token", help="HF token (for pushing)")

    asr_misc = asr.add_argument_group("misc")
    asr_misc.add_argument("--list-datasets", action="store_true",
                          help=f"list datasets under the {DATASET_ORG} org")
    asr_misc.add_argument("--list-langs", action="store_true",
                          help="list supported languages and their default text sources")

    return p


# --------------------------------------------------------------------------- #
# Shared text loading (used by both tts and asr)
# --------------------------------------------------------------------------- #

def _load_texts(dataset: str | None, text_column: str | None,
                text_file: str | None, config: str | None, split: str,
                max_samples: int | None, token: str | None,
                lang: str | None = None) -> list[str]:
    if text_file:
        texts = [ln.strip() for ln in open(text_file, encoding="utf-8") if ln.strip()]
    elif dataset and text_column:
        from datasets import load_dataset
        ds = load_dataset(dataset, config or None, split=split, streaming=True, token=token)
        if max_samples:
            ds = ds.shuffle(seed=42).take(max_samples)
        texts = [ex.get(text_column, "").strip() for ex in ds]
    elif lang:
        from . import text_sources
        texts = text_sources.load_texts(lang, max_samples, token)
    else:
        sys.exit("Provide --lang for built-in default text, --dataset + --text, "
                 "or --text-file with texts to synthesise.")
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


# --------------------------------------------------------------------------- #
# TTS flow  (small speaker set -> LJSpeech output)
# --------------------------------------------------------------------------- #

def _build_tts_speakers(args) -> list:
    """Return ``[(label, wav_path, prompt_text), ...]`` for the TTS voice set.

    Defaults to the packaged male/female voices; overridable with a custom
    ``--speaker-dir`` (NAME.wav + NAME.txt pairs) or a single ``--speaker``.
    """
    from . import generator

    speakers: list = []
    if args.speaker_dir:
        d = Path(args.speaker_dir)
        for wav in sorted(d.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                print(f"⚠️  skipping {wav.name}: no matching {txt.name}", file=sys.stderr)
                continue
            speakers.append((wav.stem, str(wav), txt.read_text(encoding="utf-8").strip()))
        if not speakers:
            sys.exit(f"No NAME.wav + NAME.txt speaker pairs found in {args.speaker_dir}.")
    elif args.speaker:
        if not args.speaker_text:
            sys.exit("--speaker requires --speaker-text (the prompt transcript).")
        label = sanitize_name(os.path.splitext(os.path.basename(args.speaker))[0])
        speakers.append((label, args.speaker, args.speaker_text.strip()))
    else:
        want = ("male", "female") if args.voices == "both" else (args.voices,)
        for g in want:
            spk = generator.SPEAKERS.get(g)
            if spk:
                speakers.append((g, spk["wav"], spk["text"]))
        if not speakers:
            sys.exit("No packaged speaker voices available; use --speaker-dir or --speaker.")

    validated = []
    for label, wav, text in speakers:
        if not os.path.isfile(wav):
            sys.exit(f"Speaker audio not found: {wav}")
        try:
            dur = sf.info(wav).duration
        except Exception as e:
            sys.exit(f"Cannot read speaker audio {wav}: {e}")
        _validate_ref_duration(dur, label, args.min_ref_duration, args.max_ref_duration)
        validated.append((label, wav, text))
    return validated


def _cmd_tts(args):
    from . import generator, text_sources

    lang = None
    if args.lang:
        try:
            lang = text_sources.resolve_lang(args.lang)
        except ValueError as e:
            sys.exit(str(e))

    push_enabled = not args.no_push
    token = _read_token(args)

    texts = _load_texts(args.dataset, args.text_column, args.text_file,
                        args.config, args.split, args.max_samples, token, lang)

    speakers = _build_tts_speakers(args)
    print(f"Using {len(speakers)} speaker voice(s): "
          f"{', '.join(s[0] for s in speakers)}", file=sys.stderr)

    if lang:
        default_name = sanitize_name(lang)
    elif args.dataset:
        default_name = sanitize_name(args.dataset.split("/")[-1])
    elif args.text_file:
        default_name = sanitize_name(os.path.splitext(os.path.basename(args.text_file))[0])
    else:
        default_name = "tts"
    name = args.name or default_name
    out_dir = args.out or os.path.join("data", name)

    on_save, push_url = (None, None)
    if push_enabled:
        token = _resolve_token(args)  # prompt-if-missing, only now that args are valid
        on_save, push_url = _build_pusher(name, token, args)
        print(f"Auto-pushing to {push_url} as data is generated "
              f"(use --no-push to disable).", file=sys.stderr)

    # Round-robin each text across the small speaker set (balanced coverage).
    pairs, speaker_labels = [], []
    for i, t in enumerate(texts):
        label, wav, ref_text = speakers[i % len(speakers)]
        pairs.append((t, wav, ref_text))
        speaker_labels.append(label)

    from tqdm.auto import tqdm
    target_seconds = round(args.hours * 3600)
    bar = tqdm(total=target_seconds, unit="s", unit_scale=False,
               desc="Synthesising TTS clips", file=sys.stderr)
    state = {"last": 0.0}

    def _on_clip(dur):
        delta = dur - state["last"]
        if delta > 0:
            bar.update(delta)
            state["last"] = dur

    summary = generator.generate_tts(
        out_dir=out_dir, pairs=pairs, speaker_labels=speaker_labels,
        min_duration=args.min_duration, max_duration=args.max_duration,
        min_samples=args.min_samples,
        target_seconds=target_seconds,
        sample_rate=args.sample_rate,
        cfg_value=args.cfg_value,
        backend=args.backend,
        lang=lang,
        on_clip=_on_clip,
        on_save=on_save,
        save_every=args.save_every,
        progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    written = " ".join(os.path.basename(w) for w in summary.get("written", []))
    print(f"\n✅ {summary['rows']} clips · {summary['hours']:.2f} h "
          f"({summary['skipped']} skipped, "
          f"{summary['duration_dropped']} dropped by duration)"
          f" → {summary['out_dir']}", file=sys.stderr)
    print(f"   wavs/  manifest.jsonl  {written}", file=sys.stderr)
    if push_url:
        print(f"   pushed to {push_url}", file=sys.stderr)

    return 0


# --------------------------------------------------------------------------- #
# ASR flow  (generate with reference audio pool)
# --------------------------------------------------------------------------- #

def _cmd_asr(args):
    from . import generator, text_sources

    lang = None
    if args.lang:
        try:
            lang = text_sources.resolve_lang(args.lang)
        except ValueError as e:
            sys.exit(str(e))

    push_enabled = not args.no_push
    token = _read_token(args)

    texts = _load_texts(args.dataset, args.text_column, args.text_file,
                        args.config, args.split, args.max_samples, token, lang)

    if args.ref_dataset:
        refs = _load_refs_from_dataset(
            args.ref_dataset, args.audio_column, args.ref_text_column,
            args.ref_config, args.ref_split, args.max_ref_samples, token,
            args.min_ref_duration, args.max_ref_duration,
        )
        default_name = sanitize_name(args.ref_dataset.split("/")[-1])
    elif args.ref_audio_dir and args.ref_metadata:
        refs = _load_refs_from_local(args.ref_audio_dir, args.ref_metadata, None,
                                     args.min_ref_duration, args.max_ref_duration)
        default_name = sanitize_name(os.path.basename(args.ref_audio_dir.rstrip("/")))
    elif lang and text_sources.LANGUAGES[lang].gs_config:
        # Default reference pool: in-language audio from ghana-speech.
        gs_config = text_sources.LANGUAGES[lang].gs_config
        print(f"No reference audio given — using {text_sources.GHANA_SPEECH} "
              f"({gs_config}) as the {lang} reference voice pool.", file=sys.stderr)
        refs = _load_refs_from_dataset(
            text_sources.GHANA_SPEECH, "audio", "text",
            gs_config, "train", args.max_ref_samples or 200, token,
            args.min_ref_duration, args.max_ref_duration,
        )
        default_name = sanitize_name(lang)
    elif lang:
        # No in-language audio (e.g. English) — fall back to packaged speakers.
        print("No reference audio given — using the packaged reference voices.",
              file=sys.stderr)
        refs = generator.builtin_speaker_refs()
        if not refs:
            sys.exit("No packaged reference voices available; provide --ref-dataset "
                     "or --ref-audio-dir + --ref-metadata.")
        default_name = sanitize_name(lang)
    else:
        sys.exit("Provide --ref-dataset, --ref-audio-dir + --ref-metadata, or "
                 "--lang (to use a default in-language reference pool).")

    # Pair texts with random refs
    random.shuffle(refs)
    pairs = []
    for i, t in enumerate(texts):
        ref = refs[i % len(refs)]
        pairs.append((t, ref[0], ref[1]))

    name = args.name or default_name
    out_dir = args.out or os.path.join("data", name)

    on_save, push_url = (None, None)
    if push_enabled:
        token = _resolve_token(args)  # prompt-if-missing, only now that args are valid
        on_save, push_url = _build_pusher(name, token, args)
        print(f"Auto-pushing to {push_url} as data is generated "
              f"(use --no-push to disable).", file=sys.stderr)

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
        lang=lang,
        on_clip=_on_clip,
        on_save=on_save,
        save_every=args.save_every,
        progress=lambda m: bar.set_description(m[:48]),
    )
    bar.close()

    written = " ".join(os.path.basename(w) for w in summary.get("written", []))
    print(f"\n✅ {summary['rows']} clips · {summary['hours']:.2f} h "
          f"({summary['skipped']} skipped, "
          f"{summary['duration_dropped']} dropped by duration)"
          f" → {summary['out_dir']}", file=sys.stderr)
    print(f"   wavs/  manifest.jsonl  {written}", file=sys.stderr)
    if push_url:
        print(f"   pushed to {push_url}", file=sys.stderr)

    return 0


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "list_langs", False):
        from . import text_sources
        print(text_sources.format_language_table())
        return 0

    if args.list_datasets:
        token = os.environ.get("HF_TOKEN") or ""
        from huggingface_hub import HfApi
        ids = sorted(d.id for d in HfApi(token=token).list_datasets(author=DATASET_ORG, limit=500))
        print("\n".join(ids) if ids else f"(no datasets found under {DATASET_ORG})")
        return 0

    elif args.command == "tts":
        return _cmd_tts(args)
    elif args.command == "asr":
        return _cmd_asr(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
