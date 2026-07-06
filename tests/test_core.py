"""Offline tests for the GPU-free helpers, the language registry, and CLI parsing.

These never touch the network or the VoxCPM server.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghana_speech_datagen import (
    clean_text,
    sanitize_name,
    export_formats,
    builtin_speaker_refs,
    generate,
    generate_asr,
    generate_tts,
    SPEAKERS,
    LANGUAGES,
    lang_tag,
    resolve_lang,
    sources_for,
)
from ghana_speech_datagen import cli, text_sources


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_clean_text():
    assert clean_text("  hello\n world  \t x ") == "hello world x"
    assert clean_text("a\n\nb") == "a b"


def test_sanitize_name():
    assert sanitize_name("My Run #1!") == "My-Run-1"
    assert sanitize_name("   ") == "run"
    assert sanitize_name("twi_run-2") == "twi_run-2"


def test_speakers_loaded():
    for g in ("male", "female"):
        assert SPEAKERS[g]["text"]
        assert Path(SPEAKERS[g]["wav"]).exists()


def test_builtin_speaker_refs():
    refs = builtin_speaker_refs()
    assert len(refs) == len(SPEAKERS)
    for wav, text in refs:
        assert Path(wav).exists()
        assert text


# --------------------------------------------------------------------------- #
# Language registry
# --------------------------------------------------------------------------- #
def test_lang_tag_format():
    # Must match the training-time tag exactly: "<|lang:CODE|> "
    assert lang_tag("ewe") == "<|lang:ewe|> "
    assert lang_tag("twi-asante") == "<|lang:twi-asante|> "


def test_resolve_lang_variants():
    assert resolve_lang("ewe") == "ewe"
    assert resolve_lang("Ewe") == "ewe"
    assert resolve_lang("Ewe_ewe") == "ewe"          # config name
    assert resolve_lang("Asante Twi") == "twi-asante"  # display name
    assert resolve_lang("twi") == "twi-asante"        # alias
    assert resolve_lang("english") == "en"


def test_resolve_lang_unknown():
    try:
        resolve_lang("klingon")
    except ValueError as e:
        assert "Unknown language" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown language")


def test_every_language_has_a_source():
    for code in LANGUAGES:
        assert sources_for(code), f"{code} has no text source"


def test_twi_has_extra_source():
    datasets = {s.dataset for s in sources_for("twi-asante")}
    assert text_sources.GHANA_SPEECH in datasets
    assert "ghananlpcommunity/twi-health-asr-gemini-500hrs" in datasets


def test_ghana_speech_configs_map_to_codes():
    # Twi is split by dialect; everything else uses the trailing ISO code.
    assert LANGUAGES["twi-akuapem"].gs_config == "Akuapem_Twi_twi"
    assert LANGUAGES["twi-asante"].gs_config == "Asante_Twi_twi"
    assert LANGUAGES["ewe"].gs_config == "Ewe_ewe"
    assert LANGUAGES["fat"].gs_config == "Fante_fat"
    assert LANGUAGES["en"].gs_config is None  # English audio lives elsewhere


def test_format_language_table():
    table = text_sources.format_language_table()
    assert "ewe" in table and "twi-asante" in table and "English" in table


# --------------------------------------------------------------------------- #
# Generation wrappers
# --------------------------------------------------------------------------- #
def test_generate_wrappers_exist():
    import inspect
    # Both wrappers must exist and route through the shared core.
    assert callable(generate) and callable(generate_asr) and callable(generate_tts)
    sig = inspect.signature(generate)
    for kw in ("pairs", "output_formats", "speaker_labels", "lang"):
        assert kw in sig.parameters


# --------------------------------------------------------------------------- #
# CLI parsing — both subcommands
# --------------------------------------------------------------------------- #
def test_cli_has_both_subcommands():
    # tts and asr must both parse.
    tts = cli.build_parser().parse_args(["tts", "--lang", "ewe"])
    asr = cli.build_parser().parse_args(["asr", "--lang", "ewe"])
    assert tts.command == "tts" and asr.command == "asr"


def test_cli_tts_speaker_args():
    a = cli.build_parser().parse_args(
        ["tts", "--lang", "ewe", "--voices", "male", "--sample-rate", "22050"])
    assert a.voices == "male" and a.sample_rate == 22050


def test_build_tts_speakers_default_packaged():
    a = cli.build_parser().parse_args(["tts", "--lang", "ewe", "--voices", "both"])
    spk = cli._build_tts_speakers(a)
    assert {s[0] for s in spk} == {"male", "female"}
    for label, wav, text in spk:
        assert Path(wav).exists() and text


def test_build_tts_speakers_single_voice():
    a = cli.build_parser().parse_args(["tts", "--lang", "ewe", "--voices", "female"])
    spk = cli._build_tts_speakers(a)
    assert [s[0] for s in spk] == ["female"]


def test_build_tts_speakers_custom_dir(tmp_path):
    import numpy as np
    import soundfile as sf
    for nm in ("alice", "bob"):
        sf.write(str(tmp_path / f"{nm}.wav"),
                 (np.sin(np.linspace(0, 100, 16000 * 3)) * 0.3).astype("float32"), 16000)
        (tmp_path / f"{nm}.txt").write_text(f"{nm} prompt", encoding="utf-8")
    a = cli.build_parser().parse_args(["tts", "--lang", "ewe", "--speaker-dir", str(tmp_path)])
    spk = cli._build_tts_speakers(a)
    assert {s[0] for s in spk} == {"alice", "bob"}


def test_push_defaults_on_for_both():
    # Auto-push is the default; --save-every controls incremental cadence.
    for cmd in ("tts", "asr"):
        a = cli.build_parser().parse_args([cmd, "--lang", "ewe"])
        assert a.no_push is False
        assert a.save_every == 200
        b = cli.build_parser().parse_args([cmd, "--lang", "ewe", "--no-push",
                                           "--save-every", "50"])
        assert b.no_push is True and b.save_every == 50


def test_generate_accepts_incremental_save_params():
    import inspect
    sig = inspect.signature(generate)
    assert sig.parameters["save_every"].default == 0
    assert "on_save" in sig.parameters


def test_cli_parses_lang():
    a = cli.build_parser().parse_args(["asr", "--lang", "ewe", "--hours", "5"])
    assert a.lang == "ewe" and a.hours == 5.0


def test_cli_parses_list_langs():
    a = cli.build_parser().parse_args(["asr", "--list-langs"])
    assert a.list_langs is True


def test_cli_parses_byo_sources():
    a = cli.build_parser().parse_args(
        ["asr", "--dataset", "org/ds", "--text", "text",
         "--ref-dataset", "org/refs", "--hours", "2", "--max-samples", "500"])
    assert a.dataset == "org/ds" and a.text_column == "text"
    assert a.ref_dataset == "org/refs" and a.max_samples == 500


def test_cmd_asr_rejects_no_text_source():
    # No --lang, --dataset, or --text-file -> should exit with guidance.
    try:
        cli.main(["asr"])
    except SystemExit as e:
        assert "lang" in str(e) or "text-file" in str(e) or "dataset" in str(e)
    else:
        raise AssertionError("expected SystemExit without a text source")


def test_cmd_asr_rejects_bad_lang():
    try:
        cli.main(["asr", "--lang", "klingon"])
    except SystemExit as e:
        assert "Unknown language" in str(e)
    else:
        raise AssertionError("expected SystemExit for unknown language")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_export_formats(tmp_path):
    run = tmp_path / "run"
    (run / "wavs").mkdir(parents=True)
    rows = [
        {"id": "0000000_ab", "file": "wavs/0000000_ab.wav", "text": "hello there",
         "duration": 1.2},
        {"id": "0000001_ab", "file": "wavs/0000001_ab.wav", "text": "good morning",
         "duration": 1.0},
    ]
    (run / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    export_formats(str(run), ["ljspeech", "asr"])

    assert (run / "metadata.csv").read_text().splitlines()[0].split("|") == \
        ["0000000_ab", "hello there", "hello there"]

    asr = [json.loads(l) for l in (run / "metadata.jsonl").read_text().splitlines() if l.strip()]
    assert asr[0] == {"audio": "wavs/0000000_ab.wav", "text": "hello there"}
    assert asr[1] == {"audio": "wavs/0000001_ab.wav", "text": "good morning"}
