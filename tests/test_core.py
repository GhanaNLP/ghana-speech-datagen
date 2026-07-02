"""Tests for the GPU-free helpers and CLI parsing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghana_tts_datagen import clean_text, pick_gender, sanitize_name, trim_silences, SPEAKERS
from ghana_tts_datagen import cli


def test_clean_text():
    assert clean_text("  hello\n world  \t x ") == "hello world x"
    assert clean_text("a\n\nb") == "a b"


def test_pick_gender_modes():
    assert pick_gender(0, "male", 50) == "male"
    assert pick_gender(7, "all male", 50) == "male"
    assert pick_gender(0, "female", 50) == "female"
    assert pick_gender(3, "all female", 50) == "female"


def test_pick_gender_custom_deterministic_and_split():
    assert pick_gender(42, "custom", 50) == pick_gender(42, "custom", 50)
    assert all(pick_gender(i, "custom", 100) == "male" for i in range(50))
    assert all(pick_gender(i, "custom", 0) == "female" for i in range(50))
    males = sum(pick_gender(i, "custom", 50) == "male" for i in range(1000))
    assert 400 < males < 600


def test_sanitize_name():
    assert sanitize_name("My Run #1!") == "My-Run-1"
    assert sanitize_name("   ") == "run"
    assert sanitize_name("twi_run-2") == "twi_run-2"


def test_trim_silences_keeps_audio_and_shortens_gaps():
    sr = 16000
    tone = np.sin(np.linspace(0, 50, sr)).astype("float32")
    gap = np.zeros(sr * 2, dtype="float32")
    wav = np.concatenate([tone, gap, tone])
    out = trim_silences(wav, sr=sr)
    assert out.size > 0
    assert out.size < wav.size
    assert out.size >= 2 * sr


def test_speakers_loaded():
    for g in ("male", "female"):
        assert SPEAKERS[g]["text"]
        assert Path(SPEAKERS[g]["wav"]).exists()


def test_precision_and_instances():
    from ghana_tts_datagen import auto_instances
    a = cli.build_parser().parse_args(
        ["--text-file", "s.txt", "--precision", "bf16", "--sample-rate", "24000"])
    assert a.precision == "bf16" and a.sample_rate == 24000
    assert cli.build_parser().parse_args([]).precision == "fp32"
    assert auto_instances("fp32") >= 1 and auto_instances("fp16") >= 1


def test_export_formats(tmp_path):
    import json
    from ghana_tts_datagen import export_formats

    run = tmp_path / "run"
    (run / "wavs").mkdir(parents=True)
    rows = [
        {"id": "0000000_ab", "file": "wavs/0000000_ab.wav", "text": "hello there",
         "gender": "male", "speaker": "male", "duration": 1.2},
        {"id": "0000001_ab", "file": "wavs/0000001_ab.wav", "text": "good morning",
         "gender": "female", "speaker": "female", "duration": 1.0},
    ]
    (run / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    export_formats(str(run), ["ljspeech", "piper", "vits", "melo"], lang="twi")

    assert (run / "metadata.csv").read_text().splitlines()[0].split("|") == \
        ["0000000_ab", "hello there", "hello there"]
    assert (run / "metadata.piper.csv").read_text().splitlines()[0].split("|") == \
        ["0000000_ab", "male", "hello there"]
    assert (run / "filelist.txt").read_text().splitlines()[0].startswith(
        "wavs/0000000_ab.wav|0|")
    assert (run / "speakers.txt").exists()
    ml = (run / "metadata.list").read_text().splitlines()[0].split("|")
    assert ml[0] == "wavs/0000000_ab.wav" and ml[2] == "TWI"


def test_cli_parser_and_requirements():
    a = cli.build_parser().parse_args(
        ["--dataset", "org/ds", "--text-column", "text", "--hours", "5",
         "--voices", "custom", "--male-pct", "60", "--name", "run1",
         "--formats", "piper,vits"])
    assert a.dataset == "org/ds" and a.text_column == "text"
    assert a.hours == 5 and a.voices == "custom" and a.male_pct == 60
    assert a.formats == "piper,vits"

    assert cli.build_parser().parse_args(["--text-file", "s.txt"]).text_file == "s.txt"

    try:
        cli.main(["--split", "train"])
    except SystemExit as e:
        assert "text-file" in str(e) or "dataset" in str(e)
    else:
        raise AssertionError("expected SystemExit without a source")
