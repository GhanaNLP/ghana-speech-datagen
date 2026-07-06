"""Run Ghana Speech Datagen on Modal (serverless GPU).

Usage:
    export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
    # Turnkey: built-in default text + in-language reference voices
    modal run examples/modal_run.py --lang ewe --hours 2
    # Or bring your own text dataset
    modal run examples/modal_run.py --dataset ghananlpcommunity/some-text --text-column text --hours 2

Required secrets:
    hf-token   Hugging Face token (read access to the private model)
"""

from __future__ import annotations

import modal

app = modal.App("ghana-speech-datagen")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "ghana-speech-datagen @ git+https://github.com/GhanaNLP/ghana-speech-datagen.git",
        "torch",
        "torchaudio",
    )
)


@app.function(
    image=image,
    gpu="any",
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
)
def run(
    lang: str | None = None,
    dataset: str | None = None,
    text_column: str | None = None,
    config: str | None = None,
    split: str = "train",
    hours: float = 1.0,
    name: str = "modal-run",
    max_samples: int | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    push_repo: str | None = None,
    private: bool = False,
):
    import os
    import sys

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN secret not set — create a secret named 'hf-token'")

    os.environ["HF_TOKEN"] = hf_token

    if not lang and not dataset:
        raise RuntimeError("Provide --lang (for built-in default text) or "
                           "--dataset + --text-column.")

    argv = [
        "ghana-speech-datagen", "asr",
        "--split", split,
        "--hours", str(hours),
        "--name", name,
    ]
    if lang:
        argv += ["--lang", lang]
    if dataset:
        argv += ["--dataset", dataset]
        if text_column:
            argv += ["--text", text_column]
    if config:
        argv += ["--config", config]
    if max_samples is not None:
        argv += ["--max-samples", str(max_samples)]
    if min_duration is not None:
        argv += ["--min-duration", str(min_duration)]
    if max_duration is not None:
        argv += ["--max-duration", str(max_duration)]
    if push_repo:
        argv += ["--push", push_repo]
    if private:
        argv += ["--private"]

    sys.argv = argv
    from ghana_speech_datagen.cli import main
    raise SystemExit(main())
