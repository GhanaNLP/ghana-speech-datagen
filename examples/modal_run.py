"""Run Ghana Speech Datagen on Modal (serverless GPU).

This spins up **one GPU container** that both (a) serves the VoxCPM2-Ghana TTS
model with vLLM-Omni and (b) runs the datagen client against it on localhost —
so a single `modal run` produces a dataset end to end.

Usage:
    export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
    # Turnkey: built-in default text + in-language reference voices
    modal run examples/modal_run.py --lang ewe --hours 2
    # Or bring your own text dataset
    modal run examples/modal_run.py --dataset ghananlpcommunity/some-text \
        --text-column text --hours 2

Required secret:
    hf-token   Hugging Face token (read for the model, write to push the dataset)
"""

from __future__ import annotations

import modal

# The container needs the TTS runtime (vLLM-Omni) AND the datagen client.
# vLLM-Omni installs from source; pin vllm to a known-good version.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("vllm==0.24.0")
    .run_commands(
        "git clone https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni",
        "pip install -e /opt/vllm-omni",
    )
    .pip_install(
        "ghana-speech-datagen @ git+https://github.com/GhanaNLP/ghana-speech-datagen.git"
    )
    # Ship the tuned deploy config with the image so the server matches deploy/.
    .add_local_file("deploy/voxcpm2.yaml", remote_path="/root/voxcpm2.yaml")
)

app = modal.App("ghana-speech-datagen")

MODEL = "ghananlpcommunity/VoxCPM2-Ghana"
PORT = 8000


@app.function(
    image=image,
    gpu="A100-40GB",  # VoxCPM2 is a 2B model; A100/H100/L4-24GB all work
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
    import subprocess
    import sys
    import time
    import urllib.request

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN secret not set — create a secret named 'hf-token'")
    os.environ["HF_TOKEN"] = hf_token

    if not lang and not dataset:
        raise RuntimeError("Provide --lang (for built-in default text) or "
                           "--dataset + --text-column.")

    # 1. Launch the vLLM-Omni VoxCPM2 server in the background.
    server = subprocess.Popen([
        "vllm", "serve", MODEL,
        "--omni",
        "--deploy-config", "/root/voxcpm2.yaml",
        "--served-model-name", "voxcpm2",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--trust-remote-code",
    ])

    # 2. Wait for it to become ready (first run also downloads ~10 GB of weights).
    health = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + 1800
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"TTS server exited early (code {server.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError("TTS server did not become ready in time")

    # 3. Run the datagen client against the local server.
    argv = [
        "ghana-speech-datagen", "asr",
        "--split", split,
        "--hours", str(hours),
        "--name", name,
        "--server-url", f"http://127.0.0.1:{PORT}",
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
    try:
        rc = main()
    finally:
        server.terminate()
    raise SystemExit(rc)
