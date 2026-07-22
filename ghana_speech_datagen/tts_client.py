# -*- coding: utf-8 -*-
"""Client for a vLLM-Omni VoxCPM2 TTS server (OpenAI-compatible speech API).

The TTS model runs as a standalone vLLM-Omni server on a GPU (see
``deploy/``); this client just talks to it over HTTP. Voice cloning is done
inline per request via ``ref_audio`` (a base64 ``data:`` URI of the reference
WAV) plus ``ref_text`` — the server caches resolved references by hash, so
re-using a small reference pool across many texts is cheap.

The public surface mirrors the old VoxCPM.cpp wrapper
(``wait_until_ready`` / ``register_voice`` / ``synthesize``) so the generation
loop is backend-agnostic, but there is no subprocess to manage: point it at a
running server with ``base_url`` (+ ``api_key`` if the server requires one).
"""

from __future__ import annotations

import base64
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("TTS_MODEL_NAME", "voxcpm2")
DEFAULT_API_KEY = os.environ.get("TTS_API_KEY")


class TTSError(Exception):
    """Base exception for TTS client errors."""


class ServerTimeoutError(TTSError):
    """Server did not become ready within the allowed time."""


class APIError(TTSError):
    """The server returned a non-2xx response."""


class VoxCPM2Client:
    """Talks to a running vLLM-Omni VoxCPM2 TTS server.

    Usage::

        client = VoxCPM2Client(base_url="http://gpu-host:8000", api_key="…")
        client.wait_until_ready()
        client.register_voice("spk1", "ref.wav", "reference transcript")
        wav_bytes = client.synthesize("spk1", "Akwaaba")
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        request_timeout: float = 300.0,
    ):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else DEFAULT_API_KEY
        self.model = model
        self.request_timeout = request_timeout

        self._session = requests.Session()
        if self.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"

        # voice_id -> (ref_audio_data_url, ref_text)
        self._voices: dict[str, tuple[str, str]] = {}

    # ── health check ─────────────────────────────────────────────────────────

    def wait_until_ready(self, timeout: float = 120.0) -> None:
        """Poll ``GET /health`` until the server returns 200.

        Raises:
            ServerTimeoutError: if not ready within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        delay = 0.5
        last_exc: Exception | None = None

        while time.monotonic() < deadline:
            try:
                resp = self._session.get(f"{self.base_url}/health", timeout=10.0)
                if resp.status_code == 200:
                    log.info("TTS server ready at %s", self.base_url)
                    return
            except requests.RequestException as e:
                last_exc = e
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)

        raise ServerTimeoutError(
            f"TTS server at {self.base_url} did not become ready within "
            f"{timeout}s. Last error: {last_exc}. Is the vLLM-Omni server "
            f"running? (see deploy/README.md)"
        )

    # ── voice registration ───────────────────────────────────────────────────

    def register_voice(self, voice_id: str, wav_path: str, text: str) -> None:
        """Register a reference voice from a WAV file and its transcript.

        This is a client-side operation: the WAV is base64-encoded once and
        reused as the ``ref_audio`` for every :meth:`synthesize` call with this
        ``voice_id``. No server round-trip is needed.

        Args:
            voice_id: Unique identifier for the voice (e.g. ``"ref_0"``).
            wav_path: Path to a mono WAV file with the reference audio.
            text: Transcript of the reference audio.
        """
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Reference WAV not found: {wav_path}")
        with open(wav_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("ascii")
        data_url = f"data:audio/wav;base64,{audio_b64}"
        self._voices[voice_id] = (data_url, text or "")

    # ── synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        voice_id: str,
        text: str,
        response_format: str = "wav",
    ) -> bytes:
        """Synthesise speech for *text* in the voice registered as *voice_id*.

        Returns raw audio bytes (WAV by default). The container carries its own
        sample rate — read it from the WAV header rather than assuming one.
        """
        try:
            ref_audio, ref_text = self._voices[voice_id]
        except KeyError:
            raise TTSError(
                f"Voice '{voice_id}' is not registered; call register_voice first."
            )

        payload = {
            "model": self.model,
            "input": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "response_format": response_format,
        }
        resp = self._session.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=self.request_timeout,
        )
        if not resp.ok:
            raise APIError(f"Synthesis failed ({resp.status_code}): {resp.text[:500]}")
        return resp.content

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "VoxCPM2Client":
        return self

    def __exit__(self, *args) -> None:
        self.close()
