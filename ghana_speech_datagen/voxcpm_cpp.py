# -*- coding: utf-8 -*-
"""Wrapper for the VoxCPM.cpp HTTP TTS server (voxcpm-server).

Manages the server as a subprocess and exposes an OpenAI-compatible TTS API:
health check, voice registration, speech synthesis, and voice deletion.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from typing import IO

import requests

log = logging.getLogger(__name__)

SERVER_BINARY = os.environ.get(
    "VOXCPM_SERVER_BIN",
    "/mnt/volume_d2wey28/projects/voxcpm-cpp/build-cuda/examples/voxcpm-server",
)
DEFAULT_MODEL_PATH = os.environ.get(
    "VOXCPM_MODEL_PATH",
    "/mnt/volume_d2wey28/projects/voxcpm-cpp/models/ghana-tts-36k-q8_0-audiovae-f16.gguf",
)


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Bind to port 0 on *host*, return the assigned port, then release."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]


class VoxCPMError(Exception):
    """Base exception for VoxCPM server errors."""


class ServerStartupError(VoxCPMError):
    """Server binary failed to start or exited prematurely."""


class ServerTimeoutError(VoxCPMError):
    """Server did not become ready within the allowed time."""


class APIError(VoxCPMError):
    """The server returned a non-2xx response."""


class VoxCPMCppServer:
    """Manages a local voxcpm-server subprocess for TTS inference.

    Usage::

        with VoxCPMCppServer(model_path="/path/to/model.gguf") as server:
            server.wait_until_ready()
            server.register_voice("spk1", "ref.wav", "reference text")
            wav = server.synthesize("spk1", "Hello world")
    """

    def __init__(
        self,
        model_path: str | None = None,
        port: int = 0,
        host: str = "127.0.0.1",
        voice_dir: str | None = None,
        model_name: str = "ghana-tts-36k",
        threads: int = 8,
        backend: str = "cuda",
        output_sample_rate: int = 16000,
        server_binary: str | None = None,
        max_decode_steps: int = 1024,
        max_queue: int = 64,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.host = host
        self.port = port if port else _find_free_port(host)
        self.voice_dir = voice_dir or os.path.join(os.getcwd(), ".voxcpm-voices")
        self.model_name = model_name
        self.threads = threads
        self.backend = backend
        self.output_sample_rate = output_sample_rate
        self.server_binary = server_binary or SERVER_BINARY
        self.max_decode_steps = max_decode_steps
        self.max_queue = max_queue

        self._process: subprocess.Popen | None = None
        self._session = requests.Session()
        self._closed = False

        if not os.path.isfile(self.server_binary):
            raise ServerStartupError(
                f"Server binary not found: {self.server_binary}. "
                "Build voxcpm-server first."
            )
        if not os.path.isfile(self.model_path):
            raise ServerStartupError(
                f"Model file not found: {self.model_path}"
            )

        os.makedirs(self.voice_dir, exist_ok=True)

        self._start_server()
        atexit.register(self.close)

    # ── subprocess management ──────────────────────────────────────────────

    def _start_server(self) -> None:
        cmd = [
            self.server_binary,
            "--host", str(self.host),
            "--port", str(self.port),
            "--model-path", self.model_path,
            "--model-name", self.model_name,
            "--threads", str(self.threads),
            "--backend", self.backend,
            "--voice-dir", self.voice_dir,
            "--max-queue", str(self.max_queue),
            "--max-decode-steps", str(self.max_decode_steps),
            "--output-sample-rate", str(self.output_sample_rate),
            "--disable-auth",
        ]

        log.info("Starting voxcpm-server on %s:%s", self.host, self.port)
        log.debug("Command: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL),
            )
        except OSError as e:
            raise ServerStartupError(
                f"Failed to launch server binary: {e}"
            )

        self._start_log_threads()

        time.sleep(0.5)
        if self._process.poll() is not None:
            self._drain_logs()
            rc = self._process.returncode
            raise ServerStartupError(
                f"Server exited immediately (code {rc}). "
                "Check that the model path, binary, and port are correct."
            )

        self._base_url = f"http://{self.host}:{self.port}"

    def _start_log_threads(self) -> None:
        def _reader(stream: IO[bytes], prefix: str) -> None:
            for line in iter(stream.readline, b""):
                log.debug("[%s] %s", prefix, line.decode("utf-8", errors="replace").rstrip())
            stream.close()

        assert self._process is not None
        if self._process.stdout:
            t = threading.Thread(
                target=_reader,
                args=(self._process.stdout, "voxcpm-server:out"),
                daemon=True,
            )
            t.start()
        if self._process.stderr:
            t = threading.Thread(
                target=_reader,
                args=(self._process.stderr, "voxcpm-server:err"),
                daemon=True,
            )
            t.start()

    def _drain_logs(self) -> None:
        if self._process is None:
            return
        for stream, prefix in [
            (self._process.stdout, "voxcpm-server:out"),
            (self._process.stderr, "voxcpm-server:err"),
        ]:
            if stream:
                for line in stream:
                    log.debug(
                        "[%s] %s", prefix,
                        line.decode("utf-8", errors="replace").rstrip()
                    )

    # ── health check ───────────────────────────────────────────────────────

    def wait_until_ready(self, timeout: float = 60.0) -> None:
        """Poll ``GET /healthz`` until the server returns 200.

        Raises:
            ServerStartupError: if the server process has already exited.
            ServerTimeoutError: if not ready within *timeout* seconds.
        """
        if self._closed:
            raise ServerStartupError("Server has been closed.")
        if self._process is not None and self._process.poll() is not None:
            self._drain_logs()
            raise ServerStartupError(
                f"Server exited with code {self._process.returncode} "
                "before becoming ready."
            )

        deadline = time.monotonic() + timeout
        delay = 0.1
        last_exc: Exception | None = None

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                self._drain_logs()
                raise ServerStartupError(
                    f"Server exited with code {self._process.returncode} "
                    "while waiting for readiness."
                )
            try:
                resp = self._session.get(f"{self._base_url}/healthz", timeout=5.0)
                if resp.status_code == 200:
                    log.info("Server ready at %s", self._base_url)
                    return
            except requests.RequestException as e:
                last_exc = e

            time.sleep(delay)
            delay = min(delay * 2, 5.0)

        raise ServerTimeoutError(
            f"Server at {self._base_url} did not become ready within {timeout}s. "
            f"Last error: {last_exc}"
        )

    # ── API methods ────────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        try:
            return self._base_url
        except AttributeError:
            raise ServerStartupError("Server has not been started yet.")

    def register_voice(self, voice_id: str, wav_path: str, text: str) -> dict:
        """Register a reference voice from a WAV file and its transcript.

        Args:
            voice_id: Unique identifier for the voice (e.g. ``"male"``).
            wav_path: Path to a 16-bit mono WAV file with the reference audio.
            text: Transcript of the reference audio.

        Returns:
            Server response parsed as a dict.
        """
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Reference WAV not found: {wav_path}")

        url = f"{self.base_url}/v1/voices"
        with open(wav_path, "rb") as fh:
            files = {
                "id": (None, voice_id),
                "text": (None, text),
                "audio": (os.path.basename(wav_path), fh, "audio/wav"),
            }
            resp = self._session.post(url, files=files, timeout=120.0)

        if not resp.ok:
            raise APIError(
                f"Voice registration failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    def synthesize(
        self,
        voice_id: str,
        text: str,
        response_format: str = "wav",
        stream_format: str | None = None,
    ) -> bytes:
        """Synthesise speech for the given text using a registered voice.

        Args:
            voice_id: Voice identifier previously registered via
                :meth:`register_voice`.
            text: Text to synthesise.
            response_format: Output container format (``"wav"``, ``"mp3"``).
            stream_format: Streaming format (default ``"wav"``).

        Returns:
            Raw audio bytes (WAV by default).
        """
        url = f"{self.base_url}/v1/audio/speech"
        payload = {
            "model": self.model_name,
            "input": text,
            "voice": voice_id,
            "response_format": response_format,
        }
        if stream_format is not None:
            payload["stream_format"] = stream_format
        resp = self._session.post(url, json=payload, timeout=300.0)

        if not resp.ok:
            raise APIError(
                f"Synthesis failed ({resp.status_code}): {resp.text}"
            )
        return resp.content

    def delete_voice(self, voice_id: str) -> dict:
        """Delete a previously registered voice.

        Args:
            voice_id: Voice identifier to remove.

        Returns:
            Server response parsed as a dict.
        """
        url = f"{self.base_url}/v1/voices/{voice_id}"
        resp = self._session.delete(url, timeout=30.0)

        if not resp.ok:
            raise APIError(
                f"Voice deletion failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop the server subprocess and release resources."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)

        self._session.close()

        if self._process is not None:
            proc = self._process
            self._process = None

            log.info("Stopping voxcpm-server (pid %d)…", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                log.warning("Server did not terminate gracefully; killing.")
                proc.kill()
                proc.wait(timeout=5.0)

            self._drain_logs()
            log.info("Server stopped (exit code %d)", proc.returncode)

    def __enter__(self) -> VoxCPMCppServer:
        return self

    def __exit__(self, *args) -> None:
        self.close()
