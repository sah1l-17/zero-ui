"""
ui_server.py — Lightweight HTTP server that exposes assistant state to the React frontend.

Endpoints:
  GET /ui/state  →  JSON { "speaking": bool, "text": str }
  GET /ui/ws     →  426 (WebSocket placeholder; frontend falls back to polling)

The server runs in a daemon thread so the main assistant loop is not blocked.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

from ui_state import AssistantUIState, AssistantUIStateStore


class _AssistantUIRequestHandler(BaseHTTPRequestHandler):
    store: Optional[AssistantUIStateStore] = None

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path == "/ui/state":
            if self.store:
                s = self.store.get()
            else:
                s = AssistantUIState(speaking=False, text="", updated_at=0.0)

            payload = json.dumps({"speaking": s.speaking, "text": s.text}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/ui/ws":
            self.send_response(426)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"WebSocket not enabled; use /ui/state polling.")
            return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not found")

    def log_message(self, *_args, **_kwargs) -> None:
        # Keep terminal output clean.
        return


class AssistantUIServer:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000):
        self._host = host
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._httpd: Optional[ThreadingHTTPServer] = None

    def attach_store(self, store: AssistantUIStateStore) -> None:
        _AssistantUIRequestHandler.store = store

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _serve() -> None:
            self._httpd = ThreadingHTTPServer(
                (self._host, self._port), _AssistantUIRequestHandler
            )
            self._httpd.serve_forever(poll_interval=0.25)

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        print(f"🌐 UI server running on http://{self._host}:{self._port}")

    def stop(self) -> None:
        if not self._httpd:
            return
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass

    def schedule_broadcast(self, _state: AssistantUIState) -> None:
        # Placeholder for a future push transport (WebSocket/SSE).
        return
