"""
ui_state.py — Thread-safe state store for the assistant UI.

Holds the current speaking/text state that the React frontend polls.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class AssistantUIState:
    speaking: bool
    text: str
    updated_at: float


class AssistantUIStateStore:
    def __init__(self, on_change: Optional[Callable[[AssistantUIState], None]] = None):
        self._lock = threading.Lock()
        self._state = AssistantUIState(speaking=False, text="", updated_at=time.time())
        self._on_change = on_change

    def get(self) -> AssistantUIState:
        with self._lock:
            return self._state

    def set(self, *, speaking: Optional[bool] = None, text: Optional[str] = None) -> AssistantUIState:
        with self._lock:
            next_state = AssistantUIState(
                speaking=self._state.speaking if speaking is None else speaking,
                text=self._state.text if text is None else text,
                updated_at=time.time(),
            )
            self._state = next_state

        if self._on_change:
            try:
                self._on_change(next_state)
            except Exception:
                pass

        return next_state
