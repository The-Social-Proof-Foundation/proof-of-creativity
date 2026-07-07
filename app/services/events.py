"""Internal asyncio event bus for WebSocket fan-out."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

Listener = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)
        self._global: list[Listener] = []

    def subscribe(self, event_type: str, listener: Listener) -> None:
        self._listeners[event_type].append(listener)

    def subscribe_all(self, listener: Listener) -> None:
        self._global.append(listener)

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        message = {"type": event_type, "data": data}
        tasks = [listener(event_type, message) for listener in self._global]
        tasks.extend(listener(event_type, message) for listener in self._listeners.get(event_type, []))
        tasks.extend(listener(event_type, message) for listener in self._listeners.get("*", []))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


event_bus = EventBus()
