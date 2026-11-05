"""WebSocket connection hub for oracle lifecycle events."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket
import structlog

logger = structlog.get_logger()


class WebSocketHub:
    def __init__(self) -> None:
        self._connections: dict[WebSocket, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, *, network: str, topics: list[str]) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[websocket] = {"network": network, "topics": set(topics or ["*"])}

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(websocket, None)

    async def broadcast(self, event_type: str, message: dict[str, Any]) -> None:
        data = message.get("data") or {}
        network = data.get("network")
        post_id = data.get("post_id")
        async with self._lock:
            targets = list(self._connections.items())
        dead: list[WebSocket] = []
        for ws, meta in targets:
            if network and meta.get("network") not in (network, "*"):
                continue
            topics = meta.get("topics") or set()
            if "*" not in topics:
                if "sync" in topics and event_type.startswith("sync."):
                    pass
                elif "jobs" in topics and "job" in event_type:
                    pass
                elif post_id and f"post:{post_id}" not in topics:
                    continue
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


ws_hub = WebSocketHub()
