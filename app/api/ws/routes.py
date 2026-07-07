"""WebSocket routes for oracle real-time events."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api.ws.hub import ws_hub
from app.network_config import SUPPORTED_NETWORKS, active_networks

router = APIRouter()


@router.websocket("/ws")
async def oracle_websocket(
    websocket: WebSocket,
    network: str | None = Query(default=None),
    topics: str | None = Query(default="sync,jobs"),
):
    selected = network or (active_networks()[0] if active_networks() else "localnet")
    topic_list = [t.strip() for t in (topics or "sync,jobs").split(",") if t.strip()]
    await ws_hub.connect(websocket, network=selected, topics=topic_list)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("action") == "subscribe":
                sub_network = msg.get("network") or selected
                sub_topics = msg.get("topics") or topic_list
                await ws_hub.disconnect(websocket)
                await ws_hub.connect(websocket, network=sub_network, topics=sub_topics)
                await websocket.send_json({"type": "subscribed", "data": {"network": sub_network, "topics": sub_topics}})
            elif msg.get("action") == "ping":
                await websocket.send_json({"type": "pong", "data": {}})
    except WebSocketDisconnect:
        await ws_hub.disconnect(websocket)
