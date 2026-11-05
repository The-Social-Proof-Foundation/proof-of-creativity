"""Programmable transaction block recipes for multi-step PoC Move flows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Placeholder token for arguments produced by a prior command in the same PTB.
RESULT_REF_PREFIX = "$result:"


@dataclass
class MoveCallStep:
    """One Move entry in a PTB recipe."""

    package_object_id: str
    module: str
    function: str
    type_arguments: list[str] = field(default_factory=list)
    arguments: list[Any] = field(default_factory=list)
    label: str | None = None

    def to_move_call_dict(self) -> dict[str, Any]:
        return {
            "packageObjectId": self.package_object_id,
            "module": self.module,
            "function": self.function,
            "typeArguments": list(self.type_arguments),
            "arguments": list(self.arguments),
        }


@dataclass
class PtbRecipe:
    """Ordered Move calls — serializable for wallet signing or oracle submission."""

    steps: list[MoveCallStep]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "steps": [s.to_move_call_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_move_calls(cls, calls: list[dict[str, Any]], *, description: str = "") -> PtbRecipe:
        steps = [
            MoveCallStep(
                package_object_id=str(c["packageObjectId"]),
                module=str(c["module"]),
                function=str(c["function"]),
                type_arguments=list(c.get("typeArguments") or []),
                arguments=list(c.get("arguments") or []),
                label=c.get("label"),
            )
            for c in calls
        ]
        return cls(steps=steps, description=description)


def result_ref(step_index: int, *, arg_index: int = 0) -> str:
    """Reference an output object from command `step_index` (client PTB builders)."""
    return f"{RESULT_REF_PREFIX}{step_index}:{arg_index}"


def is_result_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(RESULT_REF_PREFIX)


def parse_result_ref(value: str) -> tuple[int, int]:
    raw = value[len(RESULT_REF_PREFIX) :]
    if ":" in raw:
        step_s, arg_s = raw.split(":", 1)
        return int(step_s), int(arg_s)
    return int(raw), 0


def resolve_step_arguments(
    arguments: list[Any],
    *,
    created_objects: dict[int, str],
) -> list[Any]:
    """Replace `$result:N:M` placeholders with concrete object ids from prior txs."""
    resolved: list[Any] = []
    for arg in arguments:
        if is_result_ref(arg):
            step_idx, _ = parse_result_ref(arg)
            oid = created_objects.get(step_idx)
            if not oid:
                raise ValueError(f"Missing created object for PTB step {step_idx}")
            resolved.append(oid)
        else:
            resolved.append(arg)
    return resolved


def extract_created_object_id(tx_result: dict[str, Any]) -> str | None:
    """Best-effort parse of first created object id from executeTransactionBlock effects."""
    effects = tx_result.get("effects") or tx_result.get("status") or {}
    if isinstance(effects, dict):
        created = effects.get("created") or effects.get("mutated") or []
        if isinstance(created, list):
            for item in created:
                if isinstance(item, dict):
                    ref = item.get("reference") or item.get("objectId") or item.get("object_id")
                    if ref:
                        return str(ref.get("objectId") if isinstance(ref, dict) else ref)
    events = tx_result.get("events") or []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        parsed = ev.get("parsedJson") or ev.get("parsed_json") or {}
        for key in ("pending_id", "child_asset_id", "asset_id", "proposal_id", "request_id", "media_asset_id"):
            val = parsed.get(key)
            if val is None and isinstance(parsed.get("fields"), dict):
                val = parsed["fields"].get(key)
            if isinstance(val, dict):
                val = val.get("id") or val.get("value") or val.get("bytes")
            if val:
                return str(val)
    return None


def extract_event_field(tx_result: dict[str, Any], event_suffix: str, field: str) -> str | None:
    """Read a field from the first matching chain event (supports Move `fields` nesting)."""
    for ev in tx_result.get("events") or []:
        if not isinstance(ev, dict):
            continue
        ev_type = str(ev.get("type") or "")
        if event_suffix not in ev_type:
            continue
        parsed = ev.get("parsedJson") or ev.get("parsed_json") or {}
        root = parsed.get("fields") if isinstance(parsed.get("fields"), dict) else parsed
        val = root.get(field)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("id") or val.get("value") or val.get("bytes")
        if val:
            return str(val)
    return None


def extract_media_asset_resolved_id(tx_result: dict[str, Any]) -> str | None:
    return extract_event_field(tx_result, "MediaAssetResolvedEvent", "media_asset_id")
