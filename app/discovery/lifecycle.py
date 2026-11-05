"""Asset lifecycle FSM ported from myso-discovery-service-core."""

from __future__ import annotations

from enum import Enum


class AssetLifecycleState(str, Enum):
    DISCOVERED = "discovered"
    NORMALIZED = "normalized"
    QUEUED = "queued"
    ACQUIRING = "acquiring"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    MATCHED = "matched"
    PROVENANCE_CONFIRMED = "provenance_confirmed"
    VAULT_ELIGIBLE = "vault_eligible"
    VAULT_CREATED = "vault_created"
    CLAIMED = "claimed"
    FAILED = "failed"
    EXCLUDED = "excluded"
    STALE = "stale"
    SUPERSEDED = "superseded"

    def is_at_least_indexed(self) -> bool:
        return self in {
            AssetLifecycleState.INDEXED,
            AssetLifecycleState.MATCHED,
            AssetLifecycleState.PROVENANCE_CONFIRMED,
            AssetLifecycleState.VAULT_ELIGIBLE,
            AssetLifecycleState.VAULT_CREATED,
            AssetLifecycleState.CLAIMED,
        }

    def is_embed_in_progress(self) -> bool:
        return self in {
            AssetLifecycleState.QUEUED,
            AssetLifecycleState.ACQUIRING,
            AssetLifecycleState.EMBEDDED,
        }

    def needs_embed_enqueue(self) -> bool:
        return self in {
            AssetLifecycleState.DISCOVERED,
            AssetLifecycleState.NORMALIZED,
            AssetLifecycleState.FAILED,
        }


class LifecycleEvent(str, Enum):
    NORMALIZE = "normalize"
    ENQUEUE = "enqueue"
    START_ACQUIRE = "start_acquire"
    EMBED_COMPLETE = "embed_complete"
    INDEX_COMPLETE = "index_complete"
    MATCH_DETECTED = "match_detected"
    PROVENANCE_CONFIRMED = "provenance_confirmed"
    VAULT_ELIGIBLE = "vault_eligible"
    VAULT_CREATED = "vault_created"
    CLAIMED = "claimed"
    FAIL = "fail"
    EXCLUDE = "exclude"
    MARK_STALE = "mark_stale"
    SUPERSEDE = "supersede"


# Callback event names from oracle worker (legacy discovery HTTP names).
CALLBACK_EVENT_MAP: dict[str, LifecycleEvent] = {
    "match_detected": LifecycleEvent.MATCH_DETECTED,
    "provenance_confirmed": LifecycleEvent.PROVENANCE_CONFIRMED,
    "vault_eligible": LifecycleEvent.VAULT_ELIGIBLE,
    "vault_claimable": LifecycleEvent.VAULT_ELIGIBLE,
    "vault_created": LifecycleEvent.VAULT_CREATED,
    "claimed": LifecycleEvent.CLAIMED,
    "vault_claimed": LifecycleEvent.CLAIMED,
}


class LifecycleError(Exception):
    def __init__(self, from_state: AssetLifecycleState, event: LifecycleEvent) -> None:
        super().__init__(f"invalid transition from {from_state.value} via {event.value}")
        self.from_state = from_state
        self.event = event


def transition(
    from_state: AssetLifecycleState,
    event: LifecycleEvent,
) -> AssetLifecycleState:
    s = AssetLifecycleState
    e = LifecycleEvent
    table: dict[tuple[AssetLifecycleState, LifecycleEvent], AssetLifecycleState] = {
        (s.DISCOVERED, e.NORMALIZE): s.NORMALIZED,
        (s.NORMALIZED, e.ENQUEUE): s.QUEUED,
        (s.FAILED, e.ENQUEUE): s.QUEUED,
        (s.QUEUED, e.START_ACQUIRE): s.ACQUIRING,
        (s.FAILED, e.START_ACQUIRE): s.ACQUIRING,
        (s.ACQUIRING, e.START_ACQUIRE): s.ACQUIRING,
        (s.ACQUIRING, e.EMBED_COMPLETE): s.EMBEDDED,
        (s.EMBEDDED, e.INDEX_COMPLETE): s.INDEXED,
        (s.INDEXED, e.MATCH_DETECTED): s.MATCHED,
        (s.MATCHED, e.PROVENANCE_CONFIRMED): s.PROVENANCE_CONFIRMED,
        (s.PROVENANCE_CONFIRMED, e.VAULT_ELIGIBLE): s.VAULT_ELIGIBLE,
        (s.VAULT_ELIGIBLE, e.VAULT_CREATED): s.VAULT_CREATED,
        (s.VAULT_CREATED, e.CLAIMED): s.CLAIMED,
        (s.INDEXED, e.SUPERSEDE): s.SUPERSEDED,
    }
    if event in (e.FAIL,):
        return s.FAILED
    if event in (e.EXCLUDE,):
        return s.EXCLUDED
    if event in (e.MARK_STALE,):
        return s.STALE
    key = (from_state, event)
    if key not in table:
        raise LifecycleError(from_state, event)
    return table[key]


def parse_callback_event(raw: str) -> LifecycleEvent | None:
    return CALLBACK_EVENT_MAP.get(raw.strip().lower())
