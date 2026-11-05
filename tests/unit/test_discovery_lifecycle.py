"""Tests for corpus lifecycle FSM."""

from app.discovery.lifecycle import AssetLifecycleState, LifecycleEvent, transition


def test_happy_path_to_indexed():
    state = AssetLifecycleState.DISCOVERED
    state = transition(state, LifecycleEvent.NORMALIZE)
    state = transition(state, LifecycleEvent.ENQUEUE)
    state = transition(state, LifecycleEvent.START_ACQUIRE)
    state = transition(state, LifecycleEvent.EMBED_COMPLETE)
    state = transition(state, LifecycleEvent.INDEX_COMPLETE)
    assert state == AssetLifecycleState.INDEXED


def test_failed_asset_can_retry():
    state = transition(AssetLifecycleState.FAILED, LifecycleEvent.ENQUEUE)
    assert state == AssetLifecycleState.QUEUED


def test_callback_event_mapping():
    from app.discovery.lifecycle import parse_callback_event

    assert parse_callback_event("vault_claimable") == LifecycleEvent.VAULT_ELIGIBLE
    assert parse_callback_event("vault_created") == LifecycleEvent.VAULT_CREATED
