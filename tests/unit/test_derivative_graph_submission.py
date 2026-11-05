"""Unit tests for Phase 2 derivative graph MoveCall shapes."""

from __future__ import annotations

import pytest

from app.services.derivative_graph_submission import (
    ParentEdgeInput,
    PendingAssetInput,
    build_add_derivative_parent_edge_to_pending_move_call,
    build_create_pending_derivative_asset_move_call,
    build_derivative_finalize_ptb,
    build_finalize_derivative_asset_move_call,
    build_finalize_pending_as_original_move_call,
    build_materialize_initial_resolved_policy_move_call,
    build_original_finalize_ptb,
    build_propose_detected_relationship_move_call,
)


@pytest.fixture
def poc_env(monkeypatch):
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0xpackage")
    monkeypatch.setenv("MYSO_POC_CONFIG_ID", "0xconfig")


def test_create_pending_move_shape(poc_env):
    mc = build_create_pending_derivative_asset_move_call(
        PendingAssetInput(content_commitment=b"\x01\x02", media_type=2)
    )
    assert mc["module"] == "media_asset"
    assert mc["function"] == "create_pending_derivative_asset"
    assert len(mc["arguments"]) == 4


def test_derivative_finalize_ptb_golden(poc_env):
    pending = PendingAssetInput(content_commitment=b"\xaa\xbb", media_type=1)
    parents = [
        ParentEdgeInput(
            parent_asset_id="0xparent",
            license_instance_id="0xlicense",
            template_version_id="0xtemplate",
        )
    ]
    recipe = build_derivative_finalize_ptb(pending, parents)
    steps = recipe.to_dict()["steps"]
    assert len(steps) == 3
    assert steps[0]["function"] == "create_pending_derivative_asset"
    assert steps[1]["function"] == "add_derivative_parent_edge_to_pending"
    assert steps[2]["function"] == "finalize_derivative_asset"


def test_original_finalize_ptb_golden(poc_env):
    pending = PendingAssetInput(content_commitment=b"\xcc", media_type=1)
    recipe = build_original_finalize_ptb(pending)
    steps = recipe.to_dict()["steps"]
    assert steps[-1]["function"] == "finalize_pending_as_original"


def test_policy_and_discovery_builders(poc_env):
    assert build_materialize_initial_resolved_policy_move_call("0xasset")["function"] == "materialize_initial_resolved_policy"
    assert build_finalize_derivative_asset_move_call("0xpending")["function"] == "finalize_derivative_asset"
    assert build_finalize_pending_as_original_move_call("0xpending")["function"] == "finalize_pending_as_original"
    propose = build_propose_detected_relationship_move_call(
        config_id="0xconfig",
        accused_pending_id="0xpending",
        original_asset_id="0xorig",
        similarity_bps=9000,
    )
    assert propose["function"] == "propose_detected_relationship"
    edge = build_add_derivative_parent_edge_to_pending_move_call(
        pending_id="0xpending",
        parent_asset_id="0xparent",
        license_instance_id="0xlic",
        template_version_id="0xtpl",
    )
    assert edge["function"] == "add_derivative_parent_edge_to_pending"
