"""PTB builder utilities."""

from __future__ import annotations

from app.chain.ptb_builder import PtbRecipe, resolve_step_arguments, result_ref


def test_ptb_recipe_roundtrip():
    recipe = PtbRecipe.from_move_calls(
        [
            {
                "packageObjectId": "0xpkg",
                "module": "media_asset",
                "function": "create_pending_derivative_asset",
                "arguments": ["0xabc"],
            }
        ],
        description="test",
    )
    data = recipe.to_dict()
    assert data["description"] == "test"
    assert data["steps"][0]["function"] == "create_pending_derivative_asset"


def test_resolve_result_ref():
    args = resolve_step_arguments(
        ["0xclock", result_ref(0)],
        created_objects={0: "0xpending"},
    )
    assert args[1] == "0xpending"
