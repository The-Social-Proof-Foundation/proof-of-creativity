from app.services.post_enforcement_submission import (
    MEDIA_COMPONENT_AUDIO,
    EmbeddedAssetBinding,
    audio_edit_recipe_commitment,
    deterministic_binding_id,
    encode_embedded_binding,
    legacy_dripdrop_binding_id,
)


def test_audio_binding_uses_component_three_and_recipe_evidence(monkeypatch) -> None:
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0x50c1")
    post_id = "0x" + "11" * 32
    asset_id = "0x" + "22" * 32
    recipe = {"trackId": "track", "mode": "mix", "musicGain": 0.75}
    binding = EmbeddedAssetBinding(
        binding_id=deterministic_binding_id(post_id, asset_id, 5),
        source_asset_id=asset_id,
        usage_class=5,
        stem=0,
        media_component=MEDIA_COMPONENT_AUDIO,
        evidence_commitment=audio_edit_recipe_commitment(recipe),
    )
    encoded = encode_embedded_binding(binding)["fields"]
    assert encoded["usage_class"] == "5"
    assert encoded["stem"] == "0"
    assert encoded["media_component"] == "3"
    assert encoded["evidence_commitment"]["Some"].startswith("0x")
    assert binding.binding_id == deterministic_binding_id(post_id, asset_id, 5)
    assert legacy_dripdrop_binding_id(post_id, asset_id, 5) != binding.binding_id
