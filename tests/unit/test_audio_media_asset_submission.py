from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib

from nacl.signing import SigningKey

from app.services.media_asset_submission import (
    ASSET_KIND_SOUND_RECORDING,
    MediaResolutionResult,
    RIGHT_DERIVATIVE_WORK,
    RIGHT_DIGITAL_AUDIO_TRANSMISSION,
    RIGHT_PUBLIC_PERFORMANCE,
    RIGHT_REPRODUCTION,
    USAGE_MUSIC_SOUNDTRACK,
    USAGE_STREAM,
    audio_rights_intent_hash,
    audio_rights_from_intent,
    build_finalize_media_asset_move_call,
    build_update_media_asset_usage_grants_move_call,
    usage_permitted,
)


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _owner(signing_key: SigningKey) -> str:
    return "0x" + hashlib.blake2b(b"\x00" + bytes(signing_key.verify_key), digest_size=32).hexdigest()


def _sign_intent(signing_key: SigningKey, intent: dict) -> str:
    message = audio_rights_intent_hash(intent).encode()
    digest = hashlib.blake2b(b"\x03\x00\x00" + _uleb128(len(message)) + message, digest_size=32).digest()
    signature = signing_key.sign(digest).signature
    return base64.b64encode(b"\x00" + signature + bytes(signing_key.verify_key)).decode()


def _intent(owner: str) -> dict:
    stream_rights = RIGHT_REPRODUCTION | RIGHT_PUBLIC_PERFORMANCE | RIGHT_DIGITAL_AUDIO_TRANSMISSION
    soundtrack_rights = stream_rights | RIGHT_DERIVATIVE_WORK
    return {
        "version": 1,
        "trackId": "track",
        "ownerWalletAddress": owner,
        "contentCommitment": "0x0102",
        "assetKind": "sound_recording",
        "provenance": {
            "policyVersion": 1,
            "basis": "original_rightsholder_attestation",
            "source": "user_upload",
            "declaration": "I created or control this audio.",
        },
        "issuedAt": datetime.now(timezone.utc).isoformat(),
        "expiresAt": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
        "grants": [
            {
                "usageClass": USAGE_STREAM,
                "grantedRights": stream_rights,
                "licenseType": 3,
                "compensationType": 0,
                "compensationBps": 0,
                "attributionRequired": True,
                "derivativesPermitted": False,
                "commercialUsePermitted": False,
                "effectiveFrom": 0,
                "expiresAt": None,
                "revocable": True,
            },
            {
                "usageClass": USAGE_MUSIC_SOUNDTRACK,
                "grantedRights": soundtrack_rights,
                "licenseType": 3,
                "compensationType": 0,
                "compensationBps": 0,
                "attributionRequired": True,
                "derivativesPermitted": True,
                "commercialUsePermitted": False,
                "effectiveFrom": 0,
                "expiresAt": None,
                "revocable": True,
            },
        ],
    }


def test_audio_intent_yields_only_stream_and_soundtrack_grants(monkeypatch) -> None:
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0x50c1")
    monkeypatch.setenv("MYSO_POC_CONFIG_ID", "0xconfig")
    signing_key = SigningKey.generate()
    owner = _owner(signing_key)
    intent = _intent(owner)
    rights = audio_rights_from_intent(
        intent=intent,
        wallet_signature=_sign_intent(signing_key, intent),
        submitter=owner,
        content_commitment_hex="0102",
    )
    assert usage_permitted(rights, USAGE_STREAM)
    assert usage_permitted(rights, USAGE_MUSIC_SOUNDTRACK)
    assert len(rights["claims"]) == 2
    resolution = MediaResolutionResult(
        request_id="0xrequest",
        content_commitment=b"\x01\x02",
        observed_fingerprint_commitment=b"\x03\x04",
        media_type=3,
        submitter=owner,
        asset_kind=ASSET_KIND_SOUND_RECORDING,
        claims=rights["claims"],
        usage_grants=rights["usage_grants"],
    )
    call = build_finalize_media_asset_move_call(resolution)
    assert call["arguments"][5] == ASSET_KIND_SOUND_RECORDING


def test_update_grants_builder_targets_user_signed_move_call(monkeypatch) -> None:
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0x50c1")
    signing_key = SigningKey.generate()
    owner = _owner(signing_key)
    intent = _intent(owner)
    rights = audio_rights_from_intent(
        intent=intent, wallet_signature=_sign_intent(signing_key, intent), submitter=owner,
        content_commitment_hex="0102",
    )
    call = build_update_media_asset_usage_grants_move_call(
        media_asset_id="0x" + "22" * 32,
        usage_grants=rights["usage_grants"],
    )
    assert call["module"] == "media_asset"
    assert call["function"] == "update_media_asset_usage_grants"
