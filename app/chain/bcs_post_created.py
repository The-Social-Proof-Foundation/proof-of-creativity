"""BCS decoder for MySocial post::PostCreatedEvent layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BcsDecodeError(ValueError):
    pass


@dataclass
class PostAccessDecoded:
    kind: str
    mydata_id: str | None = None
    subscription_service_id: str | None = None
    subscription_min_tier_level: int | None = None
    requires_subscription: bool = False


class BcsReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def _read(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise BcsDecodeError(f"unexpected EOF at offset {self._pos}")
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def read_u8(self) -> int:
        return self._read(1)[0]

    def read_u64(self) -> int:
        return int.from_bytes(self._read(8), "little")

    def read_bool(self) -> bool:
        return self.read_u8() != 0

    def read_uleb128(self) -> int:
        result = 0
        shift = 0
        while True:
            if self._pos >= len(self._data):
                raise BcsDecodeError("unexpected EOF in uleb128")
            byte = self._data[self._pos]
            self._pos += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                return result
            shift += 7
            if shift > 35:
                raise BcsDecodeError("uleb128 overflow")

    def read_string(self) -> str:
        length = self.read_uleb128()
        return self._read(length).decode("utf-8")

    def read_address(self) -> str:
        raw = self._read(32)
        return "0x" + raw.hex()

    def read_move_object_id(self) -> str:
        return self.read_address()

    def read_option_address(self) -> str | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        if tag != 1:
            raise BcsDecodeError(f"invalid Option tag: {tag}")
        return self.read_address()

    def read_option_move_object_id(self) -> str | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        if tag != 1:
            raise BcsDecodeError(f"invalid Option tag: {tag}")
        return self.read_move_object_id()

    def read_option_u64(self) -> int | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        if tag != 1:
            raise BcsDecodeError(f"invalid Option tag: {tag}")
        return self.read_u64()

    def read_option_string(self) -> str | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        if tag != 1:
            raise BcsDecodeError(f"invalid Option tag: {tag}")
        return self.read_string()

    def read_option_vec_address(self) -> list[str] | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        if tag != 1:
            raise BcsDecodeError(f"invalid Option tag: {tag}")
        return self.read_vec_address()

    def read_option_vec_string(self) -> list[str] | None:
        tag = self.read_u8()
        if tag == 0:
            return None
        if tag != 1:
            raise BcsDecodeError(f"invalid Option tag: {tag}")
        return self.read_vec_string()

    def read_vec_address(self) -> list[str]:
        length = self.read_uleb128()
        return [self.read_address() for _ in range(length)]

    def read_vec_move_object_id(self) -> list[str]:
        length = self.read_uleb128()
        return [self.read_move_object_id() for _ in range(length)]

    def read_vec_u8(self) -> bytes:
        length = self.read_uleb128()
        return self._read(length)

    def read_vec_string(self) -> list[str]:
        length = self.read_uleb128()
        return [self.read_string() for _ in range(length)]

    def read_post_access(self) -> PostAccessDecoded:
        """Decode Move post::PostAccess (Rust/BCS enum index 0/1/2)."""
        variant = self.read_uleb128()
        if variant == 0:
            return PostAccessDecoded(kind="public", requires_subscription=False)
        if variant == 1:
            service_id = self.read_move_object_id()
            mydata_id = self.read_option_move_object_id()
            min_tier = self.read_option_u64()
            return PostAccessDecoded(
                kind="profile_subscription",
                mydata_id=mydata_id,
                subscription_service_id=service_id,
                subscription_min_tier_level=min_tier,
                requires_subscription=True,
            )
        if variant == 2:
            mydata_id = self.read_move_object_id()
            return PostAccessDecoded(
                kind="marketplace_one_time",
                mydata_id=mydata_id,
                requires_subscription=False,
            )
        raise BcsDecodeError(f"invalid PostAccess variant: {variant}")


@dataclass
class PostCreatedEventDecoded:
    post_id: str
    owner: str
    profile_id: str
    platform_id: str
    permissions: int
    content: str
    post_type: str
    parent_post_id: str | None
    mentions: list[str] | None
    media_urls: list[str] | None
    metadata_json: str | None
    mydata_id: str | None
    promotion_id: str | None
    media_asset_ids: list[str] = field(default_factory=list)
    composition_status: int = 0
    monetization_status: int = 0
    revenue_redirect_to: str | None = None
    revenue_redirect_percentage: int | None = None
    enable_spt: bool = False
    spt_id: str | None = None
    poc_redirection_kind: int = 0
    actor_address: str | None = None
    sub_agent_id: str | None = None
    organization_id: str | None = None
    action_identity_class: int = 0
    post_access_kind: str = "public"
    subscription_service_id: str | None = None
    subscription_min_tier_level: int | None = None
    requires_subscription: bool = False
    # Legacy-only fields (not on current Move wire layout).
    enable_poc: bool | None = None
    enable_spot: bool = False
    spot_id: str | None = None
    access: PostAccessDecoded | None = field(default=None, repr=False)

    def to_payload(self) -> dict[str, Any]:
        media_urls = self.media_urls or []
        payload: dict[str, Any] = {
            "post_id": self.post_id,
            "owner": self.owner,
            "creator": self.owner,
            "profile_id": self.profile_id,
            "platform_id": self.platform_id,
            "permissions": self.permissions,
            "content": self.content,
            "post_type": self.post_type,
            "parent_post_id": self.parent_post_id,
            "mentions": self.mentions,
            "media_urls": media_urls,
            "metadata_json": self.metadata_json,
            "mydata_id": self.mydata_id,
            "post_access_kind": self.post_access_kind,
            "subscription_service_id": self.subscription_service_id,
            "subscription_min_tier_level": self.subscription_min_tier_level,
            "requires_subscription": self.requires_subscription,
            "promotion_id": self.promotion_id,
            "media_asset_ids": list(self.media_asset_ids),
            "composition_status": self.composition_status,
            "monetization_status": self.monetization_status,
            "revenue_redirect_to": self.revenue_redirect_to,
            "revenue_redirect_percentage": self.revenue_redirect_percentage,
            "enable_spt": self.enable_spt,
            "spt_id": self.spt_id,
            "poc_redirection_kind": self.poc_redirection_kind,
            "actor_address": self.actor_address or self.owner,
            "sub_agent_id": self.sub_agent_id,
            "organization_id": self.organization_id,
            "action_identity_class": self.action_identity_class,
        }
        # Omit enable_poc when absent so parse_post_created defaults to True.
        if self.enable_poc is not None:
            payload["enable_poc"] = self.enable_poc
        if self.spot_id is not None:
            payload["spot_id"] = self.spot_id
            payload["enable_spot"] = self.enable_spot
        return payload


def _apply_access(ev: PostCreatedEventDecoded, access: PostAccessDecoded) -> None:
    ev.access = access
    ev.post_access_kind = access.kind
    ev.mydata_id = access.mydata_id
    ev.subscription_service_id = access.subscription_service_id
    ev.subscription_min_tier_level = access.subscription_min_tier_level
    ev.requires_subscription = access.requires_subscription


def _read_post_created_media_asset_core(reader: BcsReader) -> PostCreatedEventDecoded:
    """Current Move PostCreatedEvent with media_asset_ids + composition/monetization status."""
    post_id = reader.read_address()
    owner = reader.read_address()
    profile_id = reader.read_address()
    platform_id = reader.read_address()
    permissions = reader.read_u8()
    content = reader.read_string()
    post_type = reader.read_string()
    parent_post_id = reader.read_option_address()
    mentions = reader.read_option_vec_address()
    media_asset_ids = reader.read_vec_move_object_id()
    media_urls = reader.read_option_vec_string()
    metadata_json = reader.read_option_string()
    access = reader.read_post_access()
    promotion_id = reader.read_option_address()
    composition_status = reader.read_u8()
    monetization_status = reader.read_u8()
    enable_spt = reader.read_bool()
    spt_id = reader.read_option_address()
    ev = PostCreatedEventDecoded(
        post_id=post_id,
        owner=owner,
        profile_id=profile_id,
        platform_id=platform_id,
        permissions=permissions,
        content=content,
        post_type=post_type,
        parent_post_id=parent_post_id,
        mentions=mentions,
        media_urls=media_urls,
        metadata_json=metadata_json,
        mydata_id=None,
        promotion_id=promotion_id,
        media_asset_ids=media_asset_ids,
        composition_status=composition_status,
        monetization_status=monetization_status,
        enable_spt=enable_spt,
        spt_id=spt_id,
    )
    _apply_access(ev, access)
    return ev


def _decode_media_asset_with_organization(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_media_asset_core(reader)
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.organization_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_media_asset_with_attribution(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_media_asset_core(reader)
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _read_post_created_current_core(reader: BcsReader) -> PostCreatedEventDecoded:
    """Current Move PostCreatedEvent fields through spt_id (includes PostAccess)."""
    post_id = reader.read_address()
    owner = reader.read_address()
    profile_id = reader.read_address()
    platform_id = reader.read_address()
    permissions = reader.read_u8()
    content = reader.read_string()
    post_type = reader.read_string()
    parent_post_id = reader.read_option_address()
    mentions = reader.read_option_vec_address()
    media_urls = reader.read_option_vec_string()
    metadata_json = reader.read_option_string()
    access = reader.read_post_access()
    promotion_id = reader.read_option_address()
    revenue_redirect_to = reader.read_option_address()
    revenue_redirect_percentage = reader.read_option_u64()
    enable_spt = reader.read_bool()
    spt_id = reader.read_option_address()
    ev = PostCreatedEventDecoded(
        post_id=post_id,
        owner=owner,
        profile_id=profile_id,
        platform_id=platform_id,
        permissions=permissions,
        content=content,
        post_type=post_type,
        parent_post_id=parent_post_id,
        mentions=mentions,
        media_urls=media_urls,
        metadata_json=metadata_json,
        mydata_id=None,
        promotion_id=promotion_id,
        revenue_redirect_to=revenue_redirect_to,
        revenue_redirect_percentage=revenue_redirect_percentage,
        enable_spt=enable_spt,
        spt_id=spt_id,
    )
    _apply_access(ev, access)
    return ev


def _decode_current_with_organization(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_current_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.organization_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_current_with_attribution(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_current_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _read_post_created_legacy_core(reader: BcsReader) -> PostCreatedEventDecoded:
    """Pre-PostAccess layout: mydata_id + enable_poc/enable_spot/spot_id."""
    return PostCreatedEventDecoded(
        post_id=reader.read_address(),
        owner=reader.read_address(),
        profile_id=reader.read_address(),
        platform_id=reader.read_address(),
        permissions=reader.read_u8(),
        content=reader.read_string(),
        post_type=reader.read_string(),
        parent_post_id=reader.read_option_address(),
        mentions=reader.read_option_vec_address(),
        media_urls=reader.read_option_vec_string(),
        metadata_json=reader.read_option_string(),
        mydata_id=reader.read_option_address(),
        promotion_id=reader.read_option_address(),
        revenue_redirect_to=reader.read_option_address(),
        revenue_redirect_percentage=reader.read_option_u64(),
        enable_spt=reader.read_bool(),
        enable_poc=reader.read_bool(),
        enable_spot=reader.read_bool(),
        spot_id=reader.read_option_address(),
        spt_id=reader.read_option_address(),
        post_access_kind="legacy_mydata",
    )


def _decode_legacy_with_organization(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_legacy_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.organization_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_legacy_with_attribution(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_legacy_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_legacy_current(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_legacy_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = ev.owner
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_legacy(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_legacy_core(reader)
    ev.poc_redirection_kind = 0
    ev.actor_address = ev.owner
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def decode_post_created_event(contents: bytes) -> PostCreatedEventDecoded:
    """Decode PostCreatedEvent BCS bytes using the indexer layout fallback chain."""
    if not contents:
        raise BcsDecodeError("empty contents")

    decoders = (
        _decode_media_asset_with_organization,
        _decode_media_asset_with_attribution,
        _decode_current_with_organization,
        _decode_current_with_attribution,
        _decode_legacy_with_organization,
        _decode_legacy_with_attribution,
        _decode_legacy_current,
        _decode_legacy,
    )
    errors: list[str] = []
    for decoder in decoders:
        try:
            return decoder(BcsReader(contents))
        except BcsDecodeError as exc:
            errors.append(str(exc))
            continue
    raise BcsDecodeError("; ".join(errors))
