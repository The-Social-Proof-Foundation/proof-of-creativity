"""BCS decoder for MySocial post::PostCreatedEvent layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BcsDecodeError(ValueError):
    pass


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

    def read_vec_string(self) -> list[str]:
        length = self.read_uleb128()
        return [self.read_string() for _ in range(length)]


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
    revenue_redirect_to: str | None
    revenue_redirect_percentage: int | None
    enable_spt: bool
    enable_poc: bool
    enable_spot: bool
    spot_id: str | None
    spt_id: str | None
    poc_redirection_kind: int = 0
    actor_address: str | None = None
    sub_agent_id: str | None = None
    organization_id: str | None = None
    action_identity_class: int = 0

    def to_payload(self) -> dict[str, Any]:
        media_urls = self.media_urls or []
        return {
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
            "promotion_id": self.promotion_id,
            "revenue_redirect_to": self.revenue_redirect_to,
            "revenue_redirect_percentage": self.revenue_redirect_percentage,
            "enable_spt": self.enable_spt,
            "enable_poc": self.enable_poc,
            "enable_spot": self.enable_spot,
            "spot_id": self.spot_id,
            "spt_id": self.spt_id,
            "poc_redirection_kind": self.poc_redirection_kind,
            "actor_address": self.actor_address or self.owner,
            "sub_agent_id": self.sub_agent_id,
            "organization_id": self.organization_id,
            "action_identity_class": self.action_identity_class,
        }


def _read_post_created_core(reader: BcsReader) -> PostCreatedEventDecoded:
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
    )


def _decode_with_organization(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.organization_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_with_attribution(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = reader.read_address()
    ev.sub_agent_id = reader.read_option_move_object_id()
    ev.action_identity_class = reader.read_u8()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_current(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_core(reader)
    ev.poc_redirection_kind = reader.read_u8()
    ev.actor_address = ev.owner
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return ev


def _decode_legacy(reader: BcsReader) -> PostCreatedEventDecoded:
    ev = _read_post_created_core(reader)
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
        _decode_with_organization,
        _decode_with_attribution,
        _decode_current,
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
