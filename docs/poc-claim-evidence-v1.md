# PoC Claim Evidence Hash v1

Shared contract between `proof-of-creativity` and `myso-identity-verification`.

## Payload

Canonical JSON (sorted keys, no whitespace):

```json
{
  "attested_x_handle": "creatorname",
  "beneficiary_id": "0x...",
  "identity_hash": "0x63726561746f726e616d65",
  "identity_source": 1,
  "v": 1,
  "verifier": "myso-identity-verification",
  "verified_at": 1720000000,
  "wallet": "0x..."
}
```

## Hash

`evidence_hash = blake2b-256(canonical_json_utf8)` → 32-byte digest, hex-encoded with `0x` prefix in API responses.

## Identity hash

For X username anchors:

```
identity_hash = "0x" + hex(utf8(lowercase(trim(handle))))
```

Golden example: handle `CreatorName` → canonical `creatorname` → identity_hash `0x63726561746f726e616d65`.

## Verifier labels

| Environment | `verifier` field |
|-------------|------------------|
| Production | `myso-identity-verification` |
| Localnet mock | `mock` |
| Legacy dev | `x-oauth` |
