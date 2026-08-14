"""Build PoC transaction parameters from similarity results + on-chain config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from app.models.similarity import MediaMatch, PocChainSummary, TrackMatchSummary, VideoTrackAttestation

from app.services.poc_utils import (
    MEDIA_TYPE_AUDIO,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    OUTCOME_ROYALTY_FREE,
    similarity_float_to_u64_percent,
    truncate_evidence_urls,
    truncate_reasoning,
)

if TYPE_CHECKING:
    from app.services.poc_video_types import VideoSimilarityAnalysis


def best_match_for_attribution(matches: List[MediaMatch]) -> Optional[MediaMatch]:
    if not matches:
        return None
    high = [m for m in matches if m.confidence_level == "high"]
    pool = high if high else matches
    return max(pool, key=lambda m: m.similarity_score)


def build_image_or_audio_submission(
    matches: List[MediaMatch],
    media_type_code: int,
    image_threshold: int,
    video_threshold: int,
    audio_threshold: int,
    resolve_creator: Callable[[List[str]], Optional[str]],
) -> tuple[int, bool, Optional[str], str]:
    """Returns (similarity_u64, is_derivative_path, original_creator, reasoning_fragment)."""
    thr = (
        image_threshold
        if media_type_code == MEDIA_TYPE_IMAGE
        else audio_threshold
        if media_type_code == MEDIA_TYPE_AUDIO
        else video_threshold
    )
    score_u64 = similarity_float_to_u64_percent(max((m.similarity_score for m in matches), default=0.0))
    best = best_match_for_attribution(matches)
    creator: Optional[str] = None
    if best:
        creator = resolve_creator([best.media_id])
    derivative = score_u64 >= thr and creator is not None
    reasoning = ""
    if best:
        reasoning = (
            f"Best match media_id={best.media_id}, score_pct={similarity_float_to_u64_percent(best.similarity_score)}, "
            f"confidence={best.confidence_level}, threshold_pct={thr}."
        )
    else:
        reasoning = "No similarity matches recorded."
    return score_u64, derivative, creator if derivative else None, reasoning


def build_video_submission(
    analysis: "VideoSimilarityAnalysis",
    video_threshold: int,
    audio_threshold: int,
    resolve_creator: Callable[[List[str]], Optional[str]],
):
    from app.services.poc_video_types import decide_video_poc_for_chain

    visuals: List[str] = []
    if analysis.best_visual_match_media_id:
        visuals.append(analysis.best_visual_match_media_id)
    audio_ids: List[str] = []
    if analysis.best_audio_match_media_id:
        audio_ids.append(analysis.best_audio_match_media_id)

    visual_creator = resolve_creator(visuals) if visuals else None
    audio_creator = resolve_creator(audio_ids) if audio_ids else None
    return decide_video_poc_for_chain(analysis, video_threshold, audio_threshold, visual_creator, audio_creator)


def chain_effective_threshold_u64(
    media_type_code: int,
    embedded_audio_only_derivative: bool,
    poc_config: Dict[str, Any],
) -> int:
    img_thr = int(poc_config.get("image_threshold") or 95)
    vid_thr = int(poc_config.get("video_threshold") or 95)
    aud_thr = int(poc_config.get("audio_threshold") or 95)
    if media_type_code == MEDIA_TYPE_VIDEO and embedded_audio_only_derivative:
        return aud_thr
    if media_type_code == MEDIA_TYPE_VIDEO:
        return vid_thr
    if media_type_code == MEDIA_TYPE_AUDIO:
        return aud_thr
    return img_thr


def build_poc_chain_summary(bundle: Dict[str, Any], poc_config: Dict[str, Any]) -> PocChainSummary:
    """Interpret submission kwargs the same way Move uses thresholds + optional creator (non-explicit path)."""
    mt = int(bundle["media_type"])
    emb = bool(bundle.get("embedded_audio_only_derivative"))
    thr = chain_effective_threshold_u64(mt, emb, poc_config)
    score = int(bundle["highest_similarity_score"])
    creator = bundle.get("original_creator")
    apply_exp = bool(bundle.get("apply_explicit_outcome"))
    exp_out = int(bundle.get("explicit_poc_outcome") or 0)
    if apply_exp:
        would_deriv = False
    else:
        would_deriv = creator is not None and score >= thr
    return PocChainSummary(
        media_type_code=mt,
        highest_similarity_score_u64=score,
        effective_threshold_u64=thr,
        would_apply_derivative_redirect=would_deriv,
        embedded_audio_only_derivative=emb,
        original_creator=creator if isinstance(creator, str) else None,
        apply_explicit_outcome=apply_exp,
        explicit_poc_outcome=exp_out,
    )


def attribution_type_from_chain_summary(summary: PocChainSummary) -> str:
    if summary.apply_explicit_outcome and summary.explicit_poc_outcome == OUTCOME_ROYALTY_FREE:
        return "royalty_free"
    if summary.would_apply_derivative_redirect:
        return "derivative"
    return "original"


def build_upload_attribution_message(matches: List[MediaMatch], summary: Optional[PocChainSummary]) -> str:
    if summary is None:
        if not matches:
            return "No similar content found in corpus."
        return f"Found {len(matches)} match(es) (PoC chain summary unavailable)."

    if summary.apply_explicit_outcome and summary.explicit_poc_outcome == OUTCOME_ROYALTY_FREE:
        return (
            "Submitted explicit royalty-free PoC outcome (on-chain outcome 4); "
            "see chain for badge and vault behavior."
        )
    if summary.would_apply_derivative_redirect:
        return (
            f"Chain derivative redirect path: score {summary.highest_similarity_score_u64}/100 "
            f">= threshold {summary.effective_threshold_u64}/100 with resolvable original_creator."
        )
    if (
        summary.original_creator is None
        and summary.highest_similarity_score_u64 >= summary.effective_threshold_u64
        and not summary.apply_explicit_outcome
    ):
        return (
            f"Similarity {summary.highest_similarity_score_u64}/100 meets threshold "
            f"{summary.effective_threshold_u64}/100 but matched corpus media has no creator_address; "
            "Move treats this as non-derivative for redirect (original_creator unset)."
        )
    if not matches:
        return "No similar content found in corpus."
    return (
        f"Found {len(matches)} match(es); below on-chain derivative threshold "
        f"({summary.effective_threshold_u64}/100) or no attributable creator."
    )


def build_poc_submission_snapshot(bundle: Dict[str, Any], rpc_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge off-chain bundle with RPC metadata for persistence (proof_data / Redis)."""
    snap = dict(bundle)
    if rpc_result:
        snap["move_function"] = rpc_result.get("move_function")
        snap["resolved_spt_pool_id"] = rpc_result.get("resolved_spt_pool_id")
        snap["derivative_redirection_target"] = rpc_result.get("derivative_redirection_target")
        snap["tx_hash"] = rpc_result.get("tx_hash")
    return snap


def compose_e2e_submission_bundle(
    *,
    media_type_code: int,
    score: int,
    original_creator: str | None,
    derivative_target: int,
    poc_config: Dict[str, Any],
    royalty_free: bool = False,
) -> Dict[str, Any]:
    """Localnet/e2e override path for runnable scripts (explicit Move args via upload)."""
    mr = int(poc_config.get("max_reasoning_length") or 5000)
    reason = truncate_reasoning(
        f"E2E submission override score={score} creator={original_creator} target={derivative_target}",
        mr,
    )
    bundle: Dict[str, Any] = {
        "media_type": media_type_code,
        "highest_similarity_score": max(0, min(100, int(score))),
        "original_creator": original_creator,
        "derivative_redirection_target": int(derivative_target),
        "embedded_audio_only_derivative": False,
        "reasoning": reason,
        "evidence_urls": truncate_evidence_urls(["e2e:override"], int(poc_config.get("max_evidence_urls") or 10)),
        "apply_explicit_outcome": False,
        "explicit_poc_outcome": 0,
    }
    if royalty_free:
        bundle["apply_explicit_outcome"] = True
        bundle["explicit_poc_outcome"] = OUTCOME_ROYALTY_FREE
    return bundle


def compose_submission_bundle(
    *,
    media_kind: str,
    media_type_code: int,
    matches: List[MediaMatch],
    video_analysis: Optional["VideoSimilarityAnalysis"],
    poc_config: Dict[str, Any],
    resolve_creator: Callable[[List[str]], Optional[str]],
    royalty_free: bool = False,
) -> Dict[str, Any]:
    """
    Produce kwargs for MySocialClient.submit_poc_analysis (except post_id/spt overrides).

    Returns dict with reasoning, evidence_urls, embedded_audio_only_derivative,
    highest_similarity_score, original_creator, apply_explicit_outcome, explicit_poc_outcome, etc.
    """
    img_thr = int(poc_config.get("image_threshold") or 95)
    vid_thr = int(poc_config.get("video_threshold") or 95)
    aud_thr = int(poc_config.get("audio_threshold") or 95)

    mr = int(poc_config.get("max_reasoning_length") or 5000)
    me = int(poc_config.get("max_evidence_urls") or 10)

    evidence_seed: List[str] = []

    def pack(
        hi: int,
        emb: bool,
        creator: Optional[str],
        reason: str,
        extra_evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ev = list(evidence_seed)
        if extra_evidence:
            ev.extend(extra_evidence)
        ev = truncate_evidence_urls(ev, me) or None
        base: Dict[str, Any] = {
            "media_type": media_type_code,
            "highest_similarity_score": max(0, min(100, hi)),
            "original_creator": creator,
            "embedded_audio_only_derivative": emb,
            "reasoning": truncate_reasoning(reason, mr),
            "evidence_urls": ev,
        }
        if royalty_free:
            base["apply_explicit_outcome"] = True
            base["explicit_poc_outcome"] = OUTCOME_ROYALTY_FREE
            r = base["reasoning"] or ""
            base["reasoning"] = truncate_reasoning(
                (r + " Explicit PoC outcome: royalty-free (on-chain outcome 4)."),
                mr,
            )
        else:
            base["apply_explicit_outcome"] = False
            base["explicit_poc_outcome"] = 0
        return base

    if media_kind == "video" and video_analysis is not None:
        dec = build_video_submission(video_analysis, vid_thr, aud_thr, resolve_creator)
        ev_extra: List[str] = []
        if video_analysis.best_visual_match_media_id:
            ev_extra.append(f"visual_match_media_id:{video_analysis.best_visual_match_media_id}")
        if video_analysis.best_audio_match_media_id:
            ev_extra.append(f"audio_match_media_id:{video_analysis.best_audio_match_media_id}")

        reasoning = (
            dec.reasoning_summary
            + f" visual_similarity_pct={similarity_float_to_u64_percent(video_analysis.max_visual_similarity)};"
            f" embedded_audio_similarity_pct={similarity_float_to_u64_percent(video_analysis.embedded_audio_similarity)};"
            f" video_threshold={vid_thr}; audio_threshold={aud_thr}."
        )
        return pack(dec.highest_similarity_score_u64, dec.embedded_audio_only_derivative, dec.original_creator, reasoning, extra_evidence=ev_extra)
    score_u64, deriv, creator, reason = build_image_or_audio_submission(
        matches, media_type_code, img_thr, vid_thr, aud_thr, resolve_creator
    )
    for m in matches[:5]:
        evidence_seed.append(f"match:{m.media_id}:{m.match_type}:{similarity_float_to_u64_percent(m.similarity_score)}")
    return pack(score_u64, False, creator if deriv else None, reason)


def preview_poc_chain_summary_for_upload(
    *,
    media_cat: str,
    media_type_code: int,
    matches: List[MediaMatch],
    video_analysis: Optional["VideoSimilarityAnalysis"],
    poc_config: Dict[str, Any],
    royalty_free: bool = False,
) -> PocChainSummary:
    """Same bundle/summary as on-chain submit, without RPC (offline attribution messaging)."""

    def resolve_creator(media_ids: List[str]) -> Optional[str]:
        from app.core.database import lookup_mysocial_creator_for_media_ids

        return lookup_mysocial_creator_for_media_ids(media_ids)

    bundle = compose_submission_bundle(
        media_kind=media_cat,
        media_type_code=media_type_code,
        matches=matches,
        video_analysis=video_analysis,
        poc_config=poc_config,
        resolve_creator=resolve_creator,
        royalty_free=royalty_free,
    )
    return build_poc_chain_summary(bundle, poc_config)


def build_video_track_attestation(
    analysis: "VideoSimilarityAnalysis",
    poc_config: Dict[str, Any],
    resolve_creator: Callable[[List[str]], Optional[str]],
    *,
    embedded_only_sent_on_chain: bool,
) -> VideoTrackAttestation:
    vid_thr = int(poc_config.get("video_threshold") or 95)
    aud_thr = int(poc_config.get("audio_threshold") or 95)
    visuals = [analysis.best_visual_match_media_id] if analysis.best_visual_match_media_id else []
    audios = [analysis.best_audio_match_media_id] if analysis.best_audio_match_media_id else []
    vc = resolve_creator(visuals) if visuals else None
    ac = resolve_creator(audios) if audios else None
    vp_u = similarity_float_to_u64_percent(analysis.max_visual_similarity)
    ap_u = similarity_float_to_u64_percent(analysis.embedded_audio_similarity)
    video_deriv_flag = vp_u >= vid_thr and bool(vc)
    audio_deriv_flag = ap_u >= aud_thr and bool(ac)
    return VideoTrackAttestation(
        video_visual=TrackMatchSummary(
            similarity_score=float(analysis.max_visual_similarity),
            similarity_score_u64=vp_u,
            best_match_media_id=analysis.best_visual_match_media_id,
            derivative_by_chain_threshold=video_deriv_flag,
        ),
        embedded_audio=TrackMatchSummary(
            similarity_score=float(analysis.embedded_audio_similarity),
            similarity_score_u64=ap_u,
            best_match_media_id=analysis.best_audio_match_media_id,
            derivative_by_chain_threshold=audio_deriv_flag,
        ),
        embedded_audio_only_derivative_sent=embedded_only_sent_on_chain,
    )


def preview_video_track_attestation(
    analysis: "VideoSimilarityAnalysis",
    poc_config: Dict[str, Any],
) -> VideoTrackAttestation:
    def resolve_creator(media_ids: List[str]) -> Optional[str]:
        from app.core.database import lookup_mysocial_creator_for_media_ids

        return lookup_mysocial_creator_for_media_ids(media_ids)

    dec_preview = build_video_submission(
        analysis,
        int(poc_config.get("video_threshold") or 95),
        int(poc_config.get("audio_threshold") or 95),
        resolve_creator,
    )
    return build_video_track_attestation(
        analysis,
        poc_config=poc_config,
        resolve_creator=resolve_creator,
        embedded_only_sent_on_chain=dec_preview.embedded_audio_only_derivative,
    )


def build_composition_submission(
    *,
    post_id: str,
    assets: list[Any],
    manifest_entries: list[Any] | None = None,
    derivative_redirection_target: int = 0,
    max_embedded_asset_redirect_bps: int | None = None,
    contains_derivatives: bool = False,
    contains_unresolved_assets: bool = False,
    reasoning: str | None = None,
    evidence_urls: list[str] | None = None,
    spt_pool_id: str | None = None,
) -> Any:
    """Delegate to composition_submission for post-level Move args."""
    from app.services.composition_submission import (
        AssetVersionInput,
        build_composition_submission_from_assets,
    )

    def _coerce(raw: Any) -> AssetVersionInput:
        if isinstance(raw, AssetVersionInput):
            return raw
        if isinstance(raw, dict):
            return AssetVersionInput(**raw)
        raise TypeError(f"Expected AssetVersionInput or dict, got {type(raw)!r}")

    asset_inputs = [_coerce(a) for a in assets]
    manifest_inputs = [_coerce(m) for m in manifest_entries] if manifest_entries else None
    return build_composition_submission_from_assets(
        post_id=post_id,
        assets=asset_inputs,
        manifest_entries=manifest_inputs,
        derivative_redirection_target=derivative_redirection_target,
        max_embedded_asset_redirect_bps=max_embedded_asset_redirect_bps,
        contains_derivatives=contains_derivatives,
        contains_unresolved_assets=contains_unresolved_assets,
        reasoning=reasoning,
        evidence_urls=evidence_urls,
        spt_pool_id=spt_pool_id,
    )


def attempt_proof_of_creativity_submission(
    *,
    myso_client: Any,
    post_id: str,
    media_cat: str,
    media_type_code: int,
    matches: List[MediaMatch],
    video_analysis: Optional["VideoSimilarityAnalysis"],
    spt_pool_id: Optional[str],
    royalty_free: bool = False,
    e2e_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[VideoTrackAttestation], Dict[str, Any], PocChainSummary]:
    """Fetch PoC config, compose Move args, submit RPC transaction, return RPC + audit snapshot."""

    def resolve_creator(media_ids: List[str]) -> Optional[str]:
        from app.core.database import lookup_mysocial_creator_for_media_ids

        return lookup_mysocial_creator_for_media_ids(media_ids)

    cfg = myso_client.get_poc_config()
    if e2e_override:
        bundle = compose_e2e_submission_bundle(
            media_type_code=int(e2e_override.get("media_type") or media_type_code),
            score=int(e2e_override["score"]),
            original_creator=e2e_override.get("original_creator"),
            derivative_target=int(e2e_override.get("derivative_target") or 0),
            poc_config=cfg,
            royalty_free=bool(e2e_override.get("royalty_free") or royalty_free),
        )
    else:
        bundle = compose_submission_bundle(
            media_kind=media_cat,
            media_type_code=media_type_code,
            matches=matches,
            video_analysis=video_analysis,
            poc_config=cfg,
            resolve_creator=resolve_creator,
            royalty_free=royalty_free,
        )
    summary = build_poc_chain_summary(bundle, cfg)
    video_att: Optional[VideoTrackAttestation] = None
    if media_cat == "video" and video_analysis is not None:
        video_att = preview_video_track_attestation(video_analysis, cfg)
        if royalty_free:
            video_att = video_att.model_copy(update={"embedded_audio_only_derivative_sent": False})
    res = myso_client.submit_poc_analysis(post_id, spt_pool_id=spt_pool_id, **bundle)
    snapshot = build_poc_submission_snapshot(bundle, res)
    return res, video_att, snapshot, summary
