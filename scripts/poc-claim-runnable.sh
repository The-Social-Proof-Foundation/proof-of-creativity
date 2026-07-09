#!/usr/bin/env bash
# Copyright (c) The Social Proof Foundation, LLC.
# SPDX-License-Identifier: Apache-2.0
#
# PoC username beneficiary claim E2E (localnet mock identity path).
# Provisions a beneficiary on-chain, claims via oracle API mock headers, and
# asserts Move events on the fullnode before GraphQL cross-check.
#
# Prerequisites:
#   - PoC API :8000 with POC_IDENTITY_VERIFIER=mock
#   - myso-core bootstrap + GraphQL session populated
#
# Usage:
#   ASSUME_YES=1 ./scripts/poc-claim-runnable.sh
#   POC_SESSION_FILE=../myso-core/network.config/poc/poc-e2e-session.env ./scripts/poc-claim-runnable.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MYSO_CORE="${MYSO_CORE:-$(cd "$ROOT/../myso-core" 2>/dev/null && pwd || true)}"
if [[ ! -d "$MYSO_CORE/scripts/lib" ]]; then
    echo "myso-core not found at $MYSO_CORE — set MYSO_CORE to the myso-core checkout" >&2
    exit 1
fi

REPO_ROOT="$MYSO_CORE"
export REPO_ROOT

POC_SESSION_FILE="${POC_SESSION_FILE:-$MYSO_CORE/network.config/poc/poc-e2e-session.env}"
POC_FALLBACK_SESSION="$MYSO_CORE/network.config/poc/poc-session.env"
API_URL="${POC_API_URL:-http://127.0.0.1:8000}"
IV_URL="${MYSO_IDENTITY_VERIFICATION_URL:-http://127.0.0.1:3000}"
NETWORK="${POC_NETWORK:-localnet}"
export POC_ORACLE_URL="$API_URL"
export POC_ORACLE_NETWORK="$NETWORK"

# shellcheck source=../myso-core/scripts/lib/social-runtime-common.sh
source "$MYSO_CORE/scripts/lib/social-runtime-common.sh"
# shellcheck source=../myso-core/scripts/lib/poc-oracle-common.sh
source "$MYSO_CORE/scripts/lib/poc-oracle-common.sh"
# shellcheck source=../myso-core/scripts/lib/poc-oracle-http.sh
source "$MYSO_CORE/scripts/lib/poc-oracle-http.sh"

SOCIAL_SESSION_SAVE_PATH="$POC_SESSION_FILE"
if [[ ! -f "$SOCIAL_SESSION_SAVE_PATH" && -f "$POC_FALLBACK_SESSION" ]]; then
    SOCIAL_SESSION_SAVE_PATH="$POC_FALLBACK_SESSION"
fi
export SOCIAL_SESSION_SAVE_PATH

log() { echo ">>> $*" >&2; }

health_checks() {
    if curl -sf "$IV_URL/health" >/dev/null 2>&1; then
        log "identity-verification: ok ($IV_URL)"
    else
        log "WARN: identity-verification not reachable at $IV_URL (optional for localnet mock)"
    fi
    poc_oracle_load_localnet_env
    sync_poc_config_oracle_on_chain "$POC_DEFAULT_ORACLE_ADDRESS" || return 1
    ensure_poc_oracle_key_in_env || return 1
    poc_oracle_health_ok || return 1
    log "oracle API: ok ($API_URL)"
}

run_claim_flow() {
    local poc_session
    poc_session="$MYSO_CORE/network.config/poc/poc-session.env"
    log "Running username provision + mock claim via myso-core proof-of-creativity-runnable"
    cp "$SOCIAL_SESSION_SAVE_PATH" "$poc_session"
    ASSUME_YES="${ASSUME_YES:-1}" SKIP_CONFIRM_RUN="${SKIP_CONFIRM_RUN:-1}" \
        "$MYSO_CORE/scripts/proof-of-creativity-runnable.sh" --username-flow
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                sed -n '2,16p' "$0" | sed 's/^# \?//'
                exit 0
                ;;
            -y) ASSUME_YES=1; shift ;;
            *) echo "Unknown option: $1" >&2; exit 1 ;;
        esac
    done

    log "== PoC claim runnable =="
    log "API: $API_URL  network: $NETWORK  session: $SOCIAL_SESSION_SAVE_PATH"
    health_checks
    run_claim_flow
    log "PASS: username provision + mock claim completed (chain events asserted in proof-of-creativity-runnable)"
}

main "$@"
