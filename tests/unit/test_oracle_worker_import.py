"""Oracle worker construction smoke tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_oracle_deps():
    with (
        patch("app.workers.oracle_worker.JobRepository") as jobs,
        patch("app.workers.oracle_worker.ChainPostRepository") as posts,
        patch("app.workers.oracle_worker.AttestationRepository") as attestations,
        patch("app.workers.oracle_worker.ConfigCacheRepository") as config_cache,
        patch("app.workers.oracle_worker.AnalysisService") as analysis,
        patch("app.workers.oracle_worker.DecisionEngine") as decisions,
        patch("app.workers.oracle_worker.ProofBundleService") as proofs,
        patch("app.workers.oracle_worker.UsernameBeneficiaryService") as beneficiaries,
        patch("app.workers.oracle_worker.VaultLifecycleService") as vault_lifecycle,
        patch("app.workers.oracle_worker.TransactionSubmitter") as submitter,
        patch("app.workers.oracle_worker.load_network_profile") as load_profile,
        patch("app.workers.oracle_worker.get_settings") as get_settings,
    ):
        load_profile.return_value = MagicMock(network="localnet")
        get_settings.return_value = MagicMock(oracle_worker_concurrency=1)
        yield {
            "jobs": jobs,
            "posts": posts,
            "attestations": attestations,
            "config_cache": config_cache,
            "analysis": analysis,
            "decisions": decisions,
            "proofs": proofs,
            "beneficiaries": beneficiaries,
            "vault_lifecycle": vault_lifecycle,
            "submitter": submitter,
        }


def test_oracle_worker_instantiates(mock_oracle_deps):
    from app.workers.oracle_worker import OracleWorker

    worker = OracleWorker("localnet")
    assert worker.network == "localnet"
    mock_oracle_deps["beneficiaries"].assert_called_once_with("localnet")
