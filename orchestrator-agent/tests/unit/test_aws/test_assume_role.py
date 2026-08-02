"""Unit tests for the shared AssumedRoleClientFactory (STS AssumeRole + retry).

Covers the retry/backoff on ``refresh_metadata()`` — the one network call this
module makes, shared by every AWS-backed client in this agent — and the
session/client caching behavior.
"""

from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from src.aws.assume_role import AssumedRoleClientFactory
from src.exceptions import CredentialsError


def _factory(**overrides):
    values = dict(
        role_arn="arn:aws:iam::123456789012:role/test-role",
        session_name="test-session",
        duration_seconds=3600,
        region_name="us-east-1",
        method="test-assume-role",
    )
    values.update(overrides)
    return AssumedRoleClientFactory(**values)


def _sts_response(expiration="2099-01-01T00:00:00Z"):
    return {
        "Credentials": {
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": "secret",
            "SessionToken": "token",
            "Expiration": expiration,
        }
    }


def _throttling_error():
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "AssumeRole",
    )


class TestRefreshMetadata:
    @patch("src.aws.assume_role.boto3.client")
    def test_succeeds_on_first_attempt(self, mock_boto_client):
        mock_sts = Mock()
        mock_sts.assume_role.return_value = _sts_response()
        mock_boto_client.return_value = mock_sts

        metadata = _factory().refresh_metadata()

        assert mock_sts.assume_role.call_count == 1
        assert metadata["access_key"] == "ASIAEXAMPLE"
        assert metadata["secret_key"] == "secret"
        assert metadata["token"] == "token"

    @patch("time.sleep", return_value=None)
    @patch("src.aws.assume_role.boto3.client")
    def test_retries_transient_failure_and_recovers(
        self, mock_boto_client, _mock_sleep
    ):
        """A throttled STS call twice, then succeeding, should still return credentials."""
        mock_sts = Mock()
        mock_sts.assume_role.side_effect = [
            _throttling_error(),
            _throttling_error(),
            _sts_response(),
        ]
        mock_boto_client.return_value = mock_sts

        metadata = _factory().refresh_metadata()

        assert mock_sts.assume_role.call_count == 3
        assert metadata["access_key"] == "ASIAEXAMPLE"

    @patch("time.sleep", return_value=None)
    @patch("src.aws.assume_role.boto3.client")
    def test_gives_up_after_exhausting_retries(self, mock_boto_client, _mock_sleep):
        """A persistently failing STS call exhausts the 3 attempts and raises
        CredentialsError so callers can distinguish an auth outage from a bug."""
        mock_sts = Mock()
        mock_sts.assume_role.side_effect = _throttling_error()
        mock_boto_client.return_value = mock_sts

        with pytest.raises(CredentialsError, match="AssumeRole failed"):
            _factory().refresh_metadata()

        assert mock_sts.assume_role.call_count == 3

    @patch("time.sleep", return_value=None)
    @patch("src.aws.assume_role.boto3.client")
    def test_retries_on_unexpected_exception_too(self, mock_boto_client, _mock_sleep):
        """Non-ClientError failures (e.g. connection errors) are retried as well."""
        mock_sts = Mock()
        mock_sts.assume_role.side_effect = [
            ConnectionError("network blip"),
            _sts_response(),
        ]
        mock_boto_client.return_value = mock_sts

        metadata = _factory().refresh_metadata()

        assert mock_sts.assume_role.call_count == 2
        assert metadata["access_key"] == "ASIAEXAMPLE"


class TestSessionCaching:
    @patch("src.aws.assume_role.boto3.client")
    def test_session_is_built_once_and_reused(self, mock_boto_client):
        mock_sts = Mock()
        mock_sts.assume_role.return_value = _sts_response()
        mock_boto_client.return_value = mock_sts

        factory = _factory()
        session1 = factory.session()
        session2 = factory.session()

        assert session1 is session2
        # refresh_metadata only runs once, during the initial session build.
        assert mock_sts.assume_role.call_count == 1

    @patch("src.aws.assume_role.boto3.client")
    def test_client_delegates_to_cached_session(self, mock_boto_client):
        mock_sts = Mock()
        mock_sts.assume_role.return_value = _sts_response()
        mock_boto_client.return_value = mock_sts

        client = _factory().client("sts")

        assert client is not None
        assert mock_sts.assume_role.call_count == 1


class TestCredentialRefreshWorker:
    def test_start_is_idempotent_single_daemon_thread(self):
        from src.aws.assume_role import CredentialRefreshWorker

        factory = Mock()
        with patch("src.aws.assume_role.threading.Thread") as mock_thread_cls:
            mock_thread = Mock()
            mock_thread.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread

            worker = CredentialRefreshWorker(factory, name="test-worker")
            worker.start()
            worker.start()

            mock_thread_cls.assert_called_once()
            assert mock_thread_cls.call_args.kwargs["daemon"] is True
            mock_thread.start.assert_called_once()

    def test_warm_touches_factory_credentials(self):
        from src.aws.assume_role import CredentialRefreshWorker

        factory = Mock()
        worker = CredentialRefreshWorker(factory, name="test-worker")

        worker.warm()

        factory.session.return_value.get_credentials.return_value.get_frozen_credentials.assert_called_once()

    def test_stop_without_start_is_a_no_op(self):
        from src.aws.assume_role import CredentialRefreshWorker

        worker = CredentialRefreshWorker(Mock(), name="test-worker")
        worker.stop()  # should not raise
