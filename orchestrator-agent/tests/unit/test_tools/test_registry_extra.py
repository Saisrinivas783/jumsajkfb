"""Additional tests for tool registry to cover missing lines."""

import pytest
from unittest.mock import patch, MagicMock

from src.tools.registry import load_tools_from_yaml, load_tools_from_s3
from src.exceptions import ToolRegistryError


class TestLoadToolsFromYamlOSError:
    """Tests for OSError path in load_tools_from_yaml (lines 192-194)."""

    def test_oserror_raises_tool_registry_error(self, tmp_path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "tools:\n"
            "  - name: TestTool\n"
            "    description: Test\n"
            "    endpoint: https://test.internal/api/v1\n"
            "    capabilities: [testing]\n"
            "    parameters:\n"
            "      required: [userPrompt]\n"
            "      optional: []\n"
        )
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(ToolRegistryError) as exc_info:
                load_tools_from_yaml(yaml_file)
        assert "Unable to read" in str(exc_info.value)

    def test_oserror_message_is_correct(self, tmp_path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text("tools: []")
        with patch("builtins.open", side_effect=OSError("No permission")):
            with pytest.raises(ToolRegistryError) as exc_info:
                load_tools_from_yaml(yaml_file)
        assert isinstance(exc_info.value, ToolRegistryError)


class TestLoadToolsFromS3:
    """Tests for error paths in load_tools_from_s3."""

    def test_client_error_raises_tool_registry_error(self):
        """Lines 243-248: ClientError path."""
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
            "GetObject",
        )

        with patch("boto3.client", return_value=mock_s3):
            with pytest.raises(ToolRegistryError) as exc_info:
                load_tools_from_s3("my-bucket", "config/tools.yaml")
        assert "Unable to load" in str(exc_info.value)

    def test_boto_core_error_raises_tool_registry_error(self):
        """Lines 249-253: BotoCoreError path."""
        from botocore.exceptions import BotoCoreError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = BotoCoreError()

        with patch("boto3.client", return_value=mock_s3):
            with pytest.raises(ToolRegistryError) as exc_info:
                load_tools_from_s3("my-bucket", "config/tools.yaml")
        assert "Unable to load" in str(exc_info.value)

    def test_yaml_error_from_s3_content_raises_tool_registry_error(self):
        """Lines 257-261: yaml.YAMLError from S3 content."""
        import yaml

        mock_body = MagicMock()
        mock_body.read.return_value = b"valid: yaml: content"
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}

        with patch("boto3.client", return_value=mock_s3):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("bad yaml")):
                with pytest.raises(ToolRegistryError) as exc_info:
                    load_tools_from_s3("my-bucket", "config/tools.yaml")
        assert "Invalid tool configuration file format" in str(exc_info.value)

    def test_client_error_with_access_denied(self):
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
            "GetObject",
        )

        with patch("boto3.client", return_value=mock_s3):
            with pytest.raises(ToolRegistryError):
                load_tools_from_s3("my-bucket", "config/tools.yaml", region_name="us-east-1")
