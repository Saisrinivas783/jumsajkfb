"""Unit tests for ToolRegistry."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.tools.registry import (
    ToolRegistry,
    load_tools_from_yaml,
    load_tools_from_s3,
    _parse_tools_yaml
)
from src.exceptions import ToolRegistryError
from src.schemas.registry import ToolDefinition, ToolParameters


@pytest.fixture
def sample_tool_defs():
    """Create sample tool definitions."""
    return [
        ToolDefinition(
            name="IBTAgent",
            description="Handles insurance benefit inquiries",
            endpoint="https://ibt-service.internal/api/v1",
            capabilities=["benefit inquiries", "coverage questions"],
            parameters=ToolParameters(required=["userPrompt", "userName"], optional=["policyNumber"])
        ),
        ToolDefinition(
            name="ClaimsAgent",
            description="Handles claims inquiries",
            endpoint="https://claims-service.internal/api/v1",
            capabilities=["claim status", "claim submission"],
            parameters=ToolParameters(required=["userPrompt"], optional=["claimNumber"])
        )
    ]


@pytest.fixture
def valid_yaml_content():
    """Create valid YAML content for testing."""
    return {
        "tools": [
            {
                "name": "IBTAgent",
                "description": "Handles insurance benefit inquiries",
                "endpoint": "https://ibt-service.internal/api/v1",
                "capabilities": ["benefit inquiries", "coverage questions"],
                "parameters": {
                    "required": ["userPrompt", "userName"],
                    "optional": ["policyNumber"]
                }
            },
            {
                "name": "ClaimsAgent",
                "description": "Handles claims inquiries",
                "endpoint": "https://claims-service.internal/api/v1",
                "capabilities": ["claim status", "claim submission"],
                "parameters": {
                    "required": ["userPrompt"],
                    "optional": ["claimNumber"]
                }
            }
        ]
    }


class TestToolRegistryInit:
    """Tests for ToolRegistry initialization."""

    def test_registry_init_with_tools(self, sample_tool_defs):
        """Test registry initialization with tools."""
        registry = ToolRegistry(sample_tool_defs)

        assert len(registry) == 2
        assert "IBTAgent" in registry
        assert "ClaimsAgent" in registry

    def test_registry_init_empty(self):
        """Test registry initialization with empty list."""
        registry = ToolRegistry([])

        assert len(registry) == 0
        assert registry.list_tool_names() == []

    def test_registry_init_invalid_type(self):
        """Test registry initialization with invalid type."""
        with pytest.raises(TypeError):
            ToolRegistry("not a list")

    def test_registry_tools_property(self, sample_tool_defs):
        """Test tools property returns copy."""
        registry = ToolRegistry(sample_tool_defs)

        tools = registry.tools
        assert len(tools) == 2

        # Modifying returned list shouldn't affect registry
        tools.append(MagicMock())
        assert len(registry.tools) == 2


class TestToolRegistryMethods:
    """Tests for ToolRegistry methods."""

    def test_get_by_name_exists(self, sample_tool_defs):
        """Test getting tool by name when it exists."""
        registry = ToolRegistry(sample_tool_defs)

        tool = registry.get_by_name("IBTAgent")
        assert tool is not None
        assert tool.name == "IBTAgent"

    def test_get_by_name_not_exists(self, sample_tool_defs):
        """Test getting tool by name when it doesn't exist."""
        registry = ToolRegistry(sample_tool_defs)

        tool = registry.get_by_name("NonExistentTool")
        assert tool is None

    def test_list_tools(self, sample_tool_defs):
        """Test listing all tools."""
        registry = ToolRegistry(sample_tool_defs)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_list_tool_names(self, sample_tool_defs):
        """Test listing tool names."""
        registry = ToolRegistry(sample_tool_defs)

        names = registry.list_tool_names()
        assert names == ["IBTAgent", "ClaimsAgent"]

    def test_list_capabilities(self, sample_tool_defs):
        """Test listing capabilities."""
        registry = ToolRegistry(sample_tool_defs)

        capabilities = registry.list_capabilities()
        assert "IBTAgent" in capabilities
        assert "ClaimsAgent" in capabilities
        assert "benefit inquiries" in capabilities["IBTAgent"]
        assert "claim status" in capabilities["ClaimsAgent"]

    def test_len(self, sample_tool_defs):
        """Test __len__ method."""
        registry = ToolRegistry(sample_tool_defs)
        assert len(registry) == 2

    def test_iter(self, sample_tool_defs):
        """Test __iter__ method."""
        registry = ToolRegistry(sample_tool_defs)

        tools = list(registry)
        assert len(tools) == 2
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_contains(self, sample_tool_defs):
        """Test __contains__ method."""
        registry = ToolRegistry(sample_tool_defs)

        assert "IBTAgent" in registry
        assert "ClaimsAgent" in registry
        assert "NonExistent" not in registry

    def test_repr(self, sample_tool_defs):
        """Test __repr__ method."""
        registry = ToolRegistry(sample_tool_defs)

        repr_str = repr(registry)
        assert "ToolRegistry" in repr_str
        assert "IBTAgent" in repr_str
        assert "ClaimsAgent" in repr_str


class TestLoadFromYaml:
    """Tests for loading tools from YAML."""

    def test_load_from_yaml_success(self, tmp_path, valid_yaml_content):
        """Test successful YAML loading."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(valid_yaml_content, f)

        tools_dict = load_tools_from_yaml(str(yaml_file))

        assert len(tools_dict) == 2
        assert "IBTAgent" in tools_dict
        assert "ClaimsAgent" in tools_dict

    def test_load_from_yaml_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(ToolRegistryError) as exc_info:
            load_tools_from_yaml("/nonexistent/path/tools.yaml")

        assert "not found" in str(exc_info.value)

    def test_load_from_yaml_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ToolRegistryError) as exc_info:
            load_tools_from_yaml(str(yaml_file))

        assert "Invalid tool configuration" in str(exc_info.value)

    def test_load_from_yaml_missing_tools_key(self, tmp_path):
        """Test loading YAML without 'tools' key."""
        yaml_file = tmp_path / "no_tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump({"something_else": []}, f)

        with pytest.raises(ToolRegistryError) as exc_info:
            load_tools_from_yaml(str(yaml_file))

        assert "tools" in str(exc_info.value).lower()

    def test_from_local_yaml_classmethod(self, tmp_path, valid_yaml_content):
        """Test from_local_yaml class method."""
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(valid_yaml_content, f)

        registry = ToolRegistry.from_local_yaml(str(yaml_file))

        assert isinstance(registry, ToolRegistry)
        assert len(registry) == 2


class TestParseToolsYaml:
    """Tests for _parse_tools_yaml function."""

    def test_parse_tools_yaml_success(self, valid_yaml_content):
        """Test successful YAML parsing."""
        tools = _parse_tools_yaml(valid_yaml_content)

        assert len(tools) == 2
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_parse_tools_yaml_not_dict(self):
        """Test parsing non-dict content."""
        with pytest.raises(ToolRegistryError) as exc_info:
            _parse_tools_yaml([])

        assert "Expected dict" in str(exc_info.value)

    def test_parse_tools_yaml_no_tools_key(self):
        """Test parsing without 'tools' key."""
        with pytest.raises(ToolRegistryError) as exc_info:
            _parse_tools_yaml({"other": []})

        assert "tools" in str(exc_info.value).lower()

    def test_parse_tools_yaml_tools_not_list(self):
        """Test parsing with non-list tools."""
        with pytest.raises(ToolRegistryError) as exc_info:
            _parse_tools_yaml({"tools": "not a list"})

        assert "list" in str(exc_info.value).lower()

    def test_parse_tools_yaml_invalid_tool_definition(self):
        """Test parsing with invalid tool definition."""
        invalid_content = {
            "tools": [
                {
                    "name": "InvalidTool",
                    # Missing required fields
                }
            ]
        }

        with pytest.raises(ToolRegistryError) as exc_info:
            _parse_tools_yaml(invalid_content)

        assert "Invalid tool definition" in str(exc_info.value)


class TestLoadFromS3:
    """Tests for loading tools from S3."""

    @patch('boto3.client')
    def test_load_from_s3_success(self, mock_boto3_client, valid_yaml_content):
        """Test successful S3 loading."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_response = {
            "Body": MagicMock()
        }
        yaml_content = yaml.dump(valid_yaml_content)
        mock_response["Body"].read.return_value.decode.return_value = yaml_content
        mock_s3.get_object.return_value = mock_response
        mock_boto3_client.return_value = mock_s3

        tools_dict = load_tools_from_s3("test-bucket", "config/tools.yaml")

        assert len(tools_dict) == 2
        assert "IBTAgent" in tools_dict

    @patch('boto3.client')
    def test_from_s3_classmethod(self, mock_boto3_client, valid_yaml_content):
        """Test from_s3 class method."""
        mock_s3 = MagicMock()
        mock_response = {
            "Body": MagicMock()
        }
        yaml_content = yaml.dump(valid_yaml_content)
        mock_response["Body"].read.return_value.decode.return_value = yaml_content
        mock_s3.get_object.return_value = mock_response
        mock_boto3_client.return_value = mock_s3

        registry = ToolRegistry.from_s3("test-bucket", "config/tools.yaml")

        assert isinstance(registry, ToolRegistry)
        assert len(registry) == 2


class TestToolRegistryIntegration:
    """Integration tests for ToolRegistry."""

    def test_full_workflow_from_yaml(self, tmp_path):
        """Test complete workflow from YAML to registry usage."""
        # Create YAML file
        yaml_content = {
            "tools": [
                {
                    "name": "TestTool",
                    "description": "Test tool description",
                    "endpoint": "https://test.internal/api/v1",
                    "capabilities": ["testing"],
                    "parameters": {
                        "required": ["userPrompt"],
                        "optional": []
                    }
                }
            ]
        }
        yaml_file = tmp_path / "tools.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)

        # Load registry
        registry = ToolRegistry.from_local_yaml(str(yaml_file))

        # Use registry
        assert len(registry) == 1
        assert "TestTool" in registry

        tool = registry.get_by_name("TestTool")
        assert tool.name == "TestTool"
        assert "testing" in tool.capabilities
