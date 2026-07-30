"""Tool Registry for managing and loading tool definitions.

This module provides functionality to load tool configurations from
local YAML files or S3 buckets and manage them through a registry interface.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.exceptions import ToolRegistryError
from src.schemas.registry import ToolDefinition
from src.utils.logging import get_logger

__all__ = [
    "ToolRegistry",
    "ToolRegistryError",
    "load_tools_from_yaml",
    "load_tools_from_s3",
]

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing tool definitions.

    Provides lookup, listing, and capability querying for registered tools.

    Attributes:
        tools: List of all registered tool definitions.

    Example:
        registry = ToolRegistry.from_local_yaml("config/tools.yaml")
        tool = registry.get_by_name("IBTAgent")
        if tool:
            print(tool.capabilities)
    """

    def __init__(self, tools: list[ToolDefinition]) -> None:
        if not isinstance(tools, list):
            raise TypeError(f"Expected list of tools, got {type(tools).__name__}")

        self._tools = tools
        self._by_name: dict[str, ToolDefinition] = {tool.name: tool for tool in tools}
        logger.debug("Initialized ToolRegistry with %d tools", len(tools))

    @property
    def tools(self) -> list[ToolDefinition]:
        """Return a copy of the tools list to prevent external mutation."""
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __repr__(self) -> str:
        tool_names = [t.name for t in self._tools]
        return f"ToolRegistry(tools={tool_names})"

    def get_by_name(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name.

        Args:
            name: The tool name to look up.

        Returns:
            The ToolDefinition if found, None otherwise.
        """
        return self._by_name.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tools."""
        return self.tools

    def list_tool_names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._by_name.keys())

    def list_capabilities(self) -> dict[str, list[str]]:
        """Return a mapping of tool names to their capabilities."""
        return {tool.name: tool.capabilities for tool in self._tools}

    @classmethod
    def from_local_yaml(cls, path: str | Path) -> ToolRegistry:
        """Create a ToolRegistry from a local YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A new ToolRegistry instance.

        Raises:
            ToolRegistryError: If the file cannot be loaded or parsed.
        """
        tools_dict = load_tools_from_yaml(path)
        return cls(list(tools_dict.values()))

    @classmethod
    def from_s3(
        cls,
        bucket: str,
        key: str,
        region_name: str | None = None,
    ) -> ToolRegistry:
        """Create a ToolRegistry from a YAML file in S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key (path to YAML file).
            region_name: AWS region name (optional).

        Returns:
            A new ToolRegistry instance.

        Raises:
            ToolRegistryError: If the S3 object cannot be loaded or parsed.
        """
        tools_dict = load_tools_from_s3(bucket, key, region_name=region_name)
        return cls(list(tools_dict.values()))


def _parse_tools_yaml(raw: dict) -> list[ToolDefinition]:
    """Parse raw YAML dict into ToolDefinition list.

    Args:
        raw: Parsed YAML dictionary.

    Returns:
        List of ToolDefinition objects.

    Raises:
        ToolRegistryError: If YAML structure is invalid.
    """
    if not isinstance(raw, dict):
        raise ToolRegistryError(f"Expected dict, got {type(raw).__name__}")

    if "tools" not in raw:
        raise ToolRegistryError("YAML must contain a top-level 'tools' key")

    tools_data = raw["tools"]
    if not isinstance(tools_data, list):
        raise ToolRegistryError(
            f"'tools' must be a list, got {type(tools_data).__name__}"
        )

    if not tools_data:
        raise ToolRegistryError("'tools' list cannot be empty")

    tools = []
    for idx, tool_data in enumerate(tools_data):
        try:
            tools.append(ToolDefinition(**tool_data))
        except Exception as e:
            logger.error(f"Failed to parse tool at index {idx}: {e}", exc_info=True)
            raise ToolRegistryError(
                f"Invalid tool definition at index {idx}"
            ) from e

    return tools


def load_tools_from_yaml(path: str | Path) -> dict[str, ToolDefinition]:
    """Load tool definitions from a local YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dict mapping tool names to ToolDefinition objects.

    Raises:
        ToolRegistryError: If the file cannot be found, read, or parsed.
    """
    path = Path(path)
    logger.debug(f"Loading tools from: {path}")

    if not path.exists():
        raise ToolRegistryError(f"Tool config not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file {path}: {e}", exc_info=True)
        raise ToolRegistryError("Invalid tool configuration file format") from e
    except OSError as e:
        logger.error(f"Failed to read file {path}: {e}", exc_info=True)
        raise ToolRegistryError("Unable to read tool configuration file") from e

    tools = _parse_tools_yaml(raw)
    tool_names = [t.name for t in tools]
    logger.info(f"Tool registry loaded: {len(tools)} tools")
    logger.debug(f"Available tools: {tool_names}")

    return {tool.name: tool for tool in tools}


def load_tools_from_s3(
    bucket: str,
    key: str,
    region_name: str | None = None,
) -> dict[str, ToolDefinition]:
    """Load tool definitions from a YAML file in S3.

    Args:
        bucket: S3 bucket name.
        key: S3 object key (path to YAML file).
        region_name: AWS region name (optional, uses default if not specified).

    Returns:
        Dict mapping tool names to ToolDefinition objects.

    Raises:
        ToolRegistryError: If the S3 object cannot be loaded or parsed.

    Example:
        tools = load_tools_from_s3(
            bucket="my-config-bucket",
            key="config/tools.yaml",
            region_name="us-east-1"
        )
    """
    logger.debug(f"Loading tools from S3: s3://{bucket}/{key}")

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise ToolRegistryError(
            "boto3 is required for S3 loading. Install with: pip install boto3"
        ) from e

    try:
        s3_client = boto3.client("s3", region_name=region_name)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(f"S3 error ({error_code}) loading s3://{bucket}/{key}: {e}", exc_info=True)
        raise ToolRegistryError(
            "Unable to load tool configuration from S3"
        ) from e
    except BotoCoreError as e:
        logger.error(f"AWS error loading s3://{bucket}/{key}: {e}", exc_info=True)
        raise ToolRegistryError(
            "Unable to load tool configuration from S3"
        ) from e

    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML from s3://{bucket}/{key}: {e}", exc_info=True)
        raise ToolRegistryError(
            "Invalid tool configuration file format in S3"
        ) from e

    tools = _parse_tools_yaml(raw)
    tool_names = [t.name for t in tools]
    logger.info(f"Tool registry loaded from S3: {len(tools)} tools")
    logger.debug(f"Available tools: {tool_names}")

    return {tool.name: tool for tool in tools}
