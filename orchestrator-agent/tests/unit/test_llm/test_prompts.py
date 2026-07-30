"""Unit tests for LLM prompt builders."""

import pytest
from unittest.mock import MagicMock

from src.llm.prompts.intent_analyzer import build_tool_selection_prompt, build_tools_context


class TestBuildToolsContext:
    """Tests for build_tools_context."""

    def test_empty_registry_returns_no_tools_message(self):
        """Line 32: 'No tools available' path."""
        result = build_tools_context({})
        assert result == "No tools available"

    def test_single_tool_included_in_output(self):
        tool = MagicMock()
        tool.description = "Handles benefit inquiries"
        tool.endpoint = "https://ibt.internal/api/v1"
        tool.capabilities = ["benefit inquiries"]
        tool.parameters.required = ["userPrompt"]
        tool.parameters.optional = []

        result = build_tools_context({"IBTAgent": tool})

        assert "IBTAgent" in result
        assert "Handles benefit inquiries" in result

    def test_tool_with_optional_params_included(self):
        tool = MagicMock()
        tool.description = "Claims tool"
        tool.endpoint = "https://claims.internal/api/v1"
        tool.capabilities = ["claim status"]
        tool.parameters.required = ["userPrompt"]
        tool.parameters.optional = ["claimNumber"]

        result = build_tools_context({"ClaimsAgent": tool})

        assert "claimNumber" in result

    def test_tool_with_no_optional_params_shows_none(self):
        tool = MagicMock()
        tool.description = "Test tool"
        tool.endpoint = "https://test.internal/api/v1"
        tool.capabilities = ["testing"]
        tool.parameters.required = ["userPrompt"]
        tool.parameters.optional = []

        result = build_tools_context({"TestTool": tool})

        assert "None" in result

    def test_tool_with_examples_included(self):
        tool = MagicMock()
        tool.description = "Test tool"
        tool.endpoint = "https://test.internal/api/v1"
        tool.capabilities = ["testing"]
        tool.parameters.required = ["userPrompt"]
        tool.parameters.optional = []
        
        example1 = MagicMock()
        example1.prompt = "user prompt 1"
        example1.reasoning = "reasoning 1"
        tool.examples = [example1]

        result = build_tools_context({"TestTool": tool})

        assert "Examples:" in result
        assert "user prompt 1" in result
        assert "reasoning 1" in result

    def test_multiple_tools_all_included(self):
        tool1 = MagicMock()
        tool1.description = "Tool one"
        tool1.endpoint = "https://one.internal"
        tool1.capabilities = ["cap1"]
        tool1.parameters.required = ["userPrompt"]
        tool1.parameters.optional = []

        tool2 = MagicMock()
        tool2.description = "Tool two"
        tool2.endpoint = "https://two.internal"
        tool2.capabilities = ["cap2"]
        tool2.parameters.required = ["userPrompt"]
        tool2.parameters.optional = []

        result = build_tools_context({"ToolOne": tool1, "ToolTwo": tool2})

        assert "ToolOne" in result
        assert "ToolTwo" in result


class TestBuildToolSelectionPrompt:
    """Tests for build_tool_selection_prompt."""

    def test_returns_string(self):
        result = build_tool_selection_prompt("some tools context")
        assert isinstance(result, str)

    def test_includes_tools_context(self):
        context = "IBTAgent: Handles insurance benefits"
        result = build_tool_selection_prompt(context)
        assert context in result

    def test_prompt_mentions_no_tool(self):
        result = build_tool_selection_prompt("tools here")
        assert "NO_TOOL" in result

    def test_prompt_is_non_empty(self):
        result = build_tool_selection_prompt("")
        assert len(result) > 50
