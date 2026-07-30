"""Intent analyzer node - LLM-based tool selection."""

from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from src.exceptions import LLMFailureError
from src.llm.client import get_chat_models
from src.llm.prompts.intent_analyzer import build_tool_selection_prompt, build_tools_context
from src.schemas.llm import ToolSelectionOutput
from src.schemas.registry import ToolDefinition
from src.schemas.state import OrchestratorState
from src.schemas.tools import SelectedTool
from src.utils.logging import get_logger
from src.utils.text_cleaner import clean_text

logger = get_logger(__name__)


def _extract_token_usage(raw_response: Any) -> tuple[int | None, int | None, int | None]:
    """Extract token usage from LangChain raw message metadata when available."""
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if usage_metadata is None and isinstance(raw_response, dict):
        usage_metadata = raw_response.get("usage_metadata")

    if not isinstance(usage_metadata, dict):
        return None, None, None

    return (
        usage_metadata.get("input_tokens"),
        usage_metadata.get("output_tokens"),
        usage_metadata.get("total_tokens"),
    )


def create_intent_node(registry: dict[str, ToolDefinition]):
    tools_context = build_tools_context(registry)
    system_prompt = build_tool_selection_prompt(tools_context)

    chat_models = get_chat_models()
    logger.debug(f"Available tools: {list(registry.keys())}")

    def _get_structured_llm():
        """Get a fresh structured LLM (picks up refreshed credentials)."""
        llm = chat_models.get_model()
        return llm.with_structured_output(ToolSelectionOutput, include_raw=True)

    def intent_node(state: OrchestratorState) -> dict[str, Any]:
        user_query = state.query
        logger.info("-> intent_analyzer")
        logger.info(f"Input query: {user_query}")
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None

        cleaned_query = clean_text(user_query)
        if not cleaned_query:
            logger.info("<- intent_analyzer: query empty after cleaning, returning NO_TOOL")
            return {
                "selected_tool": SelectedTool(
                    tool_name="NO_TOOL",
                    confidence=0.0,
                    reasoning="Query contained no meaningful content after text cleaning",
                    reformulated_query=None,
                ),
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }

        try:
            logger.debug("Invoking LLM for intent analysis")
            structured_llm = _get_structured_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=cleaned_query),
            ]
            llm_response = structured_llm.invoke(messages)
            raw_response = llm_response.get("raw")
            input_tokens, output_tokens, total_tokens = _extract_token_usage(raw_response)

            parsing_error = llm_response.get("parsing_error")
            if parsing_error is not None:
                raise OutputParserException(str(parsing_error))

            parsed: ToolSelectionOutput | None = llm_response.get("parsed")

            if parsed is None:
                logger.warning("LLM bypassed tool call (safety/content filter), returning NO_TOOL")
                return {
                    "selected_tool": SelectedTool(
                        tool_name="NO_TOOL",
                        confidence=0.0,
                        reasoning="LLM returned text instead of structured tool call",
                        reformulated_query=None,
                    ),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }

            logger.info(
                "<- intent_analyzer: tool=%s, confidence=%.1f",
                parsed.selected_tool,
                parsed.confidence_score,
            )
            logger.debug(f"Reformulated query: {parsed.reformulated_query!r}")

            return {
                "selected_tool": SelectedTool(
                    tool_name=parsed.selected_tool,
                    confidence=parsed.confidence_score,
                    reasoning=parsed.reasoning,
                    reformulated_query=parsed.reformulated_query,
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        except (ClientError, BotoCoreError) as e:
            logger.error("LLM call failed: %s", e)
            raise LLMFailureError("Intent analysis failed")
        except OutputParserException as e:
            logger.warning("LLM bypassed tool call (safety/content filter): %s", e)
            return {
                "selected_tool": SelectedTool(
                    tool_name="NO_TOOL",
                    confidence=0.0,
                    reasoning=f"LLM did not produce structured output: {e}",
                    reformulated_query=None,
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

    return intent_node
