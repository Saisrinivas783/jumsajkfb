"""LLM structured output schemas."""

from pydantic import BaseModel, Field


class ToolSelectionOutput(BaseModel):
    """Structured output format for LLM tool selection."""

    selected_tool: str = Field(
        description="Name of the selected tool (e.g., 'IBTAgent', 'ClaimsAgent') or 'NO_TOOL' if no suitable tool"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Confidence score between 0 and 10"
    )
    reasoning: str = Field(
        description="Detailed explanation for selection"
    )
    reformulated_query: str = Field(
        description="Spell-corrected. Keep only medical terms, conditions, procedures, ages, subjects (child/adult), body parts. If a keyword is an abbreviation, always append its full expanded form. Return as a space-separated string."
    )