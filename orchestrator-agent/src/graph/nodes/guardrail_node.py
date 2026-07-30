"""Guardrail check node — runs before intent analysis to filter blocked queries."""

from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from src.schemas.state import OrchestratorState
from src.llm.client import get_chat_models
from src.exceptions import GuardrailError
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_BLOCKED_MESSAGE = "I'm sorry, but your request cannot be processed due to content policy."


def _is_hard_block(assessments: list) -> bool:
    """Return True if any assessment contains a BLOCKED action (not PII ANONYMIZED)."""
    for assessment in assessments:
        topic_policy = assessment.get("topicPolicy", {})
        for topic in topic_policy.get("topics", []):
            if topic.get("action") == "BLOCKED":
                return True

        content_policy = assessment.get("contentPolicy", {})
        for f in content_policy.get("filters", []):
            if f.get("action") == "BLOCKED":
                return True

        word_policy = assessment.get("wordPolicy", {})
        for word in word_policy.get("customWords", []):
            if word.get("action") == "BLOCKED":
                return True
        for word in word_policy.get("managedWordLists", []):
            if word.get("action") == "BLOCKED":
                return True

    return False


def guardrail_check_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Pre-check node that runs AWS Bedrock Guardrails before intent analysis.

    - If AWS_BEDROCK_GUARDRAIL_ID is not configured: skips check (pass-through).
    - action NONE: pass through unchanged.
    - action INTERVENED + hard block: sets final_answer + guardrail_blocked=True, routes to END.
    - action INTERVENED + PII only: updates state.query with redacted text, continues to analyzer.
    """
    logger.info("→ guardrail_check")

    chat_models = get_chat_models()

    if not chat_models.settings.aws_bedrock_guardrail_id:
        logger.info("No guardrail configured — skipping guardrail check")
        return {"guardrail_blocked": False, "guardrail_action": "NONE"}

    try:
        response = chat_models.apply_guardrail(text=state.query, source="INPUT")
    except (ClientError, BotoCoreError) as e:
        logger.error("Guardrail check failed: %s", e)
        raise GuardrailError("Guardrail check failed")
    action = response.get("action", "NONE")
    logger.info("guardrail action: %s", action)

    if action == "NONE":
        logger.info("← guardrail_check: passed")
        return {"guardrail_blocked": False, "guardrail_action": "NONE"}

    # INTERVENED — distinguish hard block vs PII masking
    assessments = response.get("assessments", [])
    outputs = response.get("outputs", [])
    output_text = outputs[0].get("text", "") if outputs else ""

    if _is_hard_block(assessments):
        logger.info("← guardrail_check: BLOCKED")
        return {
            "guardrail_blocked": True,
            "guardrail_action": "BLOCKED",
            "final_answer": [output_text or _DEFAULT_BLOCKED_MESSAGE],
        }

    # PII masking — update query with redacted version
    logger.info("← guardrail_check: PII masked, continuing to analyzer")
    return {"guardrail_blocked": False, "guardrail_action": "PII_MASKED", "query": output_text or state.query}


def guardrail_router(state: OrchestratorState) -> str:
    """Routes to END if query was blocked, otherwise to the intent analyzer."""
    return "end" if state.guardrail_blocked else "analyzer"
