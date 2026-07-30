"""Generic response messages per FEPOC specification."""

# Fallback message when search returns no results
_FALLBACK_MESSAGE = [
    "Your search did not return any results. Please refer to the "
    "<a href='https://www.fepblue.org/plan-brochures' target='_blank'>Blue Cross and Blue Shield Service Benefit Plan brochure</a> "
    "or to the "
    "<a href='https://www.fepblue.org/plan-summaries' target='_blank'>Health Plan Summaries</a>."
]

# Generic response messages as specified in requirements
GENERIC_MESSAGES = {
    "no_results_found": _FALLBACK_MESSAGE,
    "service_unavailable": [
        "I'm currently experiencing technical difficulties accessing benefit information. Please try again in a few moments."
    ],
    "query_limit_exceeded": [
        "You have reached the maximum number of search requests allowed. Please wait a moment before trying again."
    ]
}

def get_message(key: str):
    """Get generic message by key."""
    return GENERIC_MESSAGES.get(key, GENERIC_MESSAGES["service_unavailable"])
