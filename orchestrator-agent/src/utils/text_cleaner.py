"""Text cleaning pipeline for sanitizing user queries before LLM invocation."""

import html
import re
from typing import Callable

from src.utils.logging import get_logger

logger = get_logger(__name__)

PIPELINE: list[Callable[[str], str]] = []


def _decode_html_entities(text: str) -> str:
    """Remove HTML entities and apostrophe characters."""
    if not text:
        return text
    
    # Remove apostrophe characters (the actual issue)
    text = text.replace("'", " ")  # Remove apostrophe characters
    
    # Remove HTML entities (if any exist)
    text = text.replace('&#39;', ' ')  # Remove &#39;
    text = text.replace('&#34;', ' ')  # Remove &#34;
    text = text.replace('&#38;', ' ')  # Remove &#38;
    text = text.replace('&#60;', ' ')  # Remove &#60;
    text = text.replace('&#62;', ' ')  # Remove &#62;
    text = text.replace('&amp;', ' ')   # Remove &amp;
    text = text.replace('&lt;', ' ')    # Remove &lt;
    text = text.replace('&gt;', ' ')    # Remove &gt;
    text = text.replace('&quot;', ' ')  # Remove &quot;
    text = text.replace('&nbsp;', ' ')  # Remove &nbsp;
    
    # Then handle any remaining patterns with regex
    text = re.sub(r'&[a-zA-Z][a-zA-Z0-9]*;', ' ', text)  # Named entities
    text = re.sub(r'&#[0-9]+;', ' ', text)  # Numeric entities
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)  # Hex entities
    
    return text


def _remove_special_chars(text: str) -> str:
    """
    Enhanced special character removal that preserves healthcare symbols
    and handles consecutive special characters intelligently.
    """
    if not text:
        return text
    
    # Healthcare symbols to preserve
    healthcare_chars = r'%$°±×÷≤≥<>=\-'
    
    # Basic punctuation to preserve  
    basic_punct = r'.,;:!?\'"()\[\]{}/_@'
    
    # Pattern for characters to keep
    keep_pattern = f'[a-zA-Z0-9\\s{re.escape(healthcare_chars)}{re.escape(basic_punct)}]'
    
    # Replace consecutive special characters with single space
    result = []
    prev_was_special = False
    
    for char in text:
        if re.match(keep_pattern, char):
            result.append(char)
            prev_was_special = False
        else:
            # This is a special character to remove
            if not prev_was_special:
                result.append(' ')
            prev_was_special = True
    
    return ''.join(result)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


PIPELINE.append(_decode_html_entities)
PIPELINE.append(_remove_special_chars)
PIPELINE.append(_collapse_whitespace)

# Log pipeline initialization
logger.info(f"Text cleaning pipeline initialized with {len(PIPELINE)} steps: {[fn.__name__ for fn in PIPELINE]}")


def clean_text(text: str) -> str:
    for fn in PIPELINE:
        text = fn(text)
    return text