"""Unit tests for messages configuration."""

import pytest
from src.config.messages import GENERIC_MESSAGES, _FALLBACK_MESSAGE, get_message

class TestMessages:
    """Tests for message utilities."""
    
    def test_generic_messages_exist(self):
        """Test that required generic messages are defined."""
        assert "no_results_found" in GENERIC_MESSAGES
        assert "service_unavailable" in GENERIC_MESSAGES
        assert "query_limit_exceeded" in GENERIC_MESSAGES
        
        # Verify no_results_found is the FEPOC fallback message list
        assert isinstance(GENERIC_MESSAGES["no_results_found"], list)
        assert "Your search did not return any results" in GENERIC_MESSAGES["no_results_found"][0]
        assert "technical difficulties" in GENERIC_MESSAGES["service_unavailable"][0]
    
    def test_get_message_valid_key(self):
        """Test get_message with valid keys."""
        no_results_msg = get_message("no_results_found")
        service_unavailable_msg = get_message("service_unavailable")
        query_limit_msg = get_message("query_limit_exceeded")
        
        assert no_results_msg == GENERIC_MESSAGES["no_results_found"]
        assert service_unavailable_msg == GENERIC_MESSAGES["service_unavailable"]
        assert query_limit_msg == GENERIC_MESSAGES["query_limit_exceeded"]
    
    def test_get_message_invalid_key(self):
        """Test get_message with invalid key returns default."""
        invalid_msg = get_message("invalid_key")
        default_msg = get_message("nonexistent")
        
        # Should return service_unavailable as default
        assert invalid_msg == GENERIC_MESSAGES["service_unavailable"]
        assert default_msg == GENERIC_MESSAGES["service_unavailable"]
    
    def test_get_message_empty_key(self):
        """Test get_message with empty key."""
        empty_msg = get_message("")
        none_msg = get_message(None)
        
        assert empty_msg == GENERIC_MESSAGES["service_unavailable"]
        assert none_msg == GENERIC_MESSAGES["service_unavailable"]
    
    def test_message_content_requirements(self):
        """Test that messages meet FEPOC specification requirements."""
        no_results = GENERIC_MESSAGES["no_results_found"]
        service_unavailable = GENERIC_MESSAGES["service_unavailable"]
        query_limit = GENERIC_MESSAGES["query_limit_exceeded"]
        
        # no_results_found is a list with brochure/summary links
        assert isinstance(no_results, list)
        assert len(no_results) == 1
        assert "fepblue.org/plan-brochures" in no_results[0]
        assert "fepblue.org/plan-summaries" in no_results[0]
        # service_unavailable is a list
        assert isinstance(service_unavailable, list)
        assert len(service_unavailable) == 1
        assert "technical difficulties" in service_unavailable[0]
        assert "try again" in service_unavailable[0]
        
        # query_limit_exceeded is a list with rate limit message
        assert isinstance(query_limit, list)
        assert len(query_limit) == 1
        assert "maximum number of search requests" in query_limit[0]
        assert "wait" in query_limit[0]
    
    def test_fallback_message_constant(self):
        """Test that _FALLBACK_MESSAGE matches no_results_found."""
        assert _FALLBACK_MESSAGE == GENERIC_MESSAGES["no_results_found"]
        assert isinstance(_FALLBACK_MESSAGE, list)
        assert len(_FALLBACK_MESSAGE) == 1
