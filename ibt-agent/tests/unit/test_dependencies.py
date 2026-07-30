"""Clean unit tests for API dependencies."""

import pytest
from unittest.mock import patch, MagicMock
from src.api.dependencies import get_ibt
from src.agent.hybrid_ibt import HybridIBTAgent

class TestDependencies:
    """Tests for FastAPI dependency injection."""
    
    @patch('src.api.dependencies.HybridIBTAgent')
    def test_get_ibt_creates_agent(self, mock_hybrid_agent):
        """Test get_ibt creates HybridIBTAgent instance."""
        mock_agent = MagicMock()
        mock_hybrid_agent.return_value = mock_agent
        
        # Clear cache first
        get_ibt.cache_clear()
        
        result = get_ibt()
        
        mock_hybrid_agent.assert_called_once()
        assert result == mock_agent
    
    @patch('src.api.dependencies.HybridIBTAgent')  
    def test_get_ibt_cached(self, mock_hybrid_agent):
        """Test get_ibt returns cached instance due to lru_cache."""
        # Clear cache first
        get_ibt.cache_clear()
        
        mock_agent = MagicMock()
        mock_hybrid_agent.return_value = mock_agent
        
        # Call multiple times
        result1 = get_ibt()
        result2 = get_ibt()
        result3 = get_ibt()
        
        # Should only create agent once due to caching
        mock_hybrid_agent.assert_called_once()
        assert result1 is result2
        assert result2 is result3
    
    def test_get_ibt_cache_clear(self):
        """Test get_ibt cache can be cleared."""
        # Clear any existing cache
        get_ibt.cache_clear()
        
        with patch('src.api.dependencies.HybridIBTAgent') as mock_hybrid_agent:
            mock_agent1 = MagicMock()
            mock_agent2 = MagicMock()
            mock_hybrid_agent.side_effect = [mock_agent1, mock_agent2]
            
            # First call
            result1 = get_ibt()
            
            # Clear cache
            get_ibt.cache_clear()
            
            # Second call should create new instance
            result2 = get_ibt()
            
            assert mock_hybrid_agent.call_count == 2
            assert result1 is not result2
    
    def test_get_ibt_cache_info(self):
        """Test get_ibt cache info functionality."""
        # Clear cache first
        get_ibt.cache_clear()
        
        with patch('src.api.dependencies.HybridIBTAgent'):
            # Check initial cache info
            info = get_ibt.cache_info()
            assert info.hits == 0
            assert info.misses == 0
            
            # Make some calls
            get_ibt()  # miss
            get_ibt()  # hit
            get_ibt()  # hit
            
            # Check updated cache info
            info = get_ibt.cache_info()
            assert info.hits == 2
            assert info.misses == 1
    
    @patch('src.api.dependencies.HybridIBTAgent')
    def test_get_ibt_exception_handling(self, mock_hybrid_agent):
        """Test get_ibt handles HybridIBTAgent creation exceptions."""
        # Clear cache first
        get_ibt.cache_clear()
        
        mock_hybrid_agent.side_effect = RuntimeError("Agent initialization failed")
        
        with pytest.raises(RuntimeError, match="Agent initialization failed"):
            get_ibt()
    
    def test_get_ibt_import_dependencies(self):
        """Test that get_ibt imports are correct."""
        # This test ensures the imports work correctly
        from src.api.dependencies import get_ibt
        from src.agent.hybrid_ibt import HybridIBTAgent
        
        # Verify the function exists and is callable
        assert callable(get_ibt)
        
        # Verify HybridIBTAgent is imported correctly
        assert HybridIBTAgent is not None