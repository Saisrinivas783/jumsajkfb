"""Unit tests for logging utilities."""

import logging
import pytest
import src.utils.logging as logging_module


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Restore logging configured state after each test."""
    original = logging_module._logging_configured
    yield
    logging_module._logging_configured = original


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_returns_early_if_already_configured(self):
        """Line 37: early return when _logging_configured is True."""
        logging_module._logging_configured = True
        from src.utils.logging import configure_logging
        # Should not raise and should return immediately
        configure_logging()
        assert logging_module._logging_configured is True

    def test_sets_configured_flag(self):
        logging_module._logging_configured = False
        from src.utils.logging import configure_logging
        configure_logging()
        assert logging_module._logging_configured is True

    def test_configure_with_explicit_level(self):
        logging_module._logging_configured = False
        from src.utils.logging import configure_logging
        configure_logging(level="WARNING")
        assert logging_module._logging_configured is True

    def test_configure_with_custom_format(self):
        logging_module._logging_configured = False
        from src.utils.logging import configure_logging
        configure_logging(format_string="%(message)s")
        assert logging_module._logging_configured is True

    def test_idempotent_when_called_twice(self):
        logging_module._logging_configured = False
        from src.utils.logging import configure_logging
        configure_logging(level="DEBUG")
        configure_logging(level="ERROR")  # second call should be ignored
        assert logging_module._logging_configured is True


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger_instance(self):
        from src.utils.logging import get_logger
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        from src.utils.logging import get_logger
        logger = get_logger("my.test.module")
        assert logger.name == "my.test.module"

    def test_triggers_configure_if_not_configured(self):
        logging_module._logging_configured = False
        from src.utils.logging import get_logger
        get_logger("test.trigger")
        assert logging_module._logging_configured is True


