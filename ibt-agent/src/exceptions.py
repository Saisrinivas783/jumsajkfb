"""Essential exceptions for the IBT Agent."""

class IBTError(Exception):
    """Base exception for IBT Agent errors."""
    
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class UpstreamServiceError(IBTError):
    """Raised when an AWS dependency (Kendra, STS) call fails.

    Mapped to HTTP 500 by the app-level exception handler so the
    orchestrator's raise_for_status() sees the failure.
    """

    def __init__(self, service: str, message: str) -> None:
        self.service = service
        super().__init__(message)