"""Essential exceptions for the IBT Agent."""

class IBTError(Exception):
    """Base exception for IBT Agent errors."""
    
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)