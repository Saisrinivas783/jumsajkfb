"""Test script to run orchestrator agent on port 8080."""
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src.utils.logging import configure_logging
configure_logging()

from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
