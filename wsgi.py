from typing import Any
from src.app import create_app as init_app
import os


def create_app() -> Any:
    """
    Initialize and return the Flask application instance.

    Returns:
        Flask: Configured Flask application instance
    """
    return init_app()


application = create_app()

if __name__ == "__main__":
    application.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "True").lower() == "true",
    )
