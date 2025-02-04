from typing import Optional
from flask import Flask
from src.api.routes import api as api_blueprint
from src.core.config import Config


def create_app(config_object: Optional[object] = None) -> Flask:
    """Create and configure Flask application instance.

    Args:
        config_object: Optional configuration object to override defaults

    Returns:
        Flask application instance

    Raises:
        RuntimeError: If app creation fails
    """
    try:
        app = Flask(__name__)
        app.config.from_object(config_object or Config)

        # Register blueprints
        app.register_blueprint(api_blueprint, url_prefix="/api")

        return app
    except Exception as e:
        raise RuntimeError(f"Failed to create app: {str(e)}") from e


if __name__ == "__main__":
    app = create_app()
    app.run(
        host=app.config.get("HOST", "0.0.0.0"),
        port=app.config.get("PORT", 8000),
        debug=app.config.get("DEBUG", True),
    )
