import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Final


class Config:
    # Initialize environment variables once
    load_dotenv()

    # Project paths
    BASE_DIR: Final = Path(__file__).parent.parent.parent
    MODEL_DIR: Final = BASE_DIR.joinpath("src/ml/.trained")

    # Flask configuration
    FLASK_APP: Final[str] = os.getenv("FLASK_APP", "wsgi.py")
    FLASK_ENV: Final[str] = os.getenv("FLASK_ENV", "development")
    DEBUG: Final[bool] = os.getenv("FLASK_DEBUG", "1") == "1"
    HOST: Final[str] = os.getenv("HOST", "0.0.0.0")
    PORT: Final[int] = int(os.getenv("PORT", "8000"))
