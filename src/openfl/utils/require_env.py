import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from openfl.utils.printer import log

env_loaded = False

def require_env_var(name: str) -> str:
    global env_loaded
    """Get an environment variable or exit with an error message."""
    if not env_loaded:
        load_env()
        env_loaded = True
    value = os.environ.get(name)
    if not value:
        print(f"Error: Environment variable '{name}' is missing or empty.")
        sys.exit(1)
    return value

def load_env():
    log("setup_env", "Loading environment")
    # Choose env file dynamically
    env = os.getenv("ENV", "ganache")  # defaults to dev if ENV not set
    env_file = Path(__file__).parents[3] / ".env" / f".env.{env}"

    log("setup_env", env_file)

    load_dotenv(env_file)

    log("setup_env", f"Loaded environment: {env_file}")