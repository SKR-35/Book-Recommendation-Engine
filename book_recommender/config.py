from pathlib import Path
from typing import Any
import os

import yaml


REQUIRED_PATHS = (
    "raw_dir",
    "interim_dir",
    "processed_dir",
)

PATH_ENV_VARS = {
    "raw_dir": "BOOK_REC_RAW",
    "interim_dir": "BOOK_REC_INTERIM",
    "processed_dir": "BOOK_REC_PROCESSED",
}


def load_config(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    """Load project configuration.

    Required data paths are resolved in this order:

    1. Environment variable
    2. Value from config/settings.yaml

    Environment variables therefore override local YAML values.
    """

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Configuration file is empty.")

    if "paths" not in config:
        raise KeyError(
            "Missing 'paths' section in config/settings.yaml."
        )

    paths = config["paths"]

    if not isinstance(paths, dict):
        raise TypeError(
            "'paths' in config/settings.yaml must be a mapping."
        )

    for key in REQUIRED_PATHS:
        env_name = PATH_ENV_VARS[key]
        env_value = os.getenv(env_name)
        yaml_value = paths.get(key)

        value = env_value if env_value and env_value.strip() else yaml_value

        if value is None or str(value).strip() == "":
            raise ValueError(
                f"'{key}' is not configured.\n\n"
                f"Set the environment variable '{env_name}' "
                "or specify the path in config/settings.yaml."
            )

        paths[key] = Path(str(value)).expanduser().resolve()

    models_dir = paths.get("models_dir")
    if models_dir is not None and str(models_dir).strip() != "":
        paths["models_dir"] = Path(str(models_dir)).expanduser().resolve()

    return config
