from pathlib import Path
from typing import Any

import yaml


REQUIRED_PATHS = (
    "raw_dir",
    "interim_dir",
    "processed_dir",
)

def load_config(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(
            "Configuration file is empty."
        )

    if "paths" not in config:
        raise KeyError(
            "Missing 'paths' section in config/settings.yaml."
        )

    paths = config["paths"]

    for key in REQUIRED_PATHS:
        value = paths.get(key)

        if value is None or str(value).strip() == "":
            raise ValueError(
                f"'{key}' is not configured.\n\n"
                "Please edit config/settings.yaml and specify your local data directories."
            )

        paths[key] = Path(value).expanduser().resolve()

    # Optional path
    models_dir = paths.get("models_dir")
    if models_dir:
        paths["models_dir"] = Path(models_dir).expanduser().resolve()

    return config