from pathlib import Path
import sys

# Add repository root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config

def main() -> None:
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])

    print(f"Raw directory: {raw_dir.resolve()}")
    for logical_name, filename in config["files"].items():
        path = raw_dir / filename
        status = "FOUND" if path.exists() else "MISSING"
        size_gb = path.stat().st_size / 1024**3 if path.exists() else 0
        print(f"{status:7} {logical_name:15} {filename:45} {size_gb:8.2f} GB")


if __name__ == "__main__":
    main()
