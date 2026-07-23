from pathlib import Path
from typing import Iterator

import pandas as pd


def read_json_lines_gzip(path: str | Path, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    return pd.read_json(path, lines=True, compression="gzip", chunksize=chunksize)


def read_interactions(path: str | Path, chunksize: int = 1_000_000) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        path,
        usecols=["user_id", "book_id", "is_read", "rating", "is_reviewed"],
        dtype={
            "user_id": "int32",
            "book_id": "int32",
            "is_read": "int8",
            "rating": "int8",
            "is_reviewed": "int8",
        },
        chunksize=chunksize,
    )


def read_book_id_map(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"book_id_csv": "int32", "book_id": "string"})