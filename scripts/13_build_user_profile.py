from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a personalized content, collaborative, and hybrid user "
            "profile from the matched Goodreads library."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument(
        "--positive-only",
        action="store_true",
        help=(
            "Ignore negative preference weights. By default, low-rated books "
            "subtract from the user profile."
        ),
    )
    parser.add_argument(
        "--minimum-weight",
        type=float,
        default=0.05,
        help="Ignore profile rows whose absolute weight is below this value.",
    )
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=0.0,
        help=(
            "Optional exponential recency decay half-life. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.50,
        help="Content contribution to the final normalized hybrid vector.",
    )
    parser.add_argument(
        "--collaborative-weight",
        type=float,
        default=0.50,
        help="Collaborative contribution to the final normalized hybrid vector.",
    )
    parser.add_argument(
        "--top-authors",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--top-genres",
        type=int,
        default=30,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def first_existing(
    candidates: list[Path],
    label: str,
    recursive_root: Path | None = None,
    recursive_patterns: tuple[str, ...] = (),
) -> Path:
    for path in candidates:
        if path.exists():
            return path

    if recursive_root is not None and recursive_root.exists():
        matches: list[Path] = []
        for pattern in recursive_patterns:
            matches.extend(recursive_root.rglob(pattern))

        matches.extend(
            path
            for path in recursive_root.rglob("embeddings")
            if path.is_dir() and any(path.glob("*.npy"))
        )

        matches = sorted(
            {
                path.resolve()
                for path in matches
                if path.is_file()
                or (
                    path.is_dir()
                    and any(path.glob("*.npy"))
                )
            },
            key=lambda path: (
                0 if path.is_dir() and path.name.casefold() == "embeddings" else 1,
                len(path.parts),
                len(path.name),
                str(path).casefold(),
            ),
        )

        if matches:
            selected = matches[0]
            print(
                f"[auto-discovery] {label}: {selected}"
            )
            return selected

    checked = "\n  - ".join(str(path) for path in candidates)
    patterns = ", ".join(recursive_patterns) or "none"
    raise FileNotFoundError(
        f"Could not locate {label}. Checked:\n  - {checked}\n"
        f"Recursive root: {recursive_root}\n"
        f"Recursive patterns: {patterns}"
    )


class ShardedNpyMatrix:
    """Read row-indexed vectors from multiple NPY shards without concatenating them."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.files = sorted(directory.glob("part-*.npy"))
        if not self.files:
            self.files = sorted(directory.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(
                f"No NPY shards found in embedding directory: {directory}"
            )

        self.arrays = [
            np.load(path, mmap_mode="r")
            for path in self.files
        ]

        dimensions = {array.shape[1] for array in self.arrays}
        if len(dimensions) != 1:
            raise ValueError(
                f"Embedding shards have inconsistent dimensions: {dimensions}"
            )

        self.lengths = np.asarray(
            [array.shape[0] for array in self.arrays],
            dtype=np.int64,
        )
        self.offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(self.lengths),
            ]
        )
        self.shape = (
            int(self.lengths.sum()),
            int(next(iter(dimensions))),
        )

        print(
            f"[sharded embeddings] {len(self.files)} shard(s), "
            f"shape={self.shape}"
        )

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)

        shard_index = int(
            np.searchsorted(self.offsets, index, side="right") - 1
        )
        local_index = int(index - self.offsets[shard_index])
        return self.arrays[shard_index][local_index]


def load_array(path: Path, mmap: bool = True) -> Any:
    if path.is_dir():
        return ShardedNpyMatrix(path)

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None)

    if suffix == ".npz":
        data = np.load(path)
        keys = list(data.files)
        if len(keys) != 1:
            raise ValueError(
                f"{path} contains multiple arrays: {keys}. "
                "Expected exactly one."
            )
        return data[keys[0]]

    raise ValueError(f"Unsupported array format: {path}")


def discover_inputs(
    processed_dir: Path,
    profile_input: Path | None,
) -> dict[str, Path]:
    user_library = processed_dir / "user_library"
    hybrid_dir = processed_dir / "hybrid"
    content_dir = processed_dir / "content"
    collaborative_dir = processed_dir / "collaborative"

    return {
        "profile_books": profile_input or first_existing(
            [
                user_library / "profile_books.parquet",
                user_library / "imported_library.parquet",
            ],
            "user profile input",
            recursive_root=user_library,
            recursive_patterns=(
                "profile_books.parquet",
                "imported_library.parquet",
            ),
        ),
        "hybrid_map": first_existing(
            [
                hybrid_dir / "hybrid_item_map.parquet",
                hybrid_dir / "item_map.parquet",
            ],
            "hybrid item map",
            recursive_root=processed_dir,
            recursive_patterns=(
                "*hybrid*map*.parquet",
                "hybrid_item_map.parquet",
            ),
        ),
        "content_embeddings": first_existing(
            [
                content_dir / "content_embeddings.npy",
                content_dir / "book_embeddings.npy",
                content_dir / "svd_embeddings.npy",
                content_dir / "content_svd_embeddings.npy",
                content_dir / "index" / "embeddings",
                content_dir / "embeddings",
                hybrid_dir / "content_embeddings.npy",
                hybrid_dir / "hybrid_content_embeddings.npy",
            ],
            "content embeddings",
            recursive_root=processed_dir,
            recursive_patterns=(
                "*content*embedding*.npy",
                "*svd*embedding*.npy",
                "*book*embedding*.npy",
                "*content*vector*.npy",
            ),
        ),
        "collaborative_item_factors": first_existing(
            [
                collaborative_dir / "item_factors.npy",
                collaborative_dir / "als_item_factors.npy",
                processed_dir / "collaborative_model" / "item_factors.npy",
                processed_dir / "als" / "item_factors.npy",
            ],
            "collaborative item factors",
            recursive_root=processed_dir,
            recursive_patterns=(
                "*item_factors*.npy",
                "*item*factor*.npy",
                "*als*item*.npy",
                "*collaborative*embedding*.npy",
            ),
        ),
        "catalog": first_existing(
            [
                processed_dir / "catalog" / "catalog.parquet",
                processed_dir / "catalog.parquet",
                processed_dir / "catalog",
            ],
            "catalog",
            recursive_root=processed_dir,
            recursive_patterns=(
                "catalog.parquet",
                "*catalog*.parquet",
            ),
        ),
    }


def read_catalog(path: Path, book_ids: set[int]) -> pd.DataFrame:
    columns = [
        "book_id",
        "primary_author",
        "primary_genre",
        "title",
    ]

    def read_one(file: Path) -> pd.DataFrame:
        if not file.exists() or file.stat().st_size == 0:
            print(f"[catalog] skipped empty file: {file}")
            return pd.DataFrame(columns=columns)

        try:
            frame = pd.read_parquet(file, columns=columns)
        except Exception as exc:
            print(
                f"[catalog] skipped unreadable parquet: {file} "
                f"({type(exc).__name__}: {exc})"
            )
            return pd.DataFrame(columns=columns)

        if "book_id" not in frame.columns:
            return pd.DataFrame(columns=columns)

        numeric_ids = pd.to_numeric(
            frame["book_id"],
            errors="coerce",
        ).astype("Int64")

        return frame[numeric_ids.isin(book_ids)].copy()

    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=columns)

        frames = []
        for file in files:
            matched = read_one(file)
            if not matched.empty:
                frames.append(matched)

        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=columns)
        )

    frame = read_one(path)
    return frame.reset_index(drop=True)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm


def safe_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def recency_multiplier(
    date_value: Any,
    half_life_days: float,
    reference_date: pd.Timestamp,
) -> float:
    if half_life_days <= 0:
        return 1.0

    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return 1.0

    age_days = max(
        0.0,
        (reference_date.normalize() - date.normalize()).days,
    )
    return float(math.pow(0.5, age_days / half_life_days))


def prepare_profile_rows(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    required = {
        "preference_weight",
        "local_book_id",
        "content_global_row",
        "collaborative_item_index",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Profile input is missing required columns: "
            + ", ".join(sorted(missing))
        )

    profile = frame.copy()

    if "profile_eligible" in profile.columns:
        profile = profile[profile["profile_eligible"].fillna(False)]

    if "match_status" in profile.columns:
        profile = profile[profile["match_status"] == "matched"]

    profile["base_weight"] = pd.to_numeric(
        profile["preference_weight"],
        errors="coerce",
    ).fillna(0.0)

    if args.positive_only:
        profile = profile[profile["base_weight"] > 0]
    else:
        profile = profile[
            profile["base_weight"].abs() >= args.minimum_weight
        ]

    if args.positive_only:
        profile = profile[
            profile["base_weight"] >= args.minimum_weight
        ]

    reference_date = pd.Timestamp.now(tz=None)

    source_date = (
        profile["date_read"]
        if "date_read" in profile.columns
        else (
            profile["date_added"]
            if "date_added" in profile.columns
            else pd.Series([pd.NaT] * len(profile), index=profile.index)
        )
    )

    profile["recency_multiplier"] = [
        recency_multiplier(
            value,
            args.recency_half_life_days,
            reference_date,
        )
        for value in source_date
    ]

    profile["effective_weight"] = (
        profile["base_weight"] * profile["recency_multiplier"]
    )

    profile = profile[
        profile["effective_weight"].abs() >= args.minimum_weight
    ].copy()

    if profile.empty:
        raise ValueError(
            "No profile rows remain after filtering. "
            "Check ratings, shelves, and --minimum-weight."
        )

    return profile.reset_index(drop=True)


def weighted_profile(
    rows: pd.DataFrame,
    matrix: np.ndarray,
    index_column: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    selected_rows = []
    weighted_sum = np.zeros(matrix.shape[1], dtype=np.float64)
    absolute_weight_sum = 0.0

    for _, row in rows.iterrows():
        index = safe_index(row[index_column])
        if index is None or index < 0 or index >= matrix.shape[0]:
            continue

        vector = np.asarray(matrix[index], dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            continue

        weight = float(row["effective_weight"])
        weighted_sum += weight * vector
        absolute_weight_sum += abs(weight)

        selected_rows.append(
            {
                "library_row_id": safe_index(row.get("library_row_id")),
                "local_book_id": safe_index(row.get("local_book_id")),
                "input_title": row.get("input_title"),
                "input_author": row.get("input_author"),
                "index_column": index_column,
                "vector_index": index,
                "base_weight": float(row["base_weight"]),
                "recency_multiplier": float(row["recency_multiplier"]),
                "effective_weight": weight,
            }
        )

    if absolute_weight_sum <= 0:
        return np.zeros(matrix.shape[1], dtype=np.float32), pd.DataFrame(
            selected_rows
        )

    profile = weighted_sum / absolute_weight_sum
    return l2_normalize(profile), pd.DataFrame(selected_rows)


def combine_hybrid(
    content_profile: np.ndarray,
    collaborative_profile: np.ndarray,
    content_weight: float,
    collaborative_weight: float,
) -> np.ndarray:
    if content_weight < 0 or collaborative_weight < 0:
        raise ValueError("Hybrid weights cannot be negative.")

    total = content_weight + collaborative_weight
    if total <= 0:
        raise ValueError(
            "At least one of --content-weight or "
            "--collaborative-weight must be positive."
        )

    content_weight /= total
    collaborative_weight /= total

    content_component = l2_normalize(content_profile)
    collaborative_component = l2_normalize(collaborative_profile)

    hybrid = np.concatenate(
        [
            content_component * content_weight,
            collaborative_component * collaborative_weight,
        ]
    ).astype(np.float32)

    return l2_normalize(hybrid)


def preference_tables(
    profile: pd.DataFrame,
    catalog: pd.DataFrame,
    top_authors: int,
    top_genres: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if catalog.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = profile.merge(
        catalog,
        left_on="local_book_id",
        right_on="book_id",
        how="left",
    )

    positive = merged[merged["effective_weight"] > 0].copy()

    authors = (
        positive.dropna(subset=["primary_author"])
        .groupby("primary_author", as_index=False)
        .agg(
            weighted_preference=("effective_weight", "sum"),
            books=("local_book_id", "nunique"),
        )
        .sort_values(
            ["weighted_preference", "books"],
            ascending=False,
        )
        .head(top_authors)
        .reset_index(drop=True)
    )

    genres = (
        positive.dropna(subset=["primary_genre"])
        .groupby("primary_genre", as_index=False)
        .agg(
            weighted_preference=("effective_weight", "sum"),
            books=("local_book_id", "nunique"),
        )
        .sort_values(
            ["weighted_preference", "books"],
            ascending=False,
        )
        .head(top_genres)
        .reset_index(drop=True)
    )

    return authors, genres


def write_vector_parquet(
    path: Path,
    vector_name: str,
    vector: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "dimension": np.arange(len(vector), dtype=np.int32),
            vector_name: vector.astype(np.float32),
        }
    ).to_parquet(path, index=False, compression="zstd")


def main() -> None:
    args = parse_args()

    if args.minimum_weight < 0:
        raise ValueError("--minimum-weight cannot be negative.")
    if args.recency_half_life_days < 0:
        raise ValueError("--recency-half-life-days cannot be negative.")
    if args.top_authors <= 0 or args.top_genres <= 0:
        raise ValueError("--top-authors and --top-genres must be positive.")

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.processed_dir or config["paths"]["processed_dir"]
    )

    profile_input = resolve_path(args.input) if args.input else None
    output_dir = resolve_path(
        args.output_dir
        or (processed_dir / "user_profile")
    )

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Use --overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(processed_dir, profile_input)

    profile_raw = pd.read_parquet(inputs["profile_books"])
    profile = prepare_profile_rows(profile_raw, args)

    content_embeddings = load_array(
        inputs["content_embeddings"],
        mmap=True,
    )
    collaborative_factors = load_array(
        inputs["collaborative_item_factors"],
        mmap=True,
    )

    content_profile, content_sources = weighted_profile(
        profile,
        content_embeddings,
        "content_global_row",
    )
    collaborative_profile, collaborative_sources = weighted_profile(
        profile,
        collaborative_factors,
        "collaborative_item_index",
    )

    if not np.any(content_profile):
        raise ValueError(
            "Could not build a content profile. "
            "No valid content vector indices were found."
        )

    if not np.any(collaborative_profile):
        raise ValueError(
            "Could not build a collaborative profile. "
            "No valid collaborative item indices were found."
        )

    hybrid_profile = combine_hybrid(
        content_profile,
        collaborative_profile,
        args.content_weight,
        args.collaborative_weight,
    )

    local_book_ids = {
        int(value)
        for value in profile["local_book_id"].dropna().tolist()
    }
    catalog = read_catalog(inputs["catalog"], local_book_ids)

    favorite_authors, favorite_genres = preference_tables(
        profile,
        catalog,
        args.top_authors,
        args.top_genres,
    )

    outputs = {
        "content_npy": output_dir / "user_content_embedding.npy",
        "collaborative_npy": (
            output_dir / "user_collaborative_embedding.npy"
        ),
        "hybrid_npy": output_dir / "user_hybrid_embedding.npy",
        "content_parquet": (
            output_dir / "user_content_embedding.parquet"
        ),
        "collaborative_parquet": (
            output_dir / "user_collaborative_embedding.parquet"
        ),
        "hybrid_parquet": (
            output_dir / "user_hybrid_embedding.parquet"
        ),
        "profile_books": output_dir / "profile_books_used.parquet",
        "content_sources": output_dir / "content_profile_sources.parquet",
        "collaborative_sources": (
            output_dir / "collaborative_profile_sources.parquet"
        ),
        "favorite_authors": output_dir / "favorite_authors.parquet",
        "favorite_genres": output_dir / "favorite_genres.parquet",
        "statistics": output_dir / "user_statistics.json",
        "manifest": output_dir / "user_profile.json",
    }

    np.save(outputs["content_npy"], content_profile.astype(np.float32))
    np.save(
        outputs["collaborative_npy"],
        collaborative_profile.astype(np.float32),
    )
    np.save(outputs["hybrid_npy"], hybrid_profile.astype(np.float32))

    write_vector_parquet(
        outputs["content_parquet"],
        "content_value",
        content_profile,
    )
    write_vector_parquet(
        outputs["collaborative_parquet"],
        "collaborative_value",
        collaborative_profile,
    )
    write_vector_parquet(
        outputs["hybrid_parquet"],
        "hybrid_value",
        hybrid_profile,
    )

    profile.to_parquet(
        outputs["profile_books"],
        index=False,
        compression="zstd",
    )
    content_sources.to_parquet(
        outputs["content_sources"],
        index=False,
        compression="zstd",
    )
    collaborative_sources.to_parquet(
        outputs["collaborative_sources"],
        index=False,
        compression="zstd",
    )
    favorite_authors.to_parquet(
        outputs["favorite_authors"],
        index=False,
        compression="zstd",
    )
    favorite_genres.to_parquet(
        outputs["favorite_genres"],
        index=False,
        compression="zstd",
    )

    statistics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(profile_raw)),
        "profile_rows_used": int(len(profile)),
        "positive_rows": int((profile["effective_weight"] > 0).sum()),
        "negative_rows": int((profile["effective_weight"] < 0).sum()),
        "content_source_rows": int(len(content_sources)),
        "collaborative_source_rows": int(len(collaborative_sources)),
        "content_dimensions": int(len(content_profile)),
        "collaborative_dimensions": int(len(collaborative_profile)),
        "hybrid_dimensions": int(len(hybrid_profile)),
        "content_norm": float(np.linalg.norm(content_profile)),
        "collaborative_norm": float(
            np.linalg.norm(collaborative_profile)
        ),
        "hybrid_norm": float(np.linalg.norm(hybrid_profile)),
        "mean_effective_weight": float(
            profile["effective_weight"].mean()
        ),
        "absolute_weight_sum": float(
            profile["effective_weight"].abs().sum()
        ),
        "rating_distribution": (
            profile["my_rating"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
            if "my_rating" in profile.columns
            else {}
        ),
        "shelf_distribution": (
            profile["exclusive_shelf"]
            .value_counts(dropna=False)
            .to_dict()
            if "exclusive_shelf" in profile.columns
            else {}
        ),
    }

    outputs["statistics"].write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "created_utc": statistics["created_utc"],
        "profile_type": "weighted_goodreads_hybrid_user_profile",
        "settings": {
            "positive_only": args.positive_only,
            "minimum_weight": args.minimum_weight,
            "recency_half_life_days": args.recency_half_life_days,
            "content_weight": args.content_weight,
            "collaborative_weight": args.collaborative_weight,
        },
        "inputs": {
            key: str(value)
            for key, value in inputs.items()
        },
        "outputs": {
            key: str(value)
            for key, value in outputs.items()
        },
        "statistics": statistics,
    }

    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("USER PROFILE BUILD COMPLETE")
    print(f"Profile rows used     : {len(profile):,}")
    print(f"Positive rows         : {(profile['effective_weight'] > 0).sum():,}")
    print(f"Negative rows         : {(profile['effective_weight'] < 0).sum():,}")
    print(f"Content dimensions    : {len(content_profile):,}")
    print(f"Collaborative dims    : {len(collaborative_profile):,}")
    print(f"Hybrid dimensions     : {len(hybrid_profile):,}")
    print(f"Favorite authors      : {len(favorite_authors):,}")
    print(f"Favorite genres       : {len(favorite_genres):,}")
    print(f"Output directory      : {output_dir}")
    print(f"Profile manifest      : {outputs['manifest']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
