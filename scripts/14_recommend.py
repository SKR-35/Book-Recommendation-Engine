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
            "Generate personalized book recommendations from the saved "
            "192-dimensional hybrid user profile."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--user-profile", default=None)
    parser.add_argument("--hybrid-embeddings", default=None)
    parser.add_argument("--hybrid-map", default=None)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--exclusions", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=1000,
        help="Number of nearest hybrid candidates retained before reranking.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="Rows scored per batch to control RAM usage.",
    )

    parser.add_argument(
        "--similarity-weight",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--popularity-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--minimum-ratings",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional exact language-code filter.",
    )
    parser.add_argument(
        "--genre",
        default=None,
        help="Optional case-insensitive substring genre filter.",
    )
    parser.add_argument(
        "--exclude-author",
        action="append",
        default=[],
        help="Author substring to exclude. May be supplied multiple times.",
    )
    parser.add_argument(
        "--max-per-author",
        type=int,
        default=3,
        help="Maximum recommendations per author; use 0 to disable.",
    )
    parser.add_argument(
        "--novelty",
        type=float,
        default=0.0,
        help=(
            "Novelty preference from 0 to 1. Higher values mildly favor "
            "less-popular books during reranking."
        ),
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
    patterns: tuple[str, ...] = (),
) -> Path:
    for path in candidates:
        if path.exists() and (not path.is_file() or path.stat().st_size > 0):
            return path

    if recursive_root is not None and recursive_root.exists():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(recursive_root.rglob(pattern))

        matches = [
            path.resolve()
            for path in matches
            if path.exists()
            and (
                path.is_dir()
                or (path.is_file() and path.stat().st_size > 0)
            )
        ]

        if matches:
            matches = sorted(
                set(matches),
                key=lambda path: (
                    len(path.parts),
                    len(path.name),
                    str(path).casefold(),
                ),
            )
            selected = matches[0]
            print(f"[auto-discovery] {label}: {selected}")
            return selected

    raise FileNotFoundError(
        f"Could not locate {label}."
    )


def discover_inputs(
    processed_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    hybrid_dir = processed_dir / "hybrid"
    user_profile_dir = processed_dir / "user_profile"
    user_library_dir = processed_dir / "user_library"

    return {
        "user_profile": (
            resolve_path(args.user_profile)
            if args.user_profile
            else first_existing(
                [
                    user_profile_dir / "user_hybrid_embedding.npy",
                ],
                "user hybrid profile",
                recursive_root=processed_dir,
                patterns=("user_hybrid_embedding.npy",),
            )
        ),
        "hybrid_embeddings": (
            resolve_path(args.hybrid_embeddings)
            if args.hybrid_embeddings
            else first_existing(
                [
                    hybrid_dir / "hybrid_embeddings.npy",
                ],
                "hybrid embeddings",
                recursive_root=processed_dir,
                patterns=("hybrid_embeddings.npy", "*hybrid*embedding*.npy"),
            )
        ),
        "hybrid_map": (
            resolve_path(args.hybrid_map)
            if args.hybrid_map
            else first_existing(
                [
                    hybrid_dir / "hybrid_item_map.parquet",
                    hybrid_dir / "item_map.parquet",
                ],
                "hybrid item map",
                recursive_root=processed_dir,
                patterns=("*hybrid*map*.parquet", "hybrid_item_map.parquet"),
            )
        ),
        "catalog": (
            resolve_path(args.catalog)
            if args.catalog
            else first_existing(
                [
                    processed_dir / "catalog",
                    processed_dir / "catalog.parquet",
                ],
                "catalog",
                recursive_root=processed_dir,
                patterns=("catalog.parquet", "*catalog*.parquet"),
            )
        ),
        "exclusions": (
            resolve_path(args.exclusions)
            if args.exclusions
            else first_existing(
                [
                    user_library_dir / "recommendation_exclusions.parquet",
                ],
                "recommendation exclusions",
                recursive_root=user_library_dir,
                patterns=("recommendation_exclusions.parquet",),
            )
        ),
    }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Vector has zero or invalid norm.")
    return vector / norm


def top_hybrid_candidates(
    embeddings_path: Path,
    query: np.ndarray,
    candidate_pool: int,
    batch_size: int,
) -> pd.DataFrame:
    matrix = np.load(embeddings_path, mmap_mode="r")

    if matrix.ndim != 2:
        raise ValueError(
            f"Hybrid embeddings must be 2D; received shape {matrix.shape}."
        )
    if matrix.shape[1] != len(query):
        raise ValueError(
            "Dimension mismatch: "
            f"matrix has {matrix.shape[1]}, query has {len(query)}."
        )

    candidate_pool = min(candidate_pool, matrix.shape[0])
    best_indices = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.float32)

    for start in range(0, matrix.shape[0], batch_size):
        end = min(start + batch_size, matrix.shape[0])
        block = np.asarray(matrix[start:end], dtype=np.float32)

        norms = np.linalg.norm(block, axis=1)
        valid = np.isfinite(norms) & (norms > 0)

        scores = np.full(len(block), -np.inf, dtype=np.float32)
        if valid.any():
            scores[valid] = (
                block[valid] @ query
            ) / norms[valid]

        local_k = min(candidate_pool, len(scores))
        if local_k == len(scores):
            local_positions = np.arange(len(scores))
        else:
            local_positions = np.argpartition(
                scores,
                -local_k,
            )[-local_k:]

        local_indices = local_positions.astype(np.int64) + start
        local_scores = scores[local_positions]

        best_indices = np.concatenate([best_indices, local_indices])
        best_scores = np.concatenate([best_scores, local_scores])

        if len(best_scores) > candidate_pool * 2:
            keep = np.argpartition(
                best_scores,
                -candidate_pool,
            )[-candidate_pool:]
            best_indices = best_indices[keep]
            best_scores = best_scores[keep]

    order = np.argsort(best_scores)[::-1][:candidate_pool]

    return pd.DataFrame(
        {
            "hybrid_index": best_indices[order],
            "hybrid_similarity": best_scores[order],
        }
    )


def read_hybrid_map(path: Path, indices: set[int]) -> pd.DataFrame:
    columns = [
        "hybrid_index",
        "book_id",
        "content_global_row",
        "collaborative_item_index",
    ]

    frame = pd.read_parquet(path)
    available = [column for column in columns if column in frame.columns]
    frame = frame[available].copy()

    if "hybrid_index" not in frame.columns:
        frame = frame.reset_index().rename(
            columns={"index": "hybrid_index"}
        )

    numeric_index = pd.to_numeric(
        frame["hybrid_index"],
        errors="coerce",
    ).astype("Int64")

    return frame[numeric_index.isin(indices)].copy()


def read_catalog(
    path: Path,
    book_ids: set[int],
) -> pd.DataFrame:
    desired = [
        "book_id",
        "title",
        "primary_author",
        "primary_genre",
        "language_code",
        "publication_year",
        "ratings_count",
        "average_rating",
        "bayesian_rating",
        "wilson_lower_bound",
        "image_url",
        "description",
        "isbn",
        "isbn13",
    ]

    def read_one(file: Path) -> pd.DataFrame:
        if not file.exists() or file.stat().st_size == 0:
            return pd.DataFrame()

        try:
            available = pd.read_parquet(file).columns.tolist()
            columns = [column for column in desired if column in available]
            if "book_id" not in columns:
                return pd.DataFrame()

            frame = pd.read_parquet(file, columns=columns)
        except Exception as exc:
            print(
                f"[catalog] skipped {file}: "
                f"{type(exc).__name__}: {exc}"
            )
            return pd.DataFrame()

        ids = pd.to_numeric(
            frame["book_id"],
            errors="coerce",
        ).astype("Int64")
        return frame[ids.isin(book_ids)].copy()

    if path.is_dir():
        frames = []
        for file in sorted(path.rglob("*.parquet")):
            frame = read_one(file)
            if not frame.empty:
                frames.append(frame)
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )

    return read_one(path)


def read_exclusions(path: Path) -> tuple[set[int], set[int]]:
    frame = pd.read_parquet(path)

    book_ids: set[int] = set()
    work_ids: set[int] = set()

    if "local_book_id" in frame.columns:
        book_ids = {
            int(value)
            for value in pd.to_numeric(
                frame["local_book_id"],
                errors="coerce",
            ).dropna()
        }

    if "work_id" in frame.columns:
        work_ids = {
            int(value)
            for value in pd.to_numeric(
                frame["work_id"],
                errors="coerce",
            ).dropna()
        }

    return book_ids, work_ids


def minmax(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()

    if pd.isna(minimum) or pd.isna(maximum) or maximum <= minimum:
        return pd.Series(
            np.zeros(len(series), dtype=np.float32),
            index=series.index,
        )

    return (numeric - minimum) / (maximum - minimum)


def apply_filters(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    excluded_book_ids: set[int],
    excluded_work_ids: set[int],
) -> pd.DataFrame:
    filtered = frame.copy()

    if "book_id" in filtered.columns:
        filtered = filtered[
            ~pd.to_numeric(
                filtered["book_id"],
                errors="coerce",
            ).isin(excluded_book_ids)
        ]

    if "work_id" in filtered.columns and excluded_work_ids:
        filtered = filtered[
            ~pd.to_numeric(
                filtered["work_id"],
                errors="coerce",
            ).isin(excluded_work_ids)
        ]

    if args.minimum_ratings > 0 and "ratings_count" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(
                filtered["ratings_count"],
                errors="coerce",
            ).fillna(0) >= args.minimum_ratings
        ]

    if args.language and "language_code" in filtered.columns:
        filtered = filtered[
            filtered["language_code"]
            .fillna("")
            .astype(str)
            .str.casefold()
            .eq(args.language.casefold())
        ]

    if args.genre and "primary_genre" in filtered.columns:
        filtered = filtered[
            filtered["primary_genre"]
            .fillna("")
            .astype(str)
            .str.contains(
                args.genre,
                case=False,
                regex=False,
            )
        ]

    for author in args.exclude_author:
        if "primary_author" in filtered.columns:
            filtered = filtered[
                ~filtered["primary_author"]
                .fillna("")
                .astype(str)
                .str.contains(
                    author,
                    case=False,
                    regex=False,
                )
            ]

    return filtered.reset_index(drop=True)


def rerank(
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    result = frame.copy()

    similarity = minmax(result["hybrid_similarity"])

    if "ratings_count" in result.columns:
        popularity_raw = np.log1p(
            pd.to_numeric(
                result["ratings_count"],
                errors="coerce",
            ).fillna(0)
        )
        popularity = minmax(popularity_raw)
    else:
        popularity = pd.Series(
            np.zeros(len(result)),
            index=result.index,
        )

    quality_source = None
    for column in [
        "bayesian_rating",
        "wilson_lower_bound",
        "average_rating",
    ]:
        if column in result.columns:
            quality_source = column
            break

    if quality_source:
        quality = minmax(result[quality_source])
    else:
        quality = pd.Series(
            np.zeros(len(result)),
            index=result.index,
        )

    novelty = 1.0 - popularity

    weights = np.asarray(
        [
            args.similarity_weight,
            args.popularity_weight,
            args.quality_weight,
        ],
        dtype=np.float64,
    )
    if np.any(weights < 0):
        raise ValueError("Reranking weights cannot be negative.")
    if float(weights.sum()) <= 0:
        raise ValueError("At least one reranking weight must be positive.")
    weights /= weights.sum()

    result["similarity_component"] = similarity
    result["popularity_component"] = popularity
    result["quality_component"] = quality
    result["novelty_component"] = novelty

    result["recommendation_score"] = (
        weights[0] * similarity
        + weights[1] * (
            (1.0 - args.novelty) * popularity
            + args.novelty * novelty
        )
        + weights[2] * quality
    )

    return result.sort_values(
        [
            "recommendation_score",
            "hybrid_similarity",
        ],
        ascending=False,
    ).reset_index(drop=True)


def diversify_authors(
    frame: pd.DataFrame,
    top_n: int,
    max_per_author: int,
) -> pd.DataFrame:
    if max_per_author <= 0 or "primary_author" not in frame.columns:
        return frame.head(top_n).copy()

    selected = []
    counts: dict[str, int] = {}

    for _, row in frame.iterrows():
        author = str(row.get("primary_author") or "Unknown").strip()
        key = author.casefold()

        if counts.get(key, 0) >= max_per_author:
            continue

        selected.append(row)
        counts[key] = counts.get(key, 0) + 1

        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def build_reason(row: pd.Series) -> str:
    parts = [
        f"hybrid similarity {float(row['hybrid_similarity']):.3f}"
    ]

    genre = row.get("primary_genre")
    if isinstance(genre, str) and genre.strip():
        parts.append(f"genre: {genre}")

    quality = row.get("bayesian_rating")
    if quality is None or pd.isna(quality):
        quality = row.get("average_rating")

    if quality is not None and not pd.isna(quality):
        parts.append(f"rating {float(quality):.2f}")

    return "; ".join(parts)


def main() -> None:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")
    if args.candidate_pool < args.top_n:
        raise ValueError("--candidate-pool must be at least --top-n.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0 <= args.novelty <= 1:
        raise ValueError("--novelty must be between 0 and 1.")

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.processed_dir or config["paths"]["processed_dir"]
    )
    output_dir = resolve_path(
        args.output_dir
        or (processed_dir / "recommendations")
    )

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Use --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(processed_dir, args)

    query = l2_normalize(
        np.load(inputs["user_profile"]).astype(np.float32)
    )

    candidates = top_hybrid_candidates(
        embeddings_path=inputs["hybrid_embeddings"],
        query=query,
        candidate_pool=args.candidate_pool,
        batch_size=args.batch_size,
    )

    index_set = set(
        candidates["hybrid_index"].astype(int).tolist()
    )
    mapping = read_hybrid_map(
        inputs["hybrid_map"],
        index_set,
    )

    candidates = candidates.merge(
        mapping,
        on="hybrid_index",
        how="left",
    )

    if "book_id" not in candidates.columns:
        raise ValueError(
            "Hybrid map does not contain book_id."
        )

    book_ids = {
        int(value)
        for value in pd.to_numeric(
            candidates["book_id"],
            errors="coerce",
        ).dropna()
    }

    catalog = read_catalog(
        inputs["catalog"],
        book_ids,
    )

    if catalog.empty:
        raise ValueError(
            "No catalog rows were found for the hybrid candidates."
        )

    recommendations = candidates.merge(
        catalog,
        on="book_id",
        how="left",
    )

    excluded_book_ids, excluded_work_ids = read_exclusions(
        inputs["exclusions"]
    )

    recommendations = apply_filters(
        recommendations,
        args,
        excluded_book_ids,
        excluded_work_ids,
    )

    if recommendations.empty:
        raise ValueError(
            "No candidates remain after exclusions and filters."
        )

    recommendations = rerank(recommendations, args)
    recommendations = diversify_authors(
        recommendations,
        args.top_n,
        args.max_per_author,
    )

    recommendations.insert(
        0,
        "rank",
        np.arange(1, len(recommendations) + 1),
    )
    recommendations["recommendation_reason"] = (
        recommendations.apply(build_reason, axis=1)
    )

    output_parquet = output_dir / "recommendations.parquet"
    output_csv = output_dir / "recommendations.csv"
    output_json = output_dir / "recommendations.json"
    report_path = output_dir / "recommendation_report.json"

    recommendations.to_parquet(
        output_parquet,
        index=False,
        compression="zstd",
    )
    recommendations.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    output_json.write_text(
        recommendations.to_json(
            orient="records",
            force_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "top_n_requested": args.top_n,
        "recommendations_returned": len(recommendations),
        "candidate_pool": args.candidate_pool,
        "excluded_book_ids": len(excluded_book_ids),
        "excluded_work_ids": len(excluded_work_ids),
        "settings": {
            "similarity_weight": args.similarity_weight,
            "popularity_weight": args.popularity_weight,
            "quality_weight": args.quality_weight,
            "novelty": args.novelty,
            "minimum_ratings": args.minimum_ratings,
            "language": args.language,
            "genre": args.genre,
            "max_per_author": args.max_per_author,
        },
        "inputs": {
            key: str(value)
            for key, value in inputs.items()
        },
        "outputs": {
            "parquet": str(output_parquet),
            "csv": str(output_csv),
            "json": str(output_json),
        },
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("PERSONALIZED RECOMMENDATIONS COMPLETE")
    print(f"Candidates searched : {args.candidate_pool:,}")
    print(f"Recommendations     : {len(recommendations):,}")
    print(f"Excluded known books: {len(excluded_book_ids):,}")
    print(f"Output directory    : {output_dir}")
    print("=" * 76)

    display_columns = [
        column
        for column in [
            "rank",
            "title",
            "primary_author",
            "primary_genre",
            "hybrid_similarity",
            "recommendation_score",
        ]
        if column in recommendations.columns
    ]
    print(
        recommendations[display_columns]
        .head(min(20, len(recommendations)))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
