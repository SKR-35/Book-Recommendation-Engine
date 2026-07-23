from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
            "Explain personalized book recommendations by linking each "
            "recommendation to the user's most relevant liked books, shared "
            "authors/genres, and model scores."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--recommendations", default=None)
    parser.add_argument("--profile-books", default=None)
    parser.add_argument("--content-embeddings", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--evidence-books",
        type=int,
        default=3,
        help="Maximum number of liked books cited per recommendation.",
    )
    parser.add_argument(
        "--short-evidence-books",
        type=int,
        default=2,
        help="Number of evidence books shown in the short explanation.",
    )
    parser.add_argument(
        "--max-evidence-uses",
        type=int,
        default=5,
        help=(
            "Maximum number of recommendation explanations in which the same "
            "profile book may appear. Use 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--genre-bonus",
        type=float,
        default=0.10,
        help="Bonus added when recommendation and evidence share a genre.",
    )
    parser.add_argument(
        "--author-bonus",
        type=float,
        default=0.10,
        help="Bonus added when recommendation and evidence share an author.",
    )
    parser.add_argument(
        "--diversity-bonus",
        type=float,
        default=0.05,
        help=(
            "Bonus for evidence books from authors not yet used in the same "
            "explanation."
        ),
    )
    parser.add_argument(
        "--minimum-profile-weight",
        type=float,
        default=0.20,
        help="Minimum positive effective weight for evidence books.",
    )
    parser.add_argument(
        "--minimum-content-similarity",
        type=float,
        default=0.05,
        help="Minimum similarity required to cite a profile book.",
    )
    parser.add_argument(
        "--include-model-details",
        action="store_true",
        help="Include numeric scoring details in the natural-language reason.",
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
        if path.exists() and (
            path.is_dir()
            or (path.is_file() and path.stat().st_size > 0)
        ):
            return path

    if recursive_root is not None and recursive_root.exists():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(recursive_root.rglob(pattern))

        matches = sorted(
            {
                path.resolve()
                for path in matches
                if path.exists()
                and (
                    path.is_dir()
                    or (path.is_file() and path.stat().st_size > 0)
                )
            },
            key=lambda path: (
                0 if path.is_dir() else 1,
                len(path.parts),
                len(path.name),
                str(path).casefold(),
            ),
        )

        if matches:
            selected = matches[0]
            print(f"[auto-discovery] {label}: {selected}")
            return selected

    raise FileNotFoundError(f"Could not locate {label}.")


class ShardedNpyMatrix:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.files = sorted(directory.glob("part-*.npy"))
        if not self.files:
            self.files = sorted(directory.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(
                f"No NPY shards found in {directory}"
            )

        self.arrays = [
            np.load(path, mmap_mode="r")
            for path in self.files
        ]

        dimensions = {
            int(array.shape[1])
            for array in self.arrays
            if array.ndim == 2
        }
        if len(dimensions) != 1:
            raise ValueError(
                f"Inconsistent content embedding dimensions: {dimensions}"
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
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)

        shard_index = int(
            np.searchsorted(
                self.offsets,
                index,
                side="right",
            )
            - 1
        )
        local_index = int(index - self.offsets[shard_index])
        return np.asarray(
            self.arrays[shard_index][local_index],
            dtype=np.float32,
        )


def load_matrix(path: Path) -> Any:
    if path.is_dir():
        return ShardedNpyMatrix(path)

    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r")

    raise ValueError(f"Unsupported embedding source: {path}")


def discover_inputs(
    processed_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    recommendations_dir = processed_dir / "recommendations"
    user_profile_dir = processed_dir / "user_profile"
    content_dir = processed_dir / "content"

    return {
        "recommendations": (
            resolve_path(args.recommendations)
            if args.recommendations
            else first_existing(
                [
                    recommendations_dir / "recommendations.parquet",
                    recommendations_dir / "recommendations.csv",
                ],
                "recommendations",
                recursive_root=processed_dir,
                patterns=(
                    "recommendations.parquet",
                    "recommendations.csv",
                ),
            )
        ),
        "profile_books": (
            resolve_path(args.profile_books)
            if args.profile_books
            else first_existing(
                [
                    user_profile_dir / "profile_books_used.parquet",
                    processed_dir
                    / "user_library"
                    / "profile_books.parquet",
                ],
                "profile books",
                recursive_root=processed_dir,
                patterns=(
                    "profile_books_used.parquet",
                    "profile_books.parquet",
                ),
            )
        ),
        "content_embeddings": (
            resolve_path(args.content_embeddings)
            if args.content_embeddings
            else first_existing(
                [
                    content_dir / "index" / "embeddings",
                    content_dir / "embeddings",
                    content_dir / "content_embeddings.npy",
                ],
                "content embeddings",
                recursive_root=processed_dir,
                patterns=(
                    "embeddings",
                    "*content*embedding*.npy",
                    "*svd*embedding*.npy",
                ),
            )
        ),
    }


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format: {path}")


def safe_int(value: Any) -> int | None:
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


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= 0:
        return None

    return vector / norm


def prepare_profile_books(
    frame: pd.DataFrame,
    minimum_weight: float,
) -> pd.DataFrame:
    profile = frame.copy()

    weight_column = (
        "effective_weight"
        if "effective_weight" in profile.columns
        else "preference_weight"
    )

    if weight_column not in profile.columns:
        raise ValueError(
            "Profile books must contain effective_weight or "
            "preference_weight."
        )

    profile["explanation_weight"] = pd.to_numeric(
        profile[weight_column],
        errors="coerce",
    ).fillna(0.0)

    profile = profile[
        profile["explanation_weight"] >= minimum_weight
    ].copy()

    if "match_status" in profile.columns:
        profile = profile[
            profile["match_status"] == "matched"
        ].copy()

    profile["content_global_row"] = pd.to_numeric(
        profile["content_global_row"],
        errors="coerce",
    ).astype("Int64")

    profile = profile.dropna(
        subset=["content_global_row"]
    ).copy()

    if profile.empty:
        raise ValueError(
            "No positively weighted profile books remain for explanations."
        )

    profile = profile.sort_values(
        "explanation_weight",
        ascending=False,
    ).reset_index(drop=True)

    return profile


def build_profile_vector_cache(
    profile: pd.DataFrame,
    matrix: Any,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []

    for _, row in profile.iterrows():
        index = safe_int(row.get("content_global_row"))
        if index is None or index < 0 or index >= matrix.shape[0]:
            continue

        vector = normalize_vector(matrix[index])
        if vector is None:
            continue

        rows.append(row.to_dict())
        vectors.append(vector)

    if not vectors:
        raise ValueError(
            "No valid profile content vectors were available."
        )

    return (
        pd.DataFrame(rows).reset_index(drop=True),
        np.vstack(vectors).astype(np.float32),
    )


def evidence_for_recommendation(
    recommendation: pd.Series,
    profile: pd.DataFrame,
    profile_vectors: np.ndarray,
    matrix: Any,
    evidence_books: int,
    minimum_similarity: float,
    global_usage: dict[int, int],
    max_evidence_uses: int,
    genre_bonus: float,
    author_bonus: float,
    diversity_bonus: float,
) -> list[dict[str, Any]]:
    index = safe_int(recommendation.get("content_global_row"))
    if index is None or index < 0 or index >= matrix.shape[0]:
        return []

    recommendation_vector = normalize_vector(matrix[index])
    if recommendation_vector is None:
        return []

    similarities = profile_vectors @ recommendation_vector
    weights = pd.to_numeric(
        profile["explanation_weight"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)

    base_scores = similarities * np.sqrt(
        np.maximum(weights, 0.0)
    )

    recommendation_genre = recommendation.get("primary_genre")
    recommendation_author = recommendation.get("primary_author")

    candidate_rows: list[dict[str, Any]] = []

    for position in range(len(profile)):
        similarity = float(similarities[position])
        if similarity < minimum_similarity:
            continue

        row = profile.iloc[position]
        library_row_id = safe_int(row.get("library_row_id"))
        usage_key = (
            library_row_id
            if library_row_id is not None
            else int(position)
        )

        if (
            max_evidence_uses > 0
            and global_usage.get(usage_key, 0) >= max_evidence_uses
        ):
            continue

        shared_genre_flag = shared_value(
            recommendation_genre,
            row.get("matched_primary_genre")
            or row.get("primary_genre"),
        )
        shared_author_flag = shared_value(
            recommendation_author,
            row.get("matched_author")
            or row.get("input_author"),
        )

        score = float(base_scores[position])
        if shared_genre_flag:
            score += genre_bonus
        if shared_author_flag:
            score += author_bonus

        candidate_rows.append(
            {
                "position": int(position),
                "usage_key": usage_key,
                "score": score,
                "shared_genre": shared_genre_flag,
                "shared_author": shared_author_flag,
            }
        )

    candidate_rows.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    evidence: list[dict[str, Any]] = []
    used_authors: set[str] = set()

    while candidate_rows and len(evidence) < evidence_books:
        best_index = None
        best_adjusted_score = -float("inf")

        for idx, candidate in enumerate(candidate_rows):
            row = profile.iloc[candidate["position"]]
            author_key = normalized_text(
                row.get("matched_author")
                or row.get("input_author")
            )

            adjusted_score = candidate["score"]
            if author_key and author_key not in used_authors:
                adjusted_score += diversity_bonus

            usage_count = global_usage.get(
                candidate["usage_key"],
                0,
            )
            adjusted_score -= 0.02 * usage_count

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = idx

        if best_index is None:
            break

        candidate = candidate_rows.pop(best_index)
        position = candidate["position"]
        row = profile.iloc[position]

        author = (
            row.get("input_author")
            or row.get("matched_author")
            or "Unknown author"
        )
        author_key = normalized_text(author)
        if author_key:
            used_authors.add(author_key)

        global_usage[candidate["usage_key"]] = (
            global_usage.get(candidate["usage_key"], 0) + 1
        )

        evidence.append(
            {
                "title": (
                    row.get("input_title")
                    or row.get("matched_title")
                    or "Unknown title"
                ),
                "author": author,
                "rating": safe_float(row.get("my_rating")),
                "weight": float(
                    row.get("explanation_weight", 0.0)
                ),
                "content_similarity": float(
                    similarities[position]
                ),
                "evidence_score": float(
                    candidate["score"]
                ),
                "shared_genre": candidate["shared_genre"],
                "shared_author": candidate["shared_author"],
                "global_usage_count": global_usage[
                    candidate["usage_key"]
                ],
            }
        )

    return evidence


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip().casefold()


def shared_value(left: Any, right: Any) -> bool:
    left_text = normalized_text(left)
    right_text = normalized_text(right)
    return bool(left_text and right_text and left_text == right_text)


def format_title_author(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "Unknown title").strip()
    author = str(item.get("author") or "").strip()

    if author and author.casefold() != "unknown author":
        return f"{title} by {author}"

    return title


def recommendation_explanation(
    recommendation: pd.Series,
    evidence: list[dict[str, Any]],
    include_model_details: bool,
) -> str:
    title = str(recommendation.get("title") or "This book")
    author = str(
        recommendation.get("primary_author")
        or "an unknown author"
    )
    genre = str(recommendation.get("primary_genre") or "").strip()

    reasons: list[str] = []

    if evidence:
        cited = ", ".join(
            format_title_author(item)
            for item in evidence[:3]
        )
        reasons.append(
            f"It is close in content to books you liked, especially {cited}"
        )

    shared_genre_books = [
        item
        for item in evidence
        if item.get("shared_genre")
    ]
    if genre and shared_genre_books:
        reasons.append(
            f"it reinforces your apparent interest in {genre}"
        )
    elif genre:
        reasons.append(
            f"it fits the {genre} part of your reading profile"
        )

    shared_author_books = [
        item
        for item in evidence
        if item.get("shared_author")
    ]
    if shared_author_books:
        reasons.append(
            f"you have already shown interest in {author}"
        )

    rating = None
    for column in [
        "bayesian_rating",
        "average_rating",
        "wilson_lower_bound",
    ]:
        rating = safe_float(recommendation.get(column))
        if rating is not None:
            break

    ratings_count = safe_int(
        recommendation.get("ratings_count")
    )

    if rating is not None and rating >= 4.0:
        reasons.append(
            f"it also has a strong reader rating of {rating:.2f}"
        )
    elif ratings_count is not None and ratings_count >= 10000:
        reasons.append(
            "it has substantial reader support"
        )

    if include_model_details:
        similarity = safe_float(
            recommendation.get("hybrid_similarity")
        )
        score = safe_float(
            recommendation.get("recommendation_score")
        )

        numeric_parts = []
        if similarity is not None:
            numeric_parts.append(
                f"hybrid similarity {similarity:.3f}"
            )
        if score is not None:
            numeric_parts.append(
                f"final score {score:.3f}"
            )

        if numeric_parts:
            reasons.append(
                "model evidence: " + ", ".join(numeric_parts)
            )

    if not reasons:
        return (
            f"{title} by {author} was selected because it is one of the "
            "closest books to your combined content and collaborative profile."
        )

    explanation = "; ".join(reasons)
    return explanation[0].upper() + explanation[1:] + "."


def concise_reason(
    recommendation: pd.Series,
    evidence: list[dict[str, Any]],
    short_evidence_books: int,
) -> str:
    if evidence:
        selected = evidence[:short_evidence_books]
        cited = " and ".join(
            format_title_author(item)
            for item in selected
        )
        similarities = ", ".join(
            f"{item['content_similarity']:.2f}"
            for item in selected
        )
        return (
            f"Related to {cited} "
            f"(content similarities: {similarities})"
        )

    genre = recommendation.get("primary_genre")
    if isinstance(genre, str) and genre.strip():
        return f"Strong hybrid match in {genre}"

    return "Strong hybrid-profile match"


def build_explanations(
    recommendations: pd.DataFrame,
    profile: pd.DataFrame,
    profile_vectors: np.ndarray,
    matrix: Any,
    args: argparse.Namespace,
) -> pd.DataFrame:
    explained_rows: list[dict[str, Any]] = []
    global_usage: dict[int, int] = defaultdict(int)

    for _, recommendation in recommendations.head(
        args.top_n
    ).iterrows():
        evidence = evidence_for_recommendation(
            recommendation=recommendation,
            profile=profile,
            profile_vectors=profile_vectors,
            matrix=matrix,
            evidence_books=args.evidence_books,
            minimum_similarity=args.minimum_content_similarity,
            global_usage=global_usage,
            max_evidence_uses=args.max_evidence_uses,
            genre_bonus=args.genre_bonus,
            author_bonus=args.author_bonus,
            diversity_bonus=args.diversity_bonus,
        )

        output = recommendation.to_dict()
        output["short_explanation"] = concise_reason(
            recommendation,
            evidence,
            args.short_evidence_books,
        )
        output["explanation"] = recommendation_explanation(
            recommendation,
            evidence,
            args.include_model_details,
        )
        output["evidence_books"] = json.dumps(
            evidence,
            ensure_ascii=False,
        )
        output["evidence_book_count"] = len(evidence)
        output["strongest_evidence_similarity"] = (
            evidence[0]["content_similarity"]
            if evidence
            else None
        )
        explained_rows.append(output)

    return pd.DataFrame(explained_rows)


def write_markdown_report(
    path: Path,
    explained: pd.DataFrame,
) -> None:
    lines = [
        "# Personalized Book Recommendations",
        "",
        (
            f"Generated: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        "",
    ]

    for _, row in explained.iterrows():
        rank = safe_int(row.get("rank"))
        title = str(row.get("title") or "Unknown title")
        author = str(
            row.get("primary_author")
            or "Unknown author"
        )
        genre = str(row.get("primary_genre") or "").strip()

        heading = (
            f"## {rank}. {title}"
            if rank is not None
            else f"## {title}"
        )
        lines.extend(
            [
                heading,
                "",
                f"**Author:** {author}",
            ]
        )

        if genre:
            lines.append(f"**Genre:** {genre}")

        score = safe_float(
            row.get("recommendation_score")
        )
        similarity = safe_float(
            row.get("hybrid_similarity")
        )

        score_parts = []
        if score is not None:
            score_parts.append(
                f"recommendation score {score:.3f}"
            )
        if similarity is not None:
            score_parts.append(
                f"hybrid similarity {similarity:.3f}"
            )

        if score_parts:
            lines.append(
                "**Model:** " + ", ".join(score_parts)
            )

        lines.extend(
            [
                "",
                str(row.get("explanation") or ""),
                "",
            ]
        )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")
    if args.evidence_books <= 0:
        raise ValueError("--evidence-books must be positive.")
    if args.short_evidence_books <= 0:
        raise ValueError("--short-evidence-books must be positive.")
    if args.short_evidence_books > args.evidence_books:
        raise ValueError(
            "--short-evidence-books cannot exceed --evidence-books."
        )
    if args.max_evidence_uses < 0:
        raise ValueError("--max-evidence-uses cannot be negative.")
    if args.minimum_profile_weight < 0:
        raise ValueError(
            "--minimum-profile-weight cannot be negative."
        )
    if not -1 <= args.minimum_content_similarity <= 1:
        raise ValueError(
            "--minimum-content-similarity must be between -1 and 1."
        )

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.processed_dir
        or config["paths"]["processed_dir"]
    )
    output_dir = resolve_path(
        args.output_dir
        or (processed_dir / "recommendation_explanations")
    )

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Use --overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(processed_dir, args)

    recommendations = read_table(
        inputs["recommendations"]
    )
    if recommendations.empty:
        raise ValueError("Recommendations input is empty.")

    profile_raw = read_table(inputs["profile_books"])
    profile = prepare_profile_books(
        profile_raw,
        args.minimum_profile_weight,
    )

    content_matrix = load_matrix(
        inputs["content_embeddings"]
    )

    profile, profile_vectors = build_profile_vector_cache(
        profile,
        content_matrix,
    )

    explained = build_explanations(
        recommendations=recommendations,
        profile=profile,
        profile_vectors=profile_vectors,
        matrix=content_matrix,
        args=args,
    )

    output_parquet = output_dir / "explained_recommendations.parquet"
    output_csv = output_dir / "explained_recommendations.csv"
    output_json = output_dir / "explained_recommendations.json"
    output_markdown = output_dir / "recommendation_explanations.md"
    output_report = output_dir / "explanation_report.json"

    explained.to_parquet(
        output_parquet,
        index=False,
        compression="zstd",
    )
    explained.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    output_json.write_text(
        explained.to_json(
            orient="records",
            force_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(
        output_markdown,
        explained,
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendations_explained": int(len(explained)),
        "profile_evidence_books": int(len(profile)),
        "content_dimensions": int(content_matrix.shape[1]),
        "settings": {
            "top_n": args.top_n,
            "evidence_books": args.evidence_books,
            "short_evidence_books": args.short_evidence_books,
            "max_evidence_uses": args.max_evidence_uses,
            "genre_bonus": args.genre_bonus,
            "author_bonus": args.author_bonus,
            "diversity_bonus": args.diversity_bonus,
            "minimum_profile_weight": (
                args.minimum_profile_weight
            ),
            "minimum_content_similarity": (
                args.minimum_content_similarity
            ),
            "include_model_details": (
                args.include_model_details
            ),
        },
        "inputs": {
            key: str(value)
            for key, value in inputs.items()
        },
        "outputs": {
            "parquet": str(output_parquet),
            "csv": str(output_csv),
            "json": str(output_json),
            "markdown": str(output_markdown),
        },
    }

    output_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("RECOMMENDATION EXPLANATIONS COMPLETE")
    print(f"Recommendations explained : {len(explained):,}")
    print(f"Profile evidence books    : {len(profile):,}")
    print(f"Content dimensions        : {content_matrix.shape[1]:,}")
    print(f"Output directory          : {output_dir}")
    print("=" * 76)

    display_columns = [
        column
        for column in [
            "rank",
            "title",
            "primary_author",
            "short_explanation",
        ]
        if column in explained.columns
    ]

    print(
        explained[display_columns]
        .head(min(20, len(explained)))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
