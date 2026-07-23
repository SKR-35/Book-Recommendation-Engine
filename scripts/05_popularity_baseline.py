from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build popularity baselines from the filtered Goodreads catalog "
            "using Bayesian rating and Wilson lower bound scores."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--group-top-n", type=int, default=100)
    parser.add_argument("--min-overall-ratings", type=int, default=50)
    parser.add_argument("--min-group-ratings", type=int, default=20)
    parser.add_argument("--hidden-gem-min-rating", type=float, default=4.2)
    parser.add_argument("--hidden-gem-min-count", type=int, default=50)
    parser.add_argument("--hidden-gem-max-count", type=int, default=500)
    parser.add_argument("--safe-choice-min-rating", type=float, default=4.1)
    parser.add_argument("--safe-choice-min-count", type=int, default=50_000)
    parser.add_argument("--wilson-confidence", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    return f"{value:,.2f} TB"


def format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def z_score(confidence: float) -> float:
    lookup = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }
    rounded = round(confidence, 2)
    if rounded not in lookup:
        raise ValueError(
            "--wilson-confidence must be one of: 0.80, 0.90, 0.95, 0.98, 0.99"
        )
    return lookup[rounded]


def execute(
    connection: duckdb.DuckDBPyConnection,
    title: str,
    sql: str,
) -> float:
    print("\n" + "=" * 76)
    print(title)
    started = time.perf_counter()
    connection.execute(sql)
    elapsed = time.perf_counter() - started
    print(f"Completed in {format_elapsed(elapsed)}")
    return elapsed


def require_inputs(processed_dir: Path) -> dict[str, Path]:
    inputs = {
        "catalog": processed_dir / "catalog",
        "book_stats": processed_dir / "book_stats.parquet",
    }

    missing: list[str] = []
    if not inputs["catalog"].exists():
        missing.append(str(inputs["catalog"]))
    if not inputs["book_stats"].exists():
        missing.append(str(inputs["book_stats"]))

    if missing:
        raise FileNotFoundError(
            "Required inputs are missing:\n  - " + "\n  - ".join(missing)
        )

    if not any(inputs["catalog"].glob("*.parquet")):
        raise FileNotFoundError(
            f"No catalog parts found in {inputs['catalog']}"
        )

    return inputs


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    popularity_dir = output_dir / "popularity"
    outputs = {
        "directory": popularity_dir,
        "scored_books": popularity_dir / "scored_books.parquet",
        "overall": popularity_dir / "overall.parquet",
        "by_genre": popularity_dir / "by_genre.parquet",
        "by_language": popularity_dir / "by_language.parquet",
        "hidden_gems": popularity_dir / "hidden_gems.parquet",
        "safe_choices": popularity_dir / "safe_choices.parquet",
        "author_rankings": popularity_dir / "author_rankings.parquet",
        "metadata": popularity_dir / "popularity_metadata.json",
    }

    if popularity_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{popularity_dir} already exists. Use --overwrite."
            )
        shutil.rmtree(popularity_dir)

    popularity_dir.mkdir(parents=True, exist_ok=True)
    return outputs


def scoring_parameters(
    connection: duckdb.DuckDBPyConnection,
    catalog_glob: str,
) -> dict[str, float]:
    global_mean, minimum_votes = connection.execute(
        f"""
        SELECT
            AVG(TRY_CAST(average_rating AS DOUBLE)),
            QUANTILE_CONT(
                TRY_CAST(ratings_count AS DOUBLE),
                0.90
            )
        FROM read_parquet('{catalog_glob}')
        WHERE TRY_CAST(average_rating AS DOUBLE) BETWEEN 0 AND 5
          AND TRY_CAST(ratings_count AS BIGINT) > 0
        """
    ).fetchone()

    return {
        "global_mean_rating": float(global_mean or 0.0),
        "minimum_votes": max(float(minimum_votes or 1.0), 1.0),
    }


def build_scored_books(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    parameters: dict[str, float],
    z: float,
) -> float:
    catalog_glob = sql_path(inputs["catalog"] / "*.parquet")
    global_mean = parameters["global_mean_rating"]
    minimum_votes = parameters["minimum_votes"]

    query = f"""
    COPY (
        SELECT
            c.book_id,
            c.work_id,
            c.interaction_book_id,
            c.title,
            c.primary_author_id,
            c.primary_author,
            c.author_names,
            c.primary_genre,
            c.genre_names,
            c.language_code,
            c.publication_year,
            c.average_rating,
            c.ratings_count,
            c.text_reviews_count,
            c.is_canonical_edition,
            c.novelty_score,
            c.image_url,
            s.interaction_count,
            s.rating_count AS observed_rating_count,
            s.average_user_rating,
            CASE
                WHEN c.average_rating IS NULL OR c.ratings_count IS NULL
                THEN NULL
                ELSE (
                    c.ratings_count / (c.ratings_count + {minimum_votes})
                ) * c.average_rating
                + (
                    {minimum_votes} / (c.ratings_count + {minimum_votes})
                ) * {global_mean}
            END AS bayesian_score,
            CASE
                WHEN s.rating_count IS NULL OR s.rating_count <= 0
                THEN NULL
                ELSE (
                    (
                        (s.average_user_rating / 5.0)
                        + ({z} * {z}) / (2.0 * s.rating_count)
                        - {z} * SQRT(
                            (
                                (s.average_user_rating / 5.0)
                                * (1.0 - s.average_user_rating / 5.0)
                                + ({z} * {z}) / (4.0 * s.rating_count)
                            )
                            / s.rating_count
                        )
                    )
                    /
                    (
                        1.0 + ({z} * {z}) / s.rating_count
                    )
                )
            END AS wilson_score
        FROM read_parquet('{catalog_glob}') c
        LEFT JOIN read_parquet('{sql_path(inputs["book_stats"])}') s
            ON c.interaction_book_id = s.item_id
        WHERE c.is_canonical_edition = TRUE
    )
    TO '{sql_path(outputs["scored_books"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING SCORED BOOK BASE", query)


def build_overall(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
    top_n: int,
    min_ratings: int,
) -> float:
    query = f"""
    COPY (
        SELECT
            *,
            ROW_NUMBER() OVER (
                ORDER BY
                    bayesian_score DESC NULLS LAST,
                    wilson_score DESC NULLS LAST,
                    ratings_count DESC NULLS LAST,
                    book_id
            ) AS rank
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        WHERE COALESCE(ratings_count, 0) >= {min_ratings}
        QUALIFY rank <= {top_n}
    )
    TO '{sql_path(outputs["overall"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING OVERALL POPULARITY LIST", query)


def build_by_genre(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
    top_n: int,
    min_ratings: int,
) -> float:
    query = f"""
    COPY (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY primary_genre
                ORDER BY
                    bayesian_score DESC NULLS LAST,
                    wilson_score DESC NULLS LAST,
                    ratings_count DESC NULLS LAST,
                    book_id
            ) AS genre_rank
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        WHERE primary_genre IS NOT NULL
          AND COALESCE(ratings_count, 0) >= {min_ratings}
        QUALIFY genre_rank <= {top_n}
    )
    TO '{sql_path(outputs["by_genre"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING GENRE POPULARITY LISTS", query)


def build_by_language(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
    top_n: int,
    min_ratings: int,
) -> float:
    query = f"""
    COPY (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY language_code
                ORDER BY
                    bayesian_score DESC NULLS LAST,
                    wilson_score DESC NULLS LAST,
                    ratings_count DESC NULLS LAST,
                    book_id
            ) AS language_rank
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        WHERE language_code IS NOT NULL
          AND TRIM(language_code) <> ''
          AND COALESCE(ratings_count, 0) >= {min_ratings}
        QUALIFY language_rank <= {top_n}
    )
    TO '{sql_path(outputs["by_language"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING LANGUAGE POPULARITY LISTS", query)


def build_hidden_gems(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
    minimum_rating: float,
    minimum_count: int,
    maximum_count: int,
    top_n: int,
) -> float:
    query = f"""
    COPY (
        SELECT
            *,
            ROW_NUMBER() OVER (
                ORDER BY
                    wilson_score DESC NULLS LAST,
                    bayesian_score DESC NULLS LAST,
                    novelty_score DESC NULLS LAST,
                    ratings_count ASC,
                    book_id
            ) AS hidden_gem_rank
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        WHERE average_rating >= {minimum_rating}
          AND ratings_count BETWEEN {minimum_count} AND {maximum_count}
        QUALIFY hidden_gem_rank <= {top_n}
    )
    TO '{sql_path(outputs["hidden_gems"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING HIDDEN GEMS LIST", query)


def build_safe_choices(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
    minimum_rating: float,
    minimum_count: int,
    top_n: int,
) -> float:
    query = f"""
    COPY (
        SELECT
            *,
            ROW_NUMBER() OVER (
                ORDER BY
                    wilson_score DESC NULLS LAST,
                    bayesian_score DESC NULLS LAST,
                    ratings_count DESC NULLS LAST,
                    book_id
            ) AS safe_choice_rank
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        WHERE average_rating >= {minimum_rating}
          AND ratings_count >= {minimum_count}
        QUALIFY safe_choice_rank <= {top_n}
    )
    TO '{sql_path(outputs["safe_choices"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING SAFE CHOICES LIST", query)


def build_author_rankings(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
) -> float:
    query = f"""
    COPY (
        SELECT
            primary_author_id,
            primary_author,
            COUNT(*) AS canonical_book_count,
            SUM(COALESCE(ratings_count, 0)) AS total_ratings,
            MEDIAN(average_rating) AS median_rating,
            AVG(average_rating) AS mean_rating,
            AVG(bayesian_score) AS mean_bayesian_score,
            AVG(wilson_score) AS mean_wilson_score,
            MAX(bayesian_score) AS best_book_bayesian_score,
            ARG_MAX(title, bayesian_score) AS best_ranked_book
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        WHERE primary_author_id IS NOT NULL
          AND primary_author IS NOT NULL
        GROUP BY primary_author_id, primary_author
        HAVING COUNT(*) >= 2
        ORDER BY
            mean_bayesian_score DESC NULLS LAST,
            total_ratings DESC,
            primary_author
    )
    TO '{sql_path(outputs["author_rankings"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING AUTHOR RANKINGS", query)


def validate_outputs(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    validation: dict[str, Any] = {}

    for name in (
        "scored_books",
        "overall",
        "by_genre",
        "by_language",
        "hidden_gems",
        "safe_choices",
        "author_rankings",
    ):
        path = outputs[name]
        parquet = pq.ParquetFile(path)
        validation[name] = {
            "path": str(path),
            "rows": parquet.metadata.num_rows,
            "columns": len(parquet.schema.names),
            "size_bytes": path.stat().st_size,
            "size_human": human_size(path.stat().st_size),
        }

    score_checks = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT_IF(bayesian_score IS NULL) AS missing_bayesian,
            COUNT_IF(wilson_score IS NULL) AS missing_wilson,
            MIN(wilson_score) AS min_wilson,
            MAX(wilson_score) AS max_wilson,
            MIN(bayesian_score) AS min_bayesian,
            MAX(bayesian_score) AS max_bayesian
        FROM read_parquet('{sql_path(outputs["scored_books"])}')
        """
    ).fetchone()

    validation["score_checks"] = {
        "rows": int(score_checks[0]),
        "missing_bayesian": int(score_checks[1]),
        "missing_wilson": int(score_checks[2]),
        "minimum_wilson": (
            float(score_checks[3]) if score_checks[3] is not None else None
        ),
        "maximum_wilson": (
            float(score_checks[4]) if score_checks[4] is not None else None
        ),
        "minimum_bayesian": (
            float(score_checks[5]) if score_checks[5] is not None else None
        ),
        "maximum_bayesian": (
            float(score_checks[6]) if score_checks[6] is not None else None
        ),
    }

    if score_checks[3] is not None and float(score_checks[3]) < 0:
        raise RuntimeError("Wilson score below zero.")
    if score_checks[4] is not None and float(score_checks[4]) > 1:
        raise RuntimeError("Wilson score above one.")

    return validation


def main() -> None:
    args = parse_args()

    if args.top_n <= 0 or args.group_top_n <= 0:
        raise ValueError("Top-N values must be positive.")
    if args.min_overall_ratings < 0 or args.min_group_ratings < 0:
        raise ValueError("Minimum rating counts cannot be negative.")
    if args.hidden_gem_min_count > args.hidden_gem_max_count:
        raise ValueError(
            "--hidden-gem-min-count cannot exceed --hidden-gem-max-count."
        )

    z = z_score(args.wilson_confidence)

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.input_dir or config["paths"]["processed_dir"]
    )
    output_dir = resolve_path(
        args.output_dir or config["paths"]["processed_dir"]
    )

    inputs = require_inputs(processed_dir)
    outputs = prepare_outputs(output_dir, args.overwrite)

    work_db = outputs["directory"] / ".popularity_work.duckdb"
    temp_dir = outputs["directory"] / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    work_db.unlink(missing_ok=True)

    print("=" * 76)
    print("GOODREADS POPULARITY BASELINE")
    print(f"Catalog              : {inputs['catalog']}")
    print(f"Book stats           : {inputs['book_stats']}")
    print(f"Output               : {outputs['directory']}")
    print(f"Overall Top-N        : {args.top_n:,}")
    print(f"Group Top-N          : {args.group_top_n:,}")
    print(f"Wilson confidence    : {args.wilson_confidence:.0%}")
    print(f"Threads              : {args.threads}")
    print(f"Memory limit         : {args.memory_limit}")
    print("=" * 76)

    started = time.perf_counter()
    connection = duckdb.connect(str(work_db))
    connection.execute(f"SET threads = {max(args.threads, 1)}")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order = false")

    timings: dict[str, float] = {}

    try:
        catalog_glob = sql_path(inputs["catalog"] / "*.parquet")
        parameters = scoring_parameters(connection, catalog_glob)

        timings["scored_books_seconds"] = build_scored_books(
            connection,
            inputs,
            outputs,
            parameters,
            z,
        )
        timings["overall_seconds"] = build_overall(
            connection,
            outputs,
            args.top_n,
            args.min_overall_ratings,
        )
        timings["by_genre_seconds"] = build_by_genre(
            connection,
            outputs,
            args.group_top_n,
            args.min_group_ratings,
        )
        timings["by_language_seconds"] = build_by_language(
            connection,
            outputs,
            args.group_top_n,
            args.min_group_ratings,
        )
        timings["hidden_gems_seconds"] = build_hidden_gems(
            connection,
            outputs,
            args.hidden_gem_min_rating,
            args.hidden_gem_min_count,
            args.hidden_gem_max_count,
            args.top_n,
        )
        timings["safe_choices_seconds"] = build_safe_choices(
            connection,
            outputs,
            args.safe_choice_min_rating,
            args.safe_choice_min_count,
            args.top_n,
        )
        timings["author_rankings_seconds"] = build_author_rankings(
            connection,
            outputs,
        )

        print("\n" + "=" * 76)
        print("VALIDATING OUTPUTS")
        validation = validate_outputs(connection, outputs)

    finally:
        connection.close()
        work_db.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_seconds = time.perf_counter() - started
    timings["total_seconds"] = round(total_seconds, 3)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "popularity_baseline",
        "input_catalog": str(inputs["catalog"]),
        "input_book_stats": str(inputs["book_stats"]),
        "output_directory": str(outputs["directory"]),
        "settings": {
            "top_n": args.top_n,
            "group_top_n": args.group_top_n,
            "min_overall_ratings": args.min_overall_ratings,
            "min_group_ratings": args.min_group_ratings,
            "hidden_gem_min_rating": args.hidden_gem_min_rating,
            "hidden_gem_min_count": args.hidden_gem_min_count,
            "hidden_gem_max_count": args.hidden_gem_max_count,
            "safe_choice_min_rating": args.safe_choice_min_rating,
            "safe_choice_min_count": args.safe_choice_min_count,
            "wilson_confidence": args.wilson_confidence,
            "wilson_z_score": z,
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "overwrite": args.overwrite,
        },
        "scoring_parameters": parameters,
        "timings": timings,
        "validation": validation,
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("POPULARITY BASELINE COMPLETE")
    print(
        f"Scored books   : "
        f"{validation['scored_books']['rows']:,}"
    )
    print(
        f"Overall list   : "
        f"{validation['overall']['rows']:,}"
    )
    print(
        f"Genre rows     : "
        f"{validation['by_genre']['rows']:,}"
    )
    print(
        f"Language rows  : "
        f"{validation['by_language']['rows']:,}"
    )
    print(
        f"Hidden gems    : "
        f"{validation['hidden_gems']['rows']:,}"
    )
    print(
        f"Safe choices   : "
        f"{validation['safe_choices']['rows']:,}"
    )
    print(
        f"Authors ranked : "
        f"{validation['author_rankings']['rows']:,}"
    )
    print(f"Total elapsed  : {format_elapsed(total_seconds)}")
    print(f"Metadata       : {outputs['metadata']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
