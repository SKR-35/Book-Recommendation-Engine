from __future__ import annotations

import argparse
import json
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
            "Filter the Goodreads interaction graph with low-memory DuckDB "
            "queries and write a model-ready partitioned Parquet dataset."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-book-interactions", type=int, default=30)
    parser.add_argument("--min-user-interactions", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--partitions", type=int, default=64)
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


def require_inputs(
    interim_dir: Path,
    processed_dir: Path,
) -> dict[str, Path]:
    paths = {
        "interactions": interim_dir / "interactions",
        "catalog": processed_dir / "catalog",
    }

    missing: list[str] = []
    if not paths["interactions"].exists():
        missing.append(str(paths["interactions"]))
    if not paths["catalog"].exists():
        missing.append(str(paths["catalog"]))

    if missing:
        raise FileNotFoundError(
            "Required inputs are missing:\n  - " + "\n  - ".join(missing)
        )

    if not any(paths["interactions"].glob("*.parquet")):
        raise FileNotFoundError(
            f"No interaction Parquet parts found in {paths['interactions']}"
        )
    if not any(paths["catalog"].glob("*.parquet")):
        raise FileNotFoundError(
            f"No catalog Parquet parts found in {paths['catalog']}"
        )
    return paths


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "catalog_map": output_dir / "catalog_interaction_map.parquet",
        "filtered": output_dir / "interactions_filtered",
        "user_stats": output_dir / "user_stats.parquet",
        "book_stats": output_dir / "book_stats.parquet",
        "metadata": output_dir / "interaction_filter_metadata.json",
    }

    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Filtered interaction outputs already exist. Run again with --overwrite."
        )

    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    outputs["filtered"].mkdir(parents=True, exist_ok=True)
    return outputs


def build_catalog_map(
    connection: duckdb.DuckDBPyConnection,
    catalog_dir: Path,
    output_path: Path,
) -> float:
    catalog_glob = sql_path(catalog_dir / "*.parquet")
    query = f"""
    COPY (
        SELECT
            TRY_CAST(interaction_book_id AS INTEGER) AS item_id,
            TRY_CAST(book_id AS BIGINT) AS local_book_id,
            TRY_CAST(work_id AS BIGINT) AS work_id,
            title,
            primary_author,
            primary_genre,
            TRY_CAST(is_canonical_edition AS BOOLEAN) AS is_canonical_edition,
            TRY_CAST(bayesian_rating AS DOUBLE) AS bayesian_rating,
            TRY_CAST(ratings_count AS BIGINT) AS ratings_count
        FROM read_parquet('{catalog_glob}')
        WHERE interaction_book_id IS NOT NULL
    )
    TO '{sql_path(output_path)}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING NARROW CATALOG MAP", query)


def build_initial_book_set(
    connection: duckdb.DuckDBPyConnection,
    interactions_glob: str,
    catalog_map: Path,
    minimum: int,
) -> float:
    query = f"""
    CREATE OR REPLACE TABLE eligible_books AS
    SELECT
        TRY_CAST(i.book_id AS INTEGER) AS item_id,
        COUNT(*) AS interaction_count,
        COUNT_IF(TRY_CAST(i.rating AS INTEGER) > 0) AS rating_count,
        AVG(
            CASE WHEN TRY_CAST(i.rating AS INTEGER) > 0
                 THEN TRY_CAST(i.rating AS DOUBLE) END
        ) AS average_user_rating
    FROM read_parquet('{interactions_glob}') i
    INNER JOIN read_parquet('{sql_path(catalog_map)}') m
        ON TRY_CAST(i.book_id AS INTEGER) = m.item_id
    WHERE TRY_CAST(i.is_read AS INTEGER) = 1
       OR TRY_CAST(i.rating AS INTEGER) > 0
    GROUP BY TRY_CAST(i.book_id AS INTEGER)
    HAVING COUNT(*) >= {minimum};
    """
    return execute(connection, "BUILDING INITIAL ELIGIBLE BOOK SET", query)


def build_user_set(
    connection: duckdb.DuckDBPyConnection,
    interactions_glob: str,
    minimum: int,
) -> float:
    query = f"""
    CREATE OR REPLACE TABLE eligible_users AS
    SELECT
        TRY_CAST(i.user_id AS INTEGER) AS user_id,
        COUNT(*) AS interaction_count,
        COUNT_IF(TRY_CAST(i.rating AS INTEGER) > 0) AS rating_count,
        AVG(
            CASE WHEN TRY_CAST(i.rating AS INTEGER) > 0
                 THEN TRY_CAST(i.rating AS DOUBLE) END
        ) AS average_rating
    FROM read_parquet('{interactions_glob}') i
    INNER JOIN eligible_books b
        ON TRY_CAST(i.book_id AS INTEGER) = b.item_id
    WHERE TRY_CAST(i.is_read AS INTEGER) = 1
       OR TRY_CAST(i.rating AS INTEGER) > 0
    GROUP BY TRY_CAST(i.user_id AS INTEGER)
    HAVING COUNT(*) >= {minimum};
    """
    return execute(connection, "BUILDING ELIGIBLE USER SET", query)


def refine_book_set(
    connection: duckdb.DuckDBPyConnection,
    interactions_glob: str,
    minimum: int,
) -> float:
    query = f"""
    CREATE OR REPLACE TABLE eligible_books_next AS
    SELECT
        TRY_CAST(i.book_id AS INTEGER) AS item_id,
        COUNT(*) AS interaction_count,
        COUNT_IF(TRY_CAST(i.rating AS INTEGER) > 0) AS rating_count,
        AVG(
            CASE WHEN TRY_CAST(i.rating AS INTEGER) > 0
                 THEN TRY_CAST(i.rating AS DOUBLE) END
        ) AS average_user_rating
    FROM read_parquet('{interactions_glob}') i
    INNER JOIN eligible_users u
        ON TRY_CAST(i.user_id AS INTEGER) = u.user_id
    WHERE TRY_CAST(i.is_read AS INTEGER) = 1
       OR TRY_CAST(i.rating AS INTEGER) > 0
    GROUP BY TRY_CAST(i.book_id AS INTEGER)
    HAVING COUNT(*) >= {minimum};

    DROP TABLE eligible_books;
    ALTER TABLE eligible_books_next RENAME TO eligible_books;
    """
    return execute(connection, "REFINING ELIGIBLE BOOK SET", query)


def refine_user_set(
    connection: duckdb.DuckDBPyConnection,
    interactions_glob: str,
    minimum: int,
) -> float:
    query = f"""
    CREATE OR REPLACE TABLE eligible_users_next AS
    SELECT
        TRY_CAST(i.user_id AS INTEGER) AS user_id,
        COUNT(*) AS interaction_count,
        COUNT_IF(TRY_CAST(i.rating AS INTEGER) > 0) AS rating_count,
        AVG(
            CASE WHEN TRY_CAST(i.rating AS INTEGER) > 0
                 THEN TRY_CAST(i.rating AS DOUBLE) END
        ) AS average_rating
    FROM read_parquet('{interactions_glob}') i
    INNER JOIN eligible_books b
        ON TRY_CAST(i.book_id AS INTEGER) = b.item_id
    WHERE TRY_CAST(i.is_read AS INTEGER) = 1
       OR TRY_CAST(i.rating AS INTEGER) > 0
    GROUP BY TRY_CAST(i.user_id AS INTEGER)
    HAVING COUNT(*) >= {minimum};

    DROP TABLE eligible_users;
    ALTER TABLE eligible_users_next RENAME TO eligible_users;
    """
    return execute(connection, "REFINING ELIGIBLE USER SET", query)


def table_counts(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, int]:
    users = connection.execute("SELECT COUNT(*) FROM eligible_users").fetchone()[0]
    books = connection.execute("SELECT COUNT(*) FROM eligible_books").fetchone()[0]
    return int(users), int(books)


def write_stats(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
) -> dict[str, float]:
    timings: dict[str, float] = {}

    user_query = f"""
    COPY (
        SELECT
            user_id,
            interaction_count,
            rating_count,
            average_rating,
            interaction_count - rating_count AS implicit_read_count
        FROM eligible_users
        ORDER BY user_id
    )
    TO '{sql_path(outputs["user_stats"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    timings["user_stats_seconds"] = execute(
        connection, "WRITING USER STATISTICS", user_query
    )

    book_query = f"""
    COPY (
        SELECT
            b.item_id,
            m.local_book_id,
            m.work_id,
            m.title,
            m.primary_author,
            m.primary_genre,
            m.is_canonical_edition,
            m.bayesian_rating,
            m.ratings_count AS catalog_ratings_count,
            b.interaction_count,
            b.rating_count,
            b.average_user_rating,
            b.interaction_count - b.rating_count AS implicit_read_count
        FROM eligible_books b
        LEFT JOIN read_parquet('{sql_path(outputs["catalog_map"])}') m
            ON b.item_id = m.item_id
        ORDER BY b.interaction_count DESC, b.item_id
    )
    TO '{sql_path(outputs["book_stats"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    timings["book_stats_seconds"] = execute(
        connection, "WRITING BOOK STATISTICS", book_query
    )
    return timings


def write_filtered_interactions(
    connection: duckdb.DuckDBPyConnection,
    interactions_glob: str,
    outputs: dict[str, Path],
    partitions: int,
) -> float:
    query = f"""
    COPY (
        SELECT
            TRY_CAST(i.user_id AS INTEGER) AS user_id,
            TRY_CAST(i.book_id AS INTEGER) AS item_id,
            m.local_book_id,
            m.work_id,
            TRY_CAST(i.rating AS TINYINT) AS rating,
            TRY_CAST(i.is_read AS BOOLEAN) AS is_read,
            TRY_CAST(i.is_reviewed AS BOOLEAN) AS is_reviewed,
            CASE
                WHEN TRY_CAST(i.rating AS INTEGER) = 5 THEN 1.00
                WHEN TRY_CAST(i.rating AS INTEGER) = 4 THEN 0.80
                WHEN TRY_CAST(i.rating AS INTEGER) = 3 THEN 0.50
                WHEN TRY_CAST(i.rating AS INTEGER) = 2 THEN 0.20
                WHEN TRY_CAST(i.rating AS INTEGER) = 1 THEN 0.10
                WHEN TRY_CAST(i.is_read AS INTEGER) = 1 THEN 0.35
                ELSE 0.00
            END::FLOAT AS interaction_weight,
            CASE
                WHEN TRY_CAST(i.rating AS INTEGER) = 5 THEN 1.00
                WHEN TRY_CAST(i.rating AS INTEGER) = 4 THEN 0.65
                WHEN TRY_CAST(i.rating AS INTEGER) = 3 THEN 0.10
                WHEN TRY_CAST(i.rating AS INTEGER) = 2 THEN -0.50
                WHEN TRY_CAST(i.rating AS INTEGER) = 1 THEN -1.00
                ELSE 0.00
            END::FLOAT AS preference_signal,
            MOD(TRY_CAST(i.user_id AS INTEGER), {partitions}) AS user_bucket
        FROM read_parquet('{interactions_glob}') i
        INNER JOIN eligible_users u
            ON TRY_CAST(i.user_id AS INTEGER) = u.user_id
        INNER JOIN eligible_books b
            ON TRY_CAST(i.book_id AS INTEGER) = b.item_id
        INNER JOIN read_parquet('{sql_path(outputs["catalog_map"])}') m
            ON TRY_CAST(i.book_id AS INTEGER) = m.item_id
        WHERE TRY_CAST(i.is_read AS INTEGER) = 1
           OR TRY_CAST(i.rating AS INTEGER) > 0
    )
    TO '{sql_path(outputs["filtered"])}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD,
        PARTITION_BY (user_bucket),
        OVERWRITE_OR_IGNORE TRUE
    );
    """
    return execute(
        connection,
        "WRITING FILTERED PARTITIONED INTERACTIONS",
        query,
    )


def validate_outputs(
    connection: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
    min_user: int,
    min_book: int,
) -> dict[str, Any]:
    filtered_glob = sql_path(outputs["filtered"] / "**" / "*.parquet")

    checks = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT user_id),
            COUNT(DISTINCT item_id),
            MIN(rating),
            MAX(rating),
            COUNT_IF(interaction_weight < 0),
            COUNT_IF(preference_signal < 0)
        FROM read_parquet('{filtered_glob}', hive_partitioning = true)
        """
    ).fetchone()

    min_user_observed = connection.execute(
        f"SELECT MIN(interaction_count) "
        f"FROM read_parquet('{sql_path(outputs['user_stats'])}')"
    ).fetchone()[0]

    min_book_observed = connection.execute(
        f"SELECT MIN(interaction_count) "
        f"FROM read_parquet('{sql_path(outputs['book_stats'])}')"
    ).fetchone()[0]

    if int(min_user_observed) < min_user:
        raise RuntimeError("User threshold validation failed.")
    if int(min_book_observed) < min_book:
        raise RuntimeError("Book threshold validation failed.")
    if int(checks[5]) != 0:
        raise RuntimeError("Negative model weights were produced.")

    parquet_files = list(outputs["filtered"].rglob("*.parquet"))
    total_size = sum(path.stat().st_size for path in parquet_files)

    return {
        "filtered_interactions": {
            "rows": int(checks[0]),
            "users": int(checks[1]),
            "books": int(checks[2]),
            "minimum_rating": int(checks[3]),
            "maximum_rating": int(checks[4]),
            "negative_model_weights": int(checks[5]),
            "negative_preferences": int(checks[6]),
            "parquet_files": len(parquet_files),
            "size_bytes": total_size,
            "size_human": human_size(total_size),
        },
        "threshold_checks": {
            "minimum_user_interactions_observed": int(min_user_observed),
            "minimum_book_interactions_observed": int(min_book_observed),
        },
        "user_stats_rows": pq.ParquetFile(
            outputs["user_stats"]
        ).metadata.num_rows,
        "book_stats_rows": pq.ParquetFile(
            outputs["book_stats"]
        ).metadata.num_rows,
    }


def main() -> None:
    args = parse_args()

    if args.min_book_interactions <= 0:
        raise ValueError("--min-book-interactions must be positive.")
    if args.min_user_interactions <= 0:
        raise ValueError("--min-user-interactions must be positive.")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if args.partitions <= 0:
        raise ValueError("--partitions must be positive.")

    config = load_config(args.config)
    interim_dir = resolve_path(args.input_dir or config["paths"]["interim_dir"])
    processed_dir = resolve_path(
        args.output_dir or config["paths"]["processed_dir"]
    )

    inputs = require_inputs(interim_dir, processed_dir)
    outputs = prepare_outputs(processed_dir, args.overwrite)

    temp_dir = processed_dir / ".interaction_filter_tmp"
    work_db = processed_dir / ".interaction_filter.duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    work_db.unlink(missing_ok=True)

    interaction_glob = sql_path(inputs["interactions"] / "*.parquet")

    print("=" * 76)
    print("GOODREADS LOW-MEMORY INTERACTION FILTER")
    print(f"Interactions       : {inputs['interactions']}")
    print(f"Catalog            : {inputs['catalog']}")
    print(f"Output             : {processed_dir}")
    print(f"Min book count     : {args.min_book_interactions:,}")
    print(f"Min user count     : {args.min_user_interactions:,}")
    print(f"Max iterations     : {args.max_iterations}")
    print(f"Output partitions  : {args.partitions}")
    print(f"Threads            : {args.threads}")
    print(f"Memory limit       : {args.memory_limit}")
    print("=" * 76)

    total_started = time.perf_counter()
    connection = duckdb.connect(str(work_db))
    connection.execute(f"SET threads = {max(args.threads, 1)}")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order = false")

    timings: dict[str, float] = {}
    iteration_history: list[dict[str, int]] = []

    try:
        timings["catalog_map_seconds"] = build_catalog_map(
            connection,
            inputs["catalog"],
            outputs["catalog_map"],
        )

        timings["initial_books_seconds"] = build_initial_book_set(
            connection,
            interaction_glob,
            outputs["catalog_map"],
            args.min_book_interactions,
        )

        timings["initial_users_seconds"] = build_user_set(
            connection,
            interaction_glob,
            args.min_user_interactions,
        )

        previous_counts: tuple[int, int] | None = None

        for iteration in range(1, args.max_iterations + 1):
            timings[f"iteration_{iteration}_books_seconds"] = refine_book_set(
                connection,
                interaction_glob,
                args.min_book_interactions,
            )
            timings[f"iteration_{iteration}_users_seconds"] = refine_user_set(
                connection,
                interaction_glob,
                args.min_user_interactions,
            )

            users, books = table_counts(connection)
            iteration_history.append(
                {
                    "iteration": iteration,
                    "eligible_users": users,
                    "eligible_books": books,
                }
            )
            print(f"Iteration {iteration}: {users:,} users, {books:,} books")

            current_counts = (users, books)
            if previous_counts == current_counts:
                print("Eligibility sets converged.")
                break
            previous_counts = current_counts

        timings.update(write_stats(connection, outputs))

        timings["filtered_write_seconds"] = write_filtered_interactions(
            connection,
            interaction_glob,
            outputs,
            args.partitions,
        )

        print("\n" + "=" * 76)
        print("VALIDATING OUTPUTS")
        validation = validate_outputs(
            connection,
            outputs,
            args.min_user_interactions,
            args.min_book_interactions,
        )

    finally:
        connection.close()
        work_db.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_seconds = time.perf_counter() - total_started
    timings["total_seconds"] = round(total_seconds, 3)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "low_memory_iterative_k_core",
        "input_interactions": str(inputs["interactions"]),
        "input_catalog": str(inputs["catalog"]),
        "output_directory": str(processed_dir),
        "settings": {
            "min_book_interactions": args.min_book_interactions,
            "min_user_interactions": args.min_user_interactions,
            "max_iterations": args.max_iterations,
            "partitions": args.partitions,
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "overwrite": args.overwrite,
        },
        "iteration_history": iteration_history,
        "timings": timings,
        "validation": validation,
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = validation["filtered_interactions"]

    print("\n" + "=" * 76)
    print("INTERACTION FILTER COMPLETE")
    print(f"Rows          : {summary['rows']:,}")
    print(f"Users         : {summary['users']:,}")
    print(f"Books         : {summary['books']:,}")
    print(f"Parquet files : {summary['parquet_files']:,}")
    print(f"Output size   : {summary['size_human']}")
    print(f"Total elapsed : {format_elapsed(total_seconds)}")
    print(f"Metadata      : {outputs['metadata']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
