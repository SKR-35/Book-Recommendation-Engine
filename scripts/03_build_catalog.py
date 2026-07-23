from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Goodreads catalog in low-memory Parquet batches."
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--min-votes-quantile", type=float, default=0.90)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def elapsed(start: float) -> str:
    seconds = int(time.perf_counter() - start)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    return f"{value:,.2f} TB"


def require_files(input_dir: Path) -> dict[str, Path]:
    files = {
        "books": input_dir / "books.parquet",
        "authors": input_dir / "authors.parquet",
        "genres": input_dir / "genres.parquet",
        "works": input_dir / "works.parquet",
        "book_id_map": input_dir / "book_id_map.parquet",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n  - " + "\n  - ".join(missing))
    return files


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "book_authors": output_dir / "book_authors.parquet",
        "book_genres": output_dir / "book_genres.parquet",
        "author_summary": output_dir / "author_summary.parquet",
        "genre_summary": output_dir / "genre_summary.parquet",
        "canonical_editions": output_dir / "canonical_editions.parquet",
        "catalog": output_dir / "catalog",
        "metadata": output_dir / "catalog_metadata.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist. Run again with --overwrite.")
    if overwrite:
        for path in existing:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    outputs["catalog"].mkdir(parents=True, exist_ok=True)
    return outputs


def copy_query(
    con: duckdb.DuckDBPyConnection,
    title: str,
    query: str,
    output: Path,
) -> float:
    print("\n" + "=" * 72)
    print(title)
    start = time.perf_counter()
    con.execute(
        f"""
        COPY ({query})
        TO '{sql_path(output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )
    duration = time.perf_counter() - start
    print(f"Completed in {elapsed(start)}")
    return duration


def build_bridges(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    outputs: dict[str, Path],
) -> dict[str, float]:
    timings: dict[str, float] = {}

    author_query = f"""
        WITH expanded AS (
            SELECT
                TRY_CAST(b.book_id AS BIGINT) AS book_id,
                TRY_CAST(json_extract_string(j.value, '$.author_id') AS BIGINT) AS author_id,
                NULLIF(json_extract_string(j.value, '$.role'), '') AS author_role,
                TRY_CAST(j.key AS INTEGER) AS author_position
            FROM read_parquet('{sql_path(files["books"])}') b,
            LATERAL json_each(
                CASE
                    WHEN json_valid(CAST(b.authors AS VARCHAR))
                    THEN CAST(b.authors AS JSON)
                    ELSE CAST('[]' AS JSON)
                END
            ) j
        )
        SELECT
            e.book_id,
            e.author_id,
            COALESCE(NULLIF(a.name, ''), 'Unknown author') AS author_name,
            e.author_role,
            e.author_position
        FROM expanded e
        LEFT JOIN read_parquet('{sql_path(files["authors"])}') a
            ON e.author_id = TRY_CAST(a.author_id AS BIGINT)
        WHERE e.book_id IS NOT NULL AND e.author_id IS NOT NULL
    """
    timings["book_authors"] = copy_query(
        con, "BUILDING BOOK–AUTHOR BRIDGE", author_query, outputs["book_authors"]
    )

    genre_query = f"""
        WITH expanded AS (
            SELECT
                TRY_CAST(g.book_id AS BIGINT) AS book_id,
                TRIM(j.key) AS genre_name,
                TRY_CAST(json_extract_string(j.value, '$') AS BIGINT) AS genre_count
            FROM read_parquet('{sql_path(files["genres"])}') g,
            LATERAL json_each(
                CASE
                    WHEN json_valid(CAST(g.genres AS VARCHAR))
                    THEN CAST(g.genres AS JSON)
                    ELSE CAST('{{}}' AS JSON)
                END
            ) j
        )
        SELECT
            book_id,
            genre_name,
            COALESCE(genre_count, 0) AS genre_count,
            ROW_NUMBER() OVER (
                PARTITION BY book_id
                ORDER BY COALESCE(genre_count, 0) DESC, genre_name
            ) AS genre_rank
        FROM expanded
        WHERE book_id IS NOT NULL AND genre_name <> ''
    """
    timings["book_genres"] = copy_query(
        con, "BUILDING BOOK–GENRE BRIDGE", genre_query, outputs["book_genres"]
    )
    return timings


def build_summaries(
    con: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
) -> dict[str, float]:
    timings: dict[str, float] = {}

    author_summary = f"""
        SELECT
            book_id,
            STRING_AGG(author_name, ', ' ORDER BY author_position, author_name) AS author_names,
            ARG_MIN(author_name, author_position) AS primary_author,
            ARG_MIN(author_id, author_position) AS primary_author_id,
            COUNT(*) AS author_count
        FROM read_parquet('{sql_path(outputs["book_authors"])}')
        GROUP BY book_id
    """
    timings["author_summary"] = copy_query(
        con, "BUILDING AUTHOR SUMMARY", author_summary, outputs["author_summary"]
    )

    genre_summary = f"""
        SELECT
            book_id,
            STRING_AGG(genre_name, ' | ' ORDER BY genre_rank, genre_name) AS genre_names,
            ARG_MIN(genre_name, genre_rank) AS primary_genre,
            COUNT(*) AS genre_count
        FROM read_parquet('{sql_path(outputs["book_genres"])}')
        GROUP BY book_id
    """
    timings["genre_summary"] = copy_query(
        con, "BUILDING GENRE SUMMARY", genre_summary, outputs["genre_summary"]
    )
    return timings


def build_canonical_editions(
    con: duckdb.DuckDBPyConnection,
    books_path: Path,
    output_path: Path,
) -> float:
    # This expensive window runs on only four narrow numeric columns, not after all joins.
    query = f"""
        SELECT
            book_id,
            work_key,
            edition_rank
        FROM (
            SELECT
                TRY_CAST(book_id AS BIGINT) AS book_id,
                COALESCE(
                    TRY_CAST(work_id AS BIGINT),
                    -TRY_CAST(book_id AS BIGINT)
                ) AS work_key,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(
                        TRY_CAST(work_id AS BIGINT),
                        -TRY_CAST(book_id AS BIGINT)
                    )
                    ORDER BY
                        COALESCE(TRY_CAST(ratings_count AS BIGINT), 0) DESC,
                        COALESCE(TRY_CAST(text_reviews_count AS BIGINT), 0) DESC,
                        TRY_CAST(book_id AS BIGINT)
                ) AS edition_rank
            FROM read_parquet('{sql_path(books_path)}')
            WHERE TRY_CAST(book_id AS BIGINT) IS NOT NULL
        )
    """
    return copy_query(
        con, "BUILDING NARROW EDITION-RANK TABLE", query, output_path
    )


def get_scoring_parameters(
    con: duckdb.DuckDBPyConnection,
    books_path: Path,
    quantile: float,
) -> dict[str, float]:
    mean_rating, minimum_votes = con.execute(
        f"""
        SELECT
            AVG(TRY_CAST(average_rating AS DOUBLE)),
            QUANTILE_CONT(TRY_CAST(ratings_count AS DOUBLE), {quantile})
        FROM read_parquet('{sql_path(books_path)}')
        WHERE TRY_CAST(average_rating AS DOUBLE) BETWEEN 0 AND 5
          AND TRY_CAST(ratings_count AS BIGINT) > 0
        """
    ).fetchone()
    return {
        "global_mean_rating": float(mean_rating or 0),
        "minimum_votes": max(float(minimum_votes or 1), 1.0),
        "minimum_votes_quantile": quantile,
    }


def build_catalog_batches(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    outputs: dict[str, Path],
    batch_size: int,
    scoring: dict[str, float],
) -> dict[str, int | float]:
    minimum_id, maximum_id = con.execute(
        f"""
        SELECT MIN(TRY_CAST(book_id AS BIGINT)), MAX(TRY_CAST(book_id AS BIGINT))
        FROM read_parquet('{sql_path(files["books"])}')
        WHERE TRY_CAST(book_id AS BIGINT) IS NOT NULL
        """
    ).fetchone()

    if minimum_id is None or maximum_id is None:
        raise RuntimeError("No valid book IDs found.")

    mean_rating = scoring["global_mean_rating"]
    minimum_votes = scoring["minimum_votes"]
    total_rows = 0
    part_number = 0
    started = time.perf_counter()

    print("\n" + "=" * 72)
    print("BUILDING CATALOG IN BATCHES")
    print(f"Book ID range: {minimum_id:,}–{maximum_id:,}")
    print(f"Batch width  : {batch_size:,}")

    lower = int(minimum_id)
    maximum = int(maximum_id)

    while lower <= maximum:
        upper = min(lower + batch_size - 1, maximum)
        output = outputs["catalog"] / f"part-{part_number:05d}.parquet"

        query = f"""
            WITH base_books AS (
                SELECT
                    TRY_CAST(book_id AS BIGINT) AS book_id,
                    TRY_CAST(work_id AS BIGINT) AS work_id,
                    NULLIF(TRIM(title), '') AS title,
                    NULLIF(TRIM(title_without_series), '') AS title_without_series,
                    NULLIF(TRIM(description), '') AS description,
                    TRY_CAST(average_rating AS DOUBLE) AS average_rating,
                    TRY_CAST(ratings_count AS BIGINT) AS ratings_count,
                    TRY_CAST(text_reviews_count AS BIGINT) AS text_reviews_count,
                    TRY_CAST(publication_year AS INTEGER) AS publication_year,
                    NULLIF(TRIM(publisher), '') AS publisher,
                    NULLIF(TRIM(language_code), '') AS language_code,
                    NULLIF(TRIM(isbn), '') AS isbn,
                    NULLIF(TRIM(isbn13), '') AS isbn13,
                    NULLIF(TRIM(url), '') AS url,
                    NULLIF(TRIM(image_url), '') AS image_url
                FROM read_parquet('{sql_path(files["books"])}')
                WHERE TRY_CAST(book_id AS BIGINT) BETWEEN {lower} AND {upper}
            )
            SELECT
                b.*,
                a.primary_author_id,
                a.primary_author,
                a.author_names,
                COALESCE(a.author_count, 0) AS author_count,
                g.primary_genre,
                g.genre_names,
                COALESCE(g.genre_count, 0) AS genre_count,
                TRY_CAST(m.book_id_csv AS BIGINT) AS interaction_book_id,
                TRY_CAST(w.books_count AS BIGINT) AS work_books_count,
                TRY_CAST(w.ratings_count AS BIGINT) AS work_ratings_count,
                NULLIF(TRIM(w.original_title), '') AS original_title,
                TRY_CAST(w.original_publication_year AS INTEGER) AS original_publication_year,
                TRY_CAST(w.original_publication_month AS INTEGER) AS original_publication_month,
                TRY_CAST(w.original_publication_day AS INTEGER) AS original_publication_day,
                e.edition_rank,
                e.edition_rank = 1 AS is_canonical_edition,
                CASE
                    WHEN b.average_rating IS NULL OR b.ratings_count IS NULL THEN NULL
                    ELSE
                        (b.ratings_count / (b.ratings_count + {minimum_votes}))
                        * b.average_rating
                        + ({minimum_votes} / (b.ratings_count + {minimum_votes}))
                        * {mean_rating}
                END AS bayesian_rating,
                LOWER(
                    REGEXP_REPLACE(
                        TRIM(COALESCE(b.title_without_series, b.title, '')),
                        '[^[:alnum:]]+', ' ', 'g'
                    )
                ) AS title_key,
                CASE
                    WHEN b.ratings_count IS NULL OR b.ratings_count <= 0 THEN 1.0
                    ELSE 1.0 / LN(b.ratings_count + EXP(1))
                END AS novelty_score,
                COALESCE(LENGTH(b.description) >= 40, FALSE) AS has_usable_description
            FROM base_books b
            LEFT JOIN read_parquet('{sql_path(outputs["author_summary"])}') a USING (book_id)
            LEFT JOIN read_parquet('{sql_path(outputs["genre_summary"])}') g USING (book_id)
            LEFT JOIN read_parquet('{sql_path(files["book_id_map"])}') m
                ON b.book_id = TRY_CAST(m.book_id AS BIGINT)
            LEFT JOIN read_parquet('{sql_path(files["works"])}') w
                ON b.work_id = TRY_CAST(w.work_id AS BIGINT)
            LEFT JOIN read_parquet('{sql_path(outputs["canonical_editions"])}') e USING (book_id)
        """

        batch_start = time.perf_counter()
        con.execute(
            f"""
            COPY ({query})
            TO '{sql_path(output)}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
            """
        )

        rows = pq.ParquetFile(output).metadata.num_rows
        if rows == 0:
            output.unlink(missing_ok=True)
        else:
            total_rows += rows
            print(
                f"Part {part_number:05d} | IDs {lower:,}–{upper:,} | "
                f"{rows:,} rows | {elapsed(batch_start)}"
            )
            part_number += 1

        lower = upper + 1

    return {
        "rows": total_rows,
        "parts": part_number,
        "seconds": round(time.perf_counter() - started, 3),
        "minimum_book_id": int(minimum_id),
        "maximum_book_id": int(maximum_id),
    }


def validate(
    con: duckdb.DuckDBPyConnection,
    outputs: dict[str, Path],
) -> dict:
    parts = sorted(outputs["catalog"].glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("No catalog parts were created.")

    glob_path = sql_path(outputs["catalog"] / "*.parquet")
    checks = con.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT book_id),
            COUNT(*) - COUNT(DISTINCT book_id),
            COUNT_IF(title IS NULL OR TRIM(title) = ''),
            COUNT_IF(primary_author IS NULL),
            COUNT_IF(primary_genre IS NULL),
            COUNT_IF(is_canonical_edition),
            COUNT_IF(interaction_book_id IS NOT NULL)
        FROM read_parquet('{glob_path}')
        """
    ).fetchone()

    if checks[2] != 0:
        raise RuntimeError(f"Duplicate book IDs found: {checks[2]:,}")

    return {
        "catalog": {
            "path": str(outputs["catalog"]),
            "parts": len(parts),
            "size_bytes": sum(path.stat().st_size for path in parts),
            "size_human": human_size(sum(path.stat().st_size for path in parts)),
        },
        "checks": {
            "rows": checks[0],
            "distinct_books": checks[1],
            "duplicate_book_ids": checks[2],
            "missing_titles": checks[3],
            "missing_primary_authors": checks[4],
            "missing_primary_genres": checks[5],
            "canonical_editions": checks[6],
            "interaction_mapped_books": checks[7],
        },
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0 < args.min_votes_quantile < 1:
        raise ValueError("--min-votes-quantile must be between 0 and 1.")

    config = load_config(args.config)
    input_dir = resolve_path(args.input_dir or config["paths"]["interim_dir"])
    output_dir = resolve_path(args.output_dir or config["paths"]["processed_dir"])

    files = require_files(input_dir)
    outputs = prepare_outputs(output_dir, args.overwrite)

    temp_dir = output_dir / ".duckdb_tmp"
    work_db = output_dir / ".catalog_work.duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    work_db.unlink(missing_ok=True)

    print("=" * 72)
    print("GOODREADS LOW-MEMORY CATALOG BUILDER")
    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Threads     : {args.threads}")
    print(f"Memory limit: {args.memory_limit}")
    print(f"Batch size  : {args.batch_size:,}")
    print("=" * 72)

    started = time.perf_counter()
    con = duckdb.connect(str(work_db))
    con.execute(f"SET threads = {max(1, args.threads)}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order = false")

    timings = {}
    try:
        timings.update(build_bridges(con, files, outputs))
        timings.update(build_summaries(con, outputs))
        timings["canonical_editions"] = build_canonical_editions(
            con, files["books"], outputs["canonical_editions"]
        )
        scoring = get_scoring_parameters(
            con, files["books"], args.min_votes_quantile
        )
        batch_result = build_catalog_batches(
            con, files, outputs, args.batch_size, scoring
        )
        validation = validate(con, outputs)
    finally:
        con.close()
        work_db.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "low_memory_batched_catalog",
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "settings": vars(args),
        "scoring_parameters": scoring,
        "batch_result": batch_result,
        "timings": timings,
        "validation": validation,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    checks = validation["checks"]
    print("\n" + "=" * 72)
    print("CATALOG BUILD COMPLETE")
    print(f"Rows                     : {checks['rows']:,}")
    print(f"Distinct books           : {checks['distinct_books']:,}")
    print(f"Catalog parts            : {validation['catalog']['parts']:,}")
    print(f"Canonical editions       : {checks['canonical_editions']:,}")
    print(f"Interaction-mapped books : {checks['interaction_mapped_books']:,}")
    print(f"Catalog size             : {validation['catalog']['size_human']}")
    print(f"Total elapsed            : {elapsed(started)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
