from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build low-memory content features for the Goodreads catalog. "
            "The TF-IDF matrix is written as sparse shards rather than one "
            "large in-memory matrix."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--fit-sample-size",
        type=str,
        default="250000",
        help=(
            "Number of documents used to fit the TF-IDF vocabulary. "
            "Use 'all' or '0' to fit on the full content catalog."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--minimum-description-length", type=int, default=40)
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
            f"No catalog Parquet parts found in {inputs['catalog']}"
        )

    return inputs


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    content_dir = output_dir / "content"

    if content_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{content_dir} already exists. Run again with --overwrite."
            )
        shutil.rmtree(content_dir)

    outputs = {
        "directory": content_dir,
        "documents": content_dir / "content_documents.parquet",
        "fit_sample": content_dir / "tfidf_fit_sample.parquet",
        "vectorizer": content_dir / "tfidf_vectorizer.joblib",
        "matrix_dir": content_dir / "tfidf_matrix",
        "row_map": content_dir / "content_row_map.parquet",
        "metadata": content_dir / "content_metadata.json",
    }

    content_dir.mkdir(parents=True, exist_ok=True)
    outputs["matrix_dir"].mkdir(parents=True, exist_ok=True)
    return outputs


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


def build_content_documents(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    output_path: Path,
    minimum_description_length: int,
) -> float:
    """
    Build a compact document table for canonical books in the filtered graph.

    Important fields are repeated to give them more influence without requiring
    separate vectorizers:
      title × 3, author × 2, genres × 2, description × 1.
    """
    catalog_glob = sql_path(inputs["catalog"] / "*.parquet")

    query = f"""
    COPY (
        SELECT
            c.book_id,
            c.interaction_book_id AS item_id,
            c.work_id,
            c.title,
            c.primary_author,
            c.primary_genre,
            c.genre_names,
            c.language_code,
            c.publication_year,
            c.image_url,
            c.bayesian_rating,
            s.interaction_count,
            CONCAT_WS(
                ' ',
                COALESCE(c.title, ''),
                COALESCE(c.title, ''),
                COALESCE(c.title, ''),
                COALESCE(c.primary_author, ''),
                COALESCE(c.primary_author, ''),
                COALESCE(c.genre_names, ''),
                COALESCE(c.genre_names, ''),
                CASE
                    WHEN c.description IS NOT NULL
                     AND LENGTH(c.description) >= {minimum_description_length}
                    THEN c.description
                    ELSE ''
                END,
                CASE
                    WHEN c.language_code IS NOT NULL
                    THEN CONCAT('language_', LOWER(c.language_code))
                    ELSE ''
                END,
                CASE
                    WHEN c.publication_year BETWEEN 1000 AND 2100
                    THEN CONCAT(
                        'decade_',
                        CAST(
                            FLOOR(c.publication_year / 10) * 10
                            AS INTEGER
                        )
                    )
                    ELSE ''
                END
            ) AS content_text
        FROM read_parquet('{catalog_glob}') AS c
        INNER JOIN read_parquet('{sql_path(inputs["book_stats"])}') AS s
            ON c.interaction_book_id = s.item_id
        WHERE c.is_canonical_edition = TRUE
          AND c.interaction_book_id IS NOT NULL
          AND c.title IS NOT NULL
          AND TRIM(c.title) <> ''
        ORDER BY c.book_id
    )
    TO '{sql_path(output_path)}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING CONTENT DOCUMENT TABLE", query)


def build_fit_sample(
    connection: duckdb.DuckDBPyConnection,
    documents_path: Path,
    output_path: Path,
    sample_size: int | None,
) -> tuple[float, int, bool]:
    total_rows = pq.ParquetFile(documents_path).metadata.num_rows
    use_all = sample_size is None or sample_size <= 0 or sample_size >= total_rows
    actual_size = total_rows if use_all else min(sample_size, total_rows)

    if use_all:
        title = "BUILDING FULL TF-IDF FIT CORPUS"
        query = f"""
        COPY (
            SELECT book_id, content_text
            FROM read_parquet('{sql_path(documents_path)}')
            WHERE content_text IS NOT NULL
              AND LENGTH(TRIM(content_text)) > 0
            ORDER BY book_id
        )
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    else:
        title = "BUILDING DETERMINISTIC TF-IDF FIT SAMPLE"
        query = f"""
        COPY (
            SELECT book_id, content_text
            FROM read_parquet('{sql_path(documents_path)}')
            WHERE content_text IS NOT NULL
              AND LENGTH(TRIM(content_text)) > 0
            ORDER BY HASH(book_id)
            LIMIT {actual_size}
        )
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """

    elapsed = execute(connection, title, query)
    return elapsed, actual_size, use_all


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[\w][\w-]+\b",
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def fit_vectorizer(
    sample_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[TfidfVectorizer, float]:
    print("\n" + "=" * 76)
    print("FITTING TF-IDF VECTORIZER")
    started = time.perf_counter()

    sample = pd.read_parquet(sample_path, columns=["content_text"])
    texts = sample["content_text"].fillna("").astype(str)

    vectorizer = make_vectorizer(args)
    vectorizer.fit(texts)
    joblib.dump(vectorizer, output_path, compress=3)

    elapsed = time.perf_counter() - started
    print(f"Sample rows : {len(sample):,}")
    print(f"Vocabulary  : {len(vectorizer.vocabulary_):,}")
    print(f"Completed in {format_elapsed(elapsed)}")
    return vectorizer, elapsed


def iter_document_batches(
    documents_path: Path,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(documents_path)
    columns = [
        "book_id",
        "item_id",
        "work_id",
        "title",
        "primary_author",
        "primary_genre",
        "language_code",
        "publication_year",
        "image_url",
        "bayesian_rating",
        "interaction_count",
        "content_text",
    ]

    for record_batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        yield record_batch.to_pandas()


def transform_documents(
    vectorizer: TfidfVectorizer,
    documents_path: Path,
    matrix_dir: Path,
    row_map_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    print("\n" + "=" * 76)
    print("TRANSFORMING DOCUMENTS TO SPARSE TF-IDF SHARDS")

    started = time.perf_counter()
    row_maps: list[pd.DataFrame] = []
    total_rows = 0
    total_nonzero = 0
    shard_count = 0

    for shard_number, frame in enumerate(
        iter_document_batches(documents_path, batch_size)
    ):
        texts = frame["content_text"].fillna("").astype(str)
        matrix = vectorizer.transform(texts).tocsr().astype(np.float32)

        shard_path = matrix_dir / f"part-{shard_number:05d}.npz"
        sparse.save_npz(shard_path, matrix, compressed=True)

        row_map = frame.drop(columns=["content_text"]).copy()
        row_map.insert(
            0,
            "global_row",
            np.arange(total_rows, total_rows + len(frame), dtype=np.int64),
        )
        row_map.insert(1, "shard_id", shard_number)
        row_map.insert(
            2,
            "shard_row",
            np.arange(len(frame), dtype=np.int32),
        )
        row_maps.append(row_map)

        total_rows += matrix.shape[0]
        total_nonzero += matrix.nnz
        shard_count += 1

        print(
            f"Shard {shard_number:05d} | "
            f"{matrix.shape[0]:,} rows | "
            f"{matrix.nnz:,} non-zero values"
        )

        del matrix
        del frame
        del row_map

    if total_rows == 0:
        raise RuntimeError("No content rows were transformed.")

    full_row_map = pd.concat(row_maps, ignore_index=True)
    full_row_map.to_parquet(
        row_map_path,
        index=False,
        compression="zstd",
    )

    elapsed = time.perf_counter() - started
    matrix_size = sum(path.stat().st_size for path in matrix_dir.glob("*.npz"))

    return {
        "rows": total_rows,
        "columns": len(vectorizer.vocabulary_),
        "nonzero_values": total_nonzero,
        "density": total_nonzero
        / (total_rows * max(len(vectorizer.vocabulary_), 1)),
        "shards": shard_count,
        "matrix_size_bytes": matrix_size,
        "matrix_size_human": human_size(matrix_size),
        "seconds": round(elapsed, 3),
    }


def validate_outputs(
    outputs: dict[str, Path],
    transform_result: dict[str, Any],
) -> dict[str, Any]:
    documents_rows = pq.ParquetFile(
        outputs["documents"]
    ).metadata.num_rows
    row_map_rows = pq.ParquetFile(
        outputs["row_map"]
    ).metadata.num_rows
    sample_rows = pq.ParquetFile(
        outputs["fit_sample"]
    ).metadata.num_rows
    matrix_parts = sorted(outputs["matrix_dir"].glob("part-*.npz"))

    if documents_rows != row_map_rows:
        raise RuntimeError(
            f"Document/row-map mismatch: {documents_rows:,} vs "
            f"{row_map_rows:,}."
        )
    if documents_rows != transform_result["rows"]:
        raise RuntimeError("Document/matrix row-count mismatch.")
    if len(matrix_parts) != transform_result["shards"]:
        raise RuntimeError("Matrix shard-count mismatch.")
    if not outputs["vectorizer"].exists():
        raise RuntimeError("Vectorizer file was not created.")

    return {
        "documents": {
            "rows": documents_rows,
            "size_bytes": outputs["documents"].stat().st_size,
            "size_human": human_size(outputs["documents"].stat().st_size),
        },
        "fit_sample": {
            "rows": sample_rows,
            "size_bytes": outputs["fit_sample"].stat().st_size,
            "size_human": human_size(outputs["fit_sample"].stat().st_size),
        },
        "row_map": {
            "rows": row_map_rows,
            "size_bytes": outputs["row_map"].stat().st_size,
            "size_human": human_size(outputs["row_map"].stat().st_size),
        },
        "vectorizer": {
            "size_bytes": outputs["vectorizer"].stat().st_size,
            "size_human": human_size(outputs["vectorizer"].stat().st_size),
        },
        "matrix": transform_result,
    }


def main() -> None:
    args = parse_args()

    fit_value = str(args.fit_sample_size).strip().lower()
    if fit_value in {"all", "0"}:
        fit_sample_size: int | None = None
    else:
        try:
            fit_sample_size = int(fit_value)
        except ValueError as exc:
            raise ValueError(
                "--fit-sample-size must be a positive integer, 0, or 'all'."
            ) from exc
        if fit_sample_size <= 0:
            raise ValueError(
                "--fit-sample-size must be a positive integer, 0, or 'all'."
            )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.min_df <= 0:
        raise ValueError("--min-df must be positive.")
    if not 0 < args.max_df <= 1:
        raise ValueError("--max-df must be in (0, 1].")
    if args.ngram_max not in (1, 2):
        raise ValueError("--ngram-max must be 1 or 2.")

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.input_dir or config["paths"]["processed_dir"]
    )
    output_dir = resolve_path(
        args.output_dir or config["paths"]["processed_dir"]
    )

    inputs = require_inputs(processed_dir)
    outputs = prepare_outputs(output_dir, args.overwrite)

    work_db = outputs["directory"] / ".content_work.duckdb"
    temp_dir = outputs["directory"] / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    work_db.unlink(missing_ok=True)

    print("=" * 76)
    print("GOODREADS CONTENT FEATURE BUILDER")
    print(f"Catalog          : {inputs['catalog']}")
    print(f"Book stats       : {inputs['book_stats']}")
    print(f"Output           : {outputs['directory']}")
    print(f"Fit sample       : {fit_sample_size:,}" if fit_sample_size is not None else "Fit sample       : ALL DOCUMENTS")
    print(f"Transform batch  : {args.batch_size:,}")
    print(f"Max features     : {args.max_features:,}")
    print(f"N-grams          : 1–{args.ngram_max}")
    print(f"Threads          : {args.threads}")
    print(f"Memory limit     : {args.memory_limit}")
    print("=" * 76)

    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    connection = duckdb.connect(str(work_db))
    connection.execute(f"SET threads = {max(args.threads, 1)}")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order = false")

    try:
        timings["documents_seconds"] = build_content_documents(
            connection,
            inputs,
            outputs["documents"],
            args.minimum_description_length,
        )
        (
            timings["sample_seconds"],
            actual_fit_rows,
            used_full_corpus,
        ) = build_fit_sample(
            connection,
            outputs["documents"],
            outputs["fit_sample"],
            fit_sample_size,
        )
    finally:
        connection.close()
        work_db.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    vectorizer, fit_seconds = fit_vectorizer(
        outputs["fit_sample"],
        outputs["vectorizer"],
        args,
    )
    timings["vectorizer_fit_seconds"] = round(fit_seconds, 3)

    transform_result = transform_documents(
        vectorizer,
        outputs["documents"],
        outputs["matrix_dir"],
        outputs["row_map"],
        args.batch_size,
    )
    timings["transform_seconds"] = transform_result["seconds"]

    validation = validate_outputs(outputs, transform_result)

    total_seconds = time.perf_counter() - total_started
    timings["total_seconds"] = round(total_seconds, 3)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "tfidf_content_features",
        "input_catalog": str(inputs["catalog"]),
        "input_book_stats": str(inputs["book_stats"]),
        "output_directory": str(outputs["directory"]),
        "settings": {
            "fit_sample_size_requested": args.fit_sample_size,
            "fit_sample_size_actual": actual_fit_rows,
            "fit_used_full_corpus": used_full_corpus,
            "batch_size": args.batch_size,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_range": [1, args.ngram_max],
            "minimum_description_length": args.minimum_description_length,
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "overwrite": args.overwrite,
        },
        "feature_weighting": {
            "title_repetitions": 3,
            "author_repetitions": 2,
            "genre_repetitions": 2,
            "description_repetitions": 1,
            "language_token": True,
            "decade_token": True,
        },
        "timings": timings,
        "validation": validation,
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    matrix = validation["matrix"]

    print("\n" + "=" * 76)
    print("CONTENT FEATURE BUILD COMPLETE")
    print(f"Documents       : {matrix['rows']:,}")
    print(f"Vocabulary      : {matrix['columns']:,}")
    print(f"Matrix shards   : {matrix['shards']:,}")
    print(f"Non-zero values : {matrix['nonzero_values']:,}")
    print(f"Matrix density  : {matrix['density']:.8%}")
    print(f"Matrix size     : {matrix['matrix_size_human']}")
    print(f"Total elapsed   : {format_elapsed(total_seconds)}")
    print(f"Metadata        : {outputs['metadata']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
