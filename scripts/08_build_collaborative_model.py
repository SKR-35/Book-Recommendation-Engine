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
import hnswlib
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402

try:
    from implicit.als import AlternatingLeastSquares
except ImportError as exc:
    raise ImportError(
        "The 'implicit' package is required. Install it with: "
        "conda install -c conda-forge implicit -y"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a collaborative-filtering model from filtered Goodreads "
            "interactions using partition-wise low-memory preprocessing."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--max-interactions-per-user",
        type=int,
        default=100,
        help="Maximum interactions retained per user. Use 0 for all.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Retain only rows with positive preference_signal.",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--source-partitions",
        type=int,
        default=64,
        help="Number of user_bucket partitions created by script 04.",
    )

    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)

    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="2.5GB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-training-data", action="store_true")
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


def require_inputs(processed_dir: Path) -> dict[str, Path]:
    inputs = {
        "interactions": processed_dir / "interactions_filtered",
        "user_stats": processed_dir / "user_stats.parquet",
        "book_stats": processed_dir / "book_stats.parquet",
    }

    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required inputs are missing:\n  - " + "\n  - ".join(missing)
        )

    if not any(inputs["interactions"].rglob("*.parquet")):
        raise FileNotFoundError(
            f"No filtered interaction Parquet files found in "
            f"{inputs['interactions']}"
        )
    return inputs


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    model_dir = output_dir / "collaborative"

    if model_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_dir} already exists. Run again with --overwrite."
            )
        shutil.rmtree(model_dir)

    outputs = {
        "directory": model_dir,
        "user_map": model_dir / "user_map.parquet",
        "item_map": model_dir / "item_map.parquet",
        "training_dir": model_dir / "_training_dense",
        "user_factors": model_dir / "user_factors.npy",
        "item_factors": model_dir / "item_factors.npy",
        "item_hnsw": model_dir / "item_hnsw.bin",
        "metadata": model_dir / "collaborative_metadata.json",
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    outputs["training_dir"].mkdir(parents=True, exist_ok=True)
    return outputs


def build_dense_maps(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> dict[str, float]:
    timings: dict[str, float] = {}

    user_sql = f"""
    COPY (
        SELECT
            ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS user_index,
            TRY_CAST(user_id AS INTEGER) AS user_id,
            TRY_CAST(interaction_count AS BIGINT) AS interaction_count,
            TRY_CAST(rating_count AS BIGINT) AS rating_count,
            TRY_CAST(average_rating AS DOUBLE) AS average_rating
        FROM read_parquet('{sql_path(inputs["user_stats"])}')
        ORDER BY user_id
    )
    TO '{sql_path(outputs["user_map"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    timings["user_map_seconds"] = execute(
        connection, "BUILDING DENSE USER MAP", user_sql
    )

    item_sql = f"""
    COPY (
        SELECT
            ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_index,
            TRY_CAST(item_id AS INTEGER) AS item_id,
            TRY_CAST(local_book_id AS BIGINT) AS local_book_id,
            TRY_CAST(work_id AS BIGINT) AS work_id,
            title,
            primary_author,
            primary_genre,
            TRY_CAST(interaction_count AS BIGINT) AS interaction_count,
            TRY_CAST(rating_count AS BIGINT) AS rating_count,
            TRY_CAST(average_user_rating AS DOUBLE) AS average_user_rating,
            TRY_CAST(bayesian_rating AS DOUBLE) AS bayesian_rating
        FROM read_parquet('{sql_path(inputs["book_stats"])}')
        ORDER BY item_id
    )
    TO '{sql_path(outputs["item_map"])}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    timings["item_map_seconds"] = execute(
        connection, "BUILDING DENSE ITEM MAP", item_sql
    )
    return timings


def find_partition_file(
    interactions_dir: Path,
    bucket: int,
) -> Path | None:
    candidates = [
        interactions_dir / f"user_bucket={bucket}",
        interactions_dir / str(bucket),
    ]
    for directory in candidates:
        if directory.exists():
            files = sorted(directory.glob("*.parquet"))
            if files:
                return directory
    return None


def build_dense_training_data_partitioned(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    max_per_user: int,
    positive_only: bool,
    source_partitions: int,
) -> dict[str, Any]:
    """
    Process one user_bucket at a time.

    Script 04 partitioned interactions by user_id % 64, so every user's rows
    live in exactly one bucket. This makes per-user ranking safe and avoids the
    global sort/window that previously exhausted RAM.
    """
    print("\n" + "=" * 76)
    print("BUILDING DENSE TRAINING DATA PARTITION BY PARTITION")

    started = time.perf_counter()
    preference_filter = (
        "AND TRY_CAST(i.preference_signal AS DOUBLE) > 0"
        if positive_only
        else ""
    )

    part_count = 0
    total_rows = 0
    bucket_results: list[dict[str, Any]] = []

    for bucket in range(source_partitions):
        partition_path = find_partition_file(inputs["interactions"], bucket)
        if partition_path is None:
            print(f"Bucket {bucket:02d}: no source files, skipped")
            continue

        source_glob = sql_path(partition_path / "*.parquet")
        output_path = outputs["training_dir"] / f"part-{bucket:05d}.parquet"

        base = f"""
            SELECT
                TRY_CAST(u.user_index AS INTEGER) AS user_index,
                TRY_CAST(m.item_index AS INTEGER) AS item_index,
                MAX(TRY_CAST(i.interaction_weight AS FLOAT)) AS interaction_weight,
                MAX(TRY_CAST(i.preference_signal AS FLOAT)) AS preference_signal
            FROM read_parquet('{source_glob}') i
            INNER JOIN read_parquet('{sql_path(outputs["user_map"])}') u
                ON TRY_CAST(i.user_id AS INTEGER) = u.user_id
            INNER JOIN read_parquet('{sql_path(outputs["item_map"])}') m
                ON TRY_CAST(i.item_id AS INTEGER) = m.item_id
            WHERE TRY_CAST(i.interaction_weight AS DOUBLE) > 0
            {preference_filter}
            GROUP BY u.user_index, m.item_index
        """

        if max_per_user > 0:
            final_select = f"""
            SELECT
                user_index,
                item_index,
                interaction_weight,
                preference_signal
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_index
                        ORDER BY
                            interaction_weight DESC,
                            preference_signal DESC,
                            item_index
                    ) AS row_number
                FROM ({base})
            )
            WHERE row_number <= {max_per_user}
            """
        else:
            final_select = base

        sql = f"""
        COPY (
            {final_select}
        )
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """

        bucket_started = time.perf_counter()
        connection.execute(sql)
        bucket_elapsed = time.perf_counter() - bucket_started

        rows = pq.ParquetFile(output_path).metadata.num_rows
        if rows == 0:
            output_path.unlink(missing_ok=True)
        else:
            total_rows += rows
            part_count += 1

        bucket_results.append(
            {
                "bucket": bucket,
                "rows": rows,
                "seconds": round(bucket_elapsed, 3),
            }
        )
        print(
            f"Bucket {bucket:02d} | {rows:,} rows | "
            f"{format_elapsed(bucket_elapsed)}"
        )

        connection.execute("CHECKPOINT")

    if total_rows == 0:
        raise RuntimeError("No collaborative training rows were produced.")

    elapsed = time.perf_counter() - started
    print(f"Training rows : {total_rows:,}")
    print(f"Parts         : {part_count:,}")
    print(f"Completed in  : {format_elapsed(elapsed)}")

    return {
        "rows": total_rows,
        "parts": part_count,
        "seconds": round(elapsed, 3),
        "bucket_results": bucket_results,
    }


def training_row_count(training_dir: Path) -> int:
    dataset = ds.dataset(str(training_dir), format="parquet")
    return int(dataset.count_rows())


def build_sparse_matrix(
    training_dir: Path,
    n_users: int,
    n_items: int,
    scan_batch_size: int,
    alpha: float,
    scratch_dir: Path,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    print("\n" + "=" * 76)
    print("BUILDING USER–ITEM CSR MATRIX")

    started = time.perf_counter()
    n_rows = training_row_count(training_dir)
    if n_rows <= 0:
        raise RuntimeError("The collaborative training dataset is empty.")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    user_path = scratch_dir / "user_indices.dat"
    item_path = scratch_dir / "item_indices.dat"
    data_path = scratch_dir / "weights.dat"

    users = np.memmap(user_path, mode="w+", dtype=np.int32, shape=(n_rows,))
    items = np.memmap(item_path, mode="w+", dtype=np.int32, shape=(n_rows,))
    weights = np.memmap(data_path, mode="w+", dtype=np.float32, shape=(n_rows,))

    dataset = ds.dataset(str(training_dir), format="parquet")
    scanner = dataset.scanner(
        columns=["user_index", "item_index", "interaction_weight"],
        batch_size=scan_batch_size,
    )

    offset = 0
    for batch_number, batch in enumerate(scanner.to_batches()):
        user_values = batch.column("user_index").to_numpy(
            zero_copy_only=False
        ).astype(np.int32, copy=False)
        item_values = batch.column("item_index").to_numpy(
            zero_copy_only=False
        ).astype(np.int32, copy=False)
        weight_values = batch.column("interaction_weight").to_numpy(
            zero_copy_only=False
        ).astype(np.float32, copy=False)

        rows = len(user_values)
        users[offset : offset + rows] = user_values
        items[offset : offset + rows] = item_values
        weights[offset : offset + rows] = weight_values * np.float32(alpha)

        offset += rows
        print(
            f"Batch {batch_number:05d} | {rows:,} rows | "
            f"{offset:,}/{n_rows:,}"
        )

    if offset != n_rows:
        raise RuntimeError(
            f"Expected {n_rows:,} entries, loaded {offset:,}."
        )

    matrix = sparse.coo_matrix(
        (weights, (users, items)),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()

    del users, items, weights
    for path in (user_path, item_path, data_path):
        path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    memory_bytes = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )

    print(f"Shape       : {matrix.shape}")
    print(f"Non-zero    : {matrix.nnz:,}")
    print(f"CSR memory  : {human_size(memory_bytes)}")
    print(f"Completed in: {format_elapsed(elapsed)}")

    return matrix, {
        "rows": n_users,
        "columns": n_items,
        "nonzero_values": int(matrix.nnz),
        "density": float(matrix.nnz / (n_users * n_items)),
        "memory_bytes": int(memory_bytes),
        "memory_human": human_size(memory_bytes),
        "seconds": round(elapsed, 3),
    }


def train_als(
    user_item_matrix: sparse.csr_matrix,
    outputs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    print("\n" + "=" * 76)
    print("TRAINING IMPLICIT ALS MODEL")

    started = time.perf_counter()
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=args.random_state,
        num_threads=max(args.threads, 1),
        calculate_training_loss=True,
    )
    model.fit(user_item_matrix, show_progress=True)

    user_factors = np.asarray(model.user_factors, dtype=np.float32)
    item_factors = np.asarray(model.item_factors, dtype=np.float32)

    np.save(outputs["user_factors"], user_factors)
    np.save(outputs["item_factors"], item_factors)

    elapsed = time.perf_counter() - started
    return user_factors, item_factors, {
        "users": int(user_factors.shape[0]),
        "items": int(item_factors.shape[0]),
        "factors": int(item_factors.shape[1]),
        "user_factor_size_bytes": outputs["user_factors"].stat().st_size,
        "item_factor_size_bytes": outputs["item_factors"].stat().st_size,
        "seconds": round(elapsed, 3),
    }


def build_item_hnsw(
    item_factors: np.ndarray,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n" + "=" * 76)
    print("BUILDING COLLABORATIVE ITEM HNSW INDEX")

    started = time.perf_counter()
    embeddings = normalize(item_factors, norm="l2", copy=True).astype(
        np.float32, copy=False
    )

    index = hnswlib.Index(space="cosine", dim=args.factors)
    index.init_index(
        max_elements=embeddings.shape[0],
        ef_construction=args.hnsw_ef_construction,
        M=args.hnsw_m,
        random_seed=args.random_state,
    )
    index.set_num_threads(max(args.threads, 1))

    labels = np.arange(embeddings.shape[0], dtype=np.int64)
    index.add_items(
        embeddings,
        labels,
        num_threads=max(args.threads, 1),
    )
    index.set_ef(args.hnsw_ef_search)
    index.save_index(str(output_path))

    test_labels, test_distances = index.knn_query(
        embeddings[0].reshape(1, -1),
        k=min(5, embeddings.shape[0]),
    )
    if int(test_labels[0][0]) != 0:
        raise RuntimeError("Collaborative item index self-query failed.")

    elapsed = time.perf_counter() - started
    return {
        "items": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "index_size_bytes": output_path.stat().st_size,
        "index_size_human": human_size(output_path.stat().st_size),
        "self_query_label": int(test_labels[0][0]),
        "self_query_distance": float(test_distances[0][0]),
        "seconds": round(elapsed, 3),
    }


def validate_outputs(
    outputs: dict[str, Path],
    n_users: int,
    n_items: int,
    als_result: dict[str, Any],
) -> dict[str, Any]:
    user_map_rows = pq.ParquetFile(outputs["user_map"]).metadata.num_rows
    item_map_rows = pq.ParquetFile(outputs["item_map"]).metadata.num_rows

    if user_map_rows != n_users:
        raise RuntimeError("User map row count mismatch.")
    if item_map_rows != n_items:
        raise RuntimeError("Item map row count mismatch.")
    if als_result["users"] != n_users:
        raise RuntimeError("User-factor row count mismatch.")
    if als_result["items"] != n_items:
        raise RuntimeError("Item-factor row count mismatch.")

    return {
        "user_map_rows": user_map_rows,
        "item_map_rows": item_map_rows,
        "user_factors_exists": outputs["user_factors"].exists(),
        "item_factors_exists": outputs["item_factors"].exists(),
        "item_hnsw_exists": outputs["item_hnsw"].exists(),
    }


def main() -> None:
    args = parse_args()

    if args.factors <= 1:
        raise ValueError("--factors must be greater than 1.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if args.regularization <= 0:
        raise ValueError("--regularization must be positive.")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")
    if args.max_interactions_per_user < 0:
        raise ValueError("--max-interactions-per-user cannot be negative.")
    if args.scan_batch_size <= 0:
        raise ValueError("--scan-batch-size must be positive.")
    if args.source_partitions <= 0:
        raise ValueError("--source-partitions must be positive.")

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.input_dir or config["paths"]["processed_dir"]
    )
    output_processed_dir = resolve_path(
        args.output_dir or config["paths"]["processed_dir"]
    )

    inputs = require_inputs(processed_dir)
    outputs = prepare_outputs(output_processed_dir, args.overwrite)

    temp_dir = outputs["directory"] / ".duckdb_tmp"
    scratch_dir = outputs["directory"] / ".matrix_scratch"
    work_db = outputs["directory"] / ".collaborative_work.duckdb"

    temp_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    work_db.unlink(missing_ok=True)

    print("=" * 76)
    print("GOODREADS LOW-MEMORY COLLABORATIVE MODEL BUILDER")
    print(f"Interactions        : {inputs['interactions']}")
    print(f"Output              : {outputs['directory']}")
    print(f"ALS factors         : {args.factors}")
    print(f"ALS iterations      : {args.iterations}")
    print(f"Regularization      : {args.regularization}")
    print(f"Confidence alpha    : {args.alpha}")
    print(
        "Per-user cap       : "
        + (
            "ALL"
            if args.max_interactions_per_user == 0
            else f"{args.max_interactions_per_user:,}"
        )
    )
    print(f"Source partitions   : {args.source_partitions}")
    print(f"Threads             : {args.threads}")
    print(f"DuckDB memory limit : {args.memory_limit}")
    print("=" * 76)

    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    connection = duckdb.connect(str(work_db))
    connection.execute(f"SET threads = {max(args.threads, 1)}")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order = false")

    try:
        timings.update(build_dense_maps(connection, inputs, outputs))
        training_result = build_dense_training_data_partitioned(
            connection=connection,
            inputs=inputs,
            outputs=outputs,
            max_per_user=args.max_interactions_per_user,
            positive_only=args.positive_only,
            source_partitions=args.source_partitions,
        )
        timings["training_data_seconds"] = training_result["seconds"]
    finally:
        connection.close()
        work_db.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    n_users = pq.ParquetFile(outputs["user_map"]).metadata.num_rows
    n_items = pq.ParquetFile(outputs["item_map"]).metadata.num_rows

    user_item_matrix, matrix_result = build_sparse_matrix(
        training_dir=outputs["training_dir"],
        n_users=n_users,
        n_items=n_items,
        scan_batch_size=args.scan_batch_size,
        alpha=args.alpha,
        scratch_dir=scratch_dir,
    )
    timings["matrix_seconds"] = matrix_result["seconds"]

    user_factors, item_factors, als_result = train_als(
        user_item_matrix,
        outputs,
        args,
    )
    timings["als_seconds"] = als_result["seconds"]

    del user_item_matrix
    del user_factors

    hnsw_result = build_item_hnsw(
        item_factors,
        outputs["item_hnsw"],
        args,
    )
    timings["hnsw_seconds"] = hnsw_result["seconds"]
    del item_factors

    validation = validate_outputs(
        outputs,
        n_users,
        n_items,
        als_result,
    )

    if not args.keep_training_data:
        shutil.rmtree(outputs["training_dir"], ignore_errors=True)
    shutil.rmtree(scratch_dir, ignore_errors=True)

    total_seconds = time.perf_counter() - total_started
    timings["total_seconds"] = round(total_seconds, 3)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "implicit_als_collaborative_filtering",
        "architecture": "partition_wise_preprocessing",
        "input_interactions": str(inputs["interactions"]),
        "output_directory": str(outputs["directory"]),
        "settings": vars(args),
        "training_data": training_result,
        "matrix": matrix_result,
        "als": als_result,
        "item_index": hnsw_result,
        "validation": validation,
        "timings": timings,
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("COLLABORATIVE MODEL BUILD COMPLETE")
    print(f"Users          : {n_users:,}")
    print(f"Items          : {n_items:,}")
    print(f"Interactions   : {matrix_result['nonzero_values']:,}")
    print(f"Matrix memory  : {matrix_result['memory_human']}")
    print(f"Factors        : {args.factors}")
    print(
        "Factor files   : "
        + human_size(
            als_result["user_factor_size_bytes"]
            + als_result["item_factor_size_bytes"]
        )
    )
    print(f"Item HNSW size : {hnsw_result['index_size_human']}")
    print(f"Total elapsed  : {format_elapsed(total_seconds)}")
    print(f"Metadata       : {outputs['metadata']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
