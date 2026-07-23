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
import hnswlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a unified hybrid book index by aligning content and "
            "collaborative embeddings, then concatenating their normalized "
            "representations into one HNSW retrieval space."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--content-weight", type=float, default=0.45)
    parser.add_argument("--collaborative-weight", type=float, default=0.40)
    parser.add_argument("--popularity-weight", type=float, default=0.10)
    parser.add_argument("--novelty-weight", type=float, default=0.05)

    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument(
        "--embedding-dtype",
        choices=["float16", "float32"],
        default="float16",
    )

    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)

    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--random-state", type=int, default=42)
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


def validate_weights(args: argparse.Namespace) -> dict[str, float]:
    weights = {
        "content": args.content_weight,
        "collaborative": args.collaborative_weight,
        "popularity": args.popularity_weight,
        "novelty": args.novelty_weight,
    }

    if any(value < 0 for value in weights.values()):
        raise ValueError("Hybrid weights cannot be negative.")

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one hybrid weight must be positive.")

    return {name: value / total for name, value in weights.items()}


def require_inputs(processed_dir: Path) -> dict[str, Path]:
    inputs = {
        "content_index_map": (
            processed_dir / "content" / "index" / "content_index_map.parquet"
        ),
        "content_embeddings": (
            processed_dir / "content" / "index" / "embeddings"
        ),
        "content_metadata": (
            processed_dir / "content" / "index" / "content_index_metadata.json"
        ),
        "collaborative_item_map": (
            processed_dir / "collaborative" / "item_map.parquet"
        ),
        "collaborative_item_factors": (
            processed_dir / "collaborative" / "item_factors.npy"
        ),
        "collaborative_metadata": (
            processed_dir / "collaborative" / "collaborative_metadata.json"
        ),
        "popularity_scores": (
            processed_dir / "popularity" / "scored_books.parquet"
        ),
    }

    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required hybrid-model inputs are missing:\n  - "
            + "\n  - ".join(missing)
        )

    if not any(inputs["content_embeddings"].glob("part-*.npy")):
        raise FileNotFoundError(
            f"No content embedding shards found in "
            f"{inputs['content_embeddings']}"
        )

    return inputs


def prepare_outputs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    hybrid_dir = output_dir / "hybrid"

    if hybrid_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{hybrid_dir} already exists. Run again with --overwrite."
            )
        shutil.rmtree(hybrid_dir)

    outputs = {
        "directory": hybrid_dir,
        "item_map": hybrid_dir / "hybrid_item_map.parquet",
        "embeddings": hybrid_dir / "hybrid_embeddings.npy",
        "hnsw_index": hybrid_dir / "hybrid_hnsw.bin",
        "scoring_config": hybrid_dir / "hybrid_scoring_config.json",
        "metadata": hybrid_dir / "hybrid_metadata.json",
    }

    hybrid_dir.mkdir(parents=True, exist_ok=True)
    return outputs


def build_hybrid_item_map(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    output_path: Path,
) -> float:
    query = f"""
    COPY (
        WITH joined AS (
            SELECT
                c.global_row AS content_global_row,
                c.shard_id AS content_shard_id,
                c.shard_row AS content_shard_row,
                c.book_id,
                c.item_id,
                c.work_id,
                c.title,
                c.primary_author,
                c.primary_genre,
                c.language_code,
                c.publication_year,
                c.image_url,
                c.bayesian_rating,
                c.interaction_count,
                i.item_index AS collaborative_item_index,
                i.average_user_rating,
                i.rating_count AS collaborative_rating_count,
                p.wilson_score,
                p.bayesian_score,
                p.novelty_score
            FROM read_parquet(
                '{sql_path(inputs["content_index_map"])}'
            ) c
            INNER JOIN read_parquet(
                '{sql_path(inputs["collaborative_item_map"])}'
            ) i
                ON TRY_CAST(c.book_id AS BIGINT)
                 = TRY_CAST(i.local_book_id AS BIGINT)
            LEFT JOIN read_parquet(
                '{sql_path(inputs["popularity_scores"])}'
            ) p
                ON TRY_CAST(c.book_id AS BIGINT)
                 = TRY_CAST(p.book_id AS BIGINT)
        ),
        normalized AS (
            SELECT
                *,
                COALESCE(
                    CUME_DIST() OVER (
                        ORDER BY wilson_score NULLS FIRST
                    ),
                    0.0
                ) AS popularity_percentile,
                COALESCE(
                    CUME_DIST() OVER (
                        ORDER BY novelty_score NULLS FIRST
                    ),
                    0.0
                ) AS novelty_percentile
            FROM joined
        )
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY content_global_row
            ) - 1 AS hybrid_index,
            *
        FROM normalized
        ORDER BY content_global_row
    )
    TO '{sql_path(output_path)}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    return execute(connection, "BUILDING HYBRID ITEM ALIGNMENT MAP", query)


def read_dimensions(inputs: dict[str, Path]) -> tuple[int, int]:
    content_meta = json.loads(
        inputs["content_metadata"].read_text(encoding="utf-8")
    )
    collaborative_meta = json.loads(
        inputs["collaborative_metadata"].read_text(encoding="utf-8")
    )

    content_dimensions = int(content_meta["index"]["dimensions"])
    collaborative_dimensions = int(
        collaborative_meta["als"]["factors"]
    )
    return content_dimensions, collaborative_dimensions


def load_content_shards(directory: Path) -> dict[int, np.ndarray]:
    shards: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob("part-*.npy")):
        shard_id = int(path.stem.split("-")[-1])
        shards[shard_id] = np.load(path, mmap_mode="r")
    return shards


def build_hybrid_embeddings_and_index(
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    weights: dict[str, float],
    content_dimensions: int,
    collaborative_dimensions: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n" + "=" * 76)
    print("BUILDING HYBRID EMBEDDINGS AND HNSW INDEX")

    started = time.perf_counter()
    item_map = pd.read_parquet(outputs["item_map"])
    item_map = item_map.sort_values("hybrid_index").reset_index(drop=True)

    total_rows = len(item_map)
    hybrid_dimensions = content_dimensions + collaborative_dimensions

    disk_dtype = (
        np.float16 if args.embedding_dtype == "float16" else np.float32
    )

    embedding_store = np.lib.format.open_memmap(
        outputs["embeddings"],
        mode="w+",
        dtype=disk_dtype,
        shape=(total_rows, hybrid_dimensions),
    )

    content_shards = load_content_shards(inputs["content_embeddings"])
    collaborative_factors = np.load(
        inputs["collaborative_item_factors"],
        mmap_mode="r",
    )

    index = hnswlib.Index(space="cosine", dim=hybrid_dimensions)
    index.init_index(
        max_elements=total_rows,
        ef_construction=args.hnsw_ef_construction,
        M=args.hnsw_m,
        random_seed=args.random_state,
    )
    index.set_num_threads(max(args.threads, 1))

    content_scale = math.sqrt(
        weights["content"]
        / max(weights["content"] + weights["collaborative"], 1e-12)
    )
    collaborative_scale = math.sqrt(
        weights["collaborative"]
        / max(weights["content"] + weights["collaborative"], 1e-12)
    )

    for start in range(0, total_rows, args.batch_size):
        end = min(start + args.batch_size, total_rows)
        batch = item_map.iloc[start:end]
        rows = len(batch)

        content_batch = np.empty(
            (rows, content_dimensions),
            dtype=np.float32,
        )

        for shard_id, group in batch.groupby("content_shard_id", sort=False):
            positions = group.index.to_numpy() - start
            shard_rows = group["content_shard_row"].to_numpy(dtype=np.int64)
            shard = content_shards[int(shard_id)]
            content_batch[positions] = np.asarray(
                shard[shard_rows],
                dtype=np.float32,
            )

        collaborative_indices = batch[
            "collaborative_item_index"
        ].to_numpy(dtype=np.int64)
        collaborative_batch = np.asarray(
            collaborative_factors[collaborative_indices],
            dtype=np.float32,
        )

        content_batch = normalize(
            content_batch,
            norm="l2",
            copy=False,
        )
        collaborative_batch = normalize(
            collaborative_batch,
            norm="l2",
            copy=False,
        )

        hybrid_batch = np.hstack(
            [
                content_batch * content_scale,
                collaborative_batch * collaborative_scale,
            ]
        ).astype(np.float32, copy=False)

        hybrid_batch = normalize(
            hybrid_batch,
            norm="l2",
            copy=False,
        )

        labels = batch["hybrid_index"].to_numpy(dtype=np.int64)
        index.add_items(
            hybrid_batch,
            labels,
            num_threads=max(args.threads, 1),
        )

        embedding_store[start:end] = hybrid_batch.astype(
            disk_dtype,
            copy=False,
        )
        embedding_store.flush()

        print(
            f"Rows {start:,}–{end - 1:,} | "
            f"{rows:,} embeddings"
        )

        del content_batch
        del collaborative_batch
        del hybrid_batch

    index.set_ef(args.hnsw_ef_search)
    index.save_index(str(outputs["hnsw_index"]))

    query = np.asarray(
        embedding_store[0],
        dtype=np.float32,
    ).reshape(1, -1)
    labels, distances = index.knn_query(
        query,
        k=min(5, total_rows),
    )

    if int(labels[0][0]) != 0:
        raise RuntimeError(
            f"Hybrid self-query failed: expected 0, got {labels[0][0]}."
        )

    elapsed = time.perf_counter() - started

    return {
        "rows": total_rows,
        "content_dimensions": content_dimensions,
        "collaborative_dimensions": collaborative_dimensions,
        "hybrid_dimensions": hybrid_dimensions,
        "embedding_dtype": args.embedding_dtype,
        "embedding_size_bytes": outputs["embeddings"].stat().st_size,
        "embedding_size_human": human_size(
            outputs["embeddings"].stat().st_size
        ),
        "index_size_bytes": outputs["hnsw_index"].stat().st_size,
        "index_size_human": human_size(
            outputs["hnsw_index"].stat().st_size
        ),
        "self_query_label": int(labels[0][0]),
        "self_query_distance": float(distances[0][0]),
        "content_embedding_scale": content_scale,
        "collaborative_embedding_scale": collaborative_scale,
        "seconds": round(elapsed, 3),
    }


def validate_outputs(
    outputs: dict[str, Path],
    result: dict[str, Any],
) -> dict[str, Any]:
    map_rows = pq.ParquetFile(outputs["item_map"]).metadata.num_rows

    if map_rows != result["rows"]:
        raise RuntimeError("Hybrid item-map row count mismatch.")
    if not outputs["embeddings"].exists():
        raise RuntimeError("Hybrid embedding file is missing.")
    if not outputs["hnsw_index"].exists():
        raise RuntimeError("Hybrid HNSW index is missing.")

    map_frame = pd.read_parquet(
        outputs["item_map"],
        columns=[
            "hybrid_index",
            "book_id",
            "collaborative_item_index",
            "popularity_percentile",
            "novelty_percentile",
        ],
    )

    duplicate_indices = int(
        map_frame["hybrid_index"].duplicated().sum()
    )
    duplicate_books = int(map_frame["book_id"].duplicated().sum())

    if duplicate_indices != 0:
        raise RuntimeError("Duplicate hybrid indices were created.")
    if duplicate_books != 0:
        raise RuntimeError("Duplicate book IDs were created.")

    return {
        "item_map_rows": map_rows,
        "duplicate_hybrid_indices": duplicate_indices,
        "duplicate_books": duplicate_books,
        "minimum_popularity_percentile": float(
            map_frame["popularity_percentile"].min()
        ),
        "maximum_popularity_percentile": float(
            map_frame["popularity_percentile"].max()
        ),
        "minimum_novelty_percentile": float(
            map_frame["novelty_percentile"].min()
        ),
        "maximum_novelty_percentile": float(
            map_frame["novelty_percentile"].max()
        ),
    }


def main() -> None:
    args = parse_args()
    weights = validate_weights(args)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.hnsw_m <= 1:
        raise ValueError("--hnsw-m must be greater than 1.")
    if args.hnsw_ef_construction <= 0:
        raise ValueError("--hnsw-ef-construction must be positive.")
    if args.hnsw_ef_search <= 0:
        raise ValueError("--hnsw-ef-search must be positive.")

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
    work_db = outputs["directory"] / ".hybrid_work.duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    work_db.unlink(missing_ok=True)

    print("=" * 76)
    print("GOODREADS HYBRID MODEL BUILDER")
    print(f"Processed input      : {processed_dir}")
    print(f"Output               : {outputs['directory']}")
    print(f"Content weight       : {weights['content']:.3f}")
    print(f"Collaborative weight : {weights['collaborative']:.3f}")
    print(f"Popularity weight    : {weights['popularity']:.3f}")
    print(f"Novelty weight       : {weights['novelty']:.3f}")
    print(f"Batch size           : {args.batch_size:,}")
    print(f"Threads              : {args.threads}")
    print(f"Memory limit         : {args.memory_limit}")
    print("=" * 76)

    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    connection = duckdb.connect(str(work_db))
    connection.execute(f"SET threads = {max(args.threads, 1)}")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")
    connection.execute(f"SET temp_directory = '{sql_path(temp_dir)}'")
    connection.execute("SET preserve_insertion_order = false")

    try:
        timings["item_map_seconds"] = build_hybrid_item_map(
            connection,
            inputs,
            outputs["item_map"],
        )
    finally:
        connection.close()
        work_db.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    content_dimensions, collaborative_dimensions = read_dimensions(inputs)

    index_result = build_hybrid_embeddings_and_index(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        content_dimensions=content_dimensions,
        collaborative_dimensions=collaborative_dimensions,
        args=args,
    )
    timings["index_seconds"] = index_result["seconds"]

    validation = validate_outputs(outputs, index_result)

    scoring_config = {
        "retrieval": {
            "content_weight": weights["content"],
            "collaborative_weight": weights["collaborative"],
            "content_embedding_scale": index_result[
                "content_embedding_scale"
            ],
            "collaborative_embedding_scale": index_result[
                "collaborative_embedding_scale"
            ],
        },
        "reranking": {
            "similarity_weight": weights["content"]
            + weights["collaborative"],
            "popularity_weight": weights["popularity"],
            "novelty_weight": weights["novelty"],
            "formula": (
                "final_score = similarity_weight * hybrid_similarity "
                "+ popularity_weight * popularity_percentile "
                "+ novelty_weight * novelty_percentile"
            ),
        },
        "cold_start": {
            "strategy": "content_only_then_popularity_novelty_rerank",
            "google_books_supported": True,
            "google_books_api_key_required": False,
            "google_books_api_key_optional": True,
        },
    }
    outputs["scoring_config"].write_text(
        json.dumps(scoring_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_seconds = time.perf_counter() - total_started
    timings["total_seconds"] = round(total_seconds, 3)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "concatenated_content_collaborative_hnsw",
        "input_directory": str(processed_dir),
        "output_directory": str(outputs["directory"]),
        "weights": weights,
        "settings": {
            "batch_size": args.batch_size,
            "embedding_dtype": args.embedding_dtype,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "hnsw_ef_search": args.hnsw_ef_search,
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "random_state": args.random_state,
            "overwrite": args.overwrite,
        },
        "index": index_result,
        "validation": validation,
        "timings": timings,
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("HYBRID MODEL BUILD COMPLETE")
    print(f"Hybrid books      : {index_result['rows']:,}")
    print(f"Content dims      : {content_dimensions:,}")
    print(f"Collaborative dims: {collaborative_dimensions:,}")
    print(f"Hybrid dims       : {index_result['hybrid_dimensions']:,}")
    print(f"Embedding size    : {index_result['embedding_size_human']}")
    print(f"HNSW index size   : {index_result['index_size_human']}")
    print(f"Total elapsed     : {format_elapsed(total_seconds)}")
    print(f"Metadata          : {outputs['metadata']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
