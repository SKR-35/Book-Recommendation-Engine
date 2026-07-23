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

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402

try:
    import hnswlib
except ImportError as exc:
    raise ImportError(
        "hnswlib is required. Install it in the active environment with: "
        "pip install hnswlib"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a low-memory content index from TF-IDF shards by fitting "
            "TruncatedSVD and incrementally constructing an HNSW index."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument("--svd-fit-sample-size", type=str, default="100000")
    parser.add_argument("--svd-iterations", type=int, default=7)
    parser.add_argument(
        "--embedding-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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


def parse_sample_size(value: str) -> int | None:
    cleaned = str(value).strip().lower()
    if cleaned in {"all", "0"}:
        return None
    try:
        parsed = int(cleaned)
    except ValueError as exc:
        raise ValueError(
            "--svd-fit-sample-size must be a positive integer, 0, or 'all'."
        ) from exc
    if parsed <= 0:
        raise ValueError(
            "--svd-fit-sample-size must be a positive integer, 0, or 'all'."
        )
    return parsed


def require_inputs(content_dir: Path) -> tuple[list[Path], Path]:
    matrix_dir = content_dir / "tfidf_matrix"
    row_map = content_dir / "content_row_map.parquet"

    if not matrix_dir.exists():
        raise FileNotFoundError(f"Missing TF-IDF matrix directory: {matrix_dir}")
    if not row_map.exists():
        raise FileNotFoundError(f"Missing row map: {row_map}")

    shards = sorted(matrix_dir.glob("part-*.npz"))
    if not shards:
        raise FileNotFoundError(f"No TF-IDF shards found in {matrix_dir}")

    return shards, row_map


def prepare_outputs(content_dir: Path, overwrite: bool) -> dict[str, Path]:
    index_dir = content_dir / "index"

    if index_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{index_dir} already exists. Run again with --overwrite."
            )
        shutil.rmtree(index_dir)

    outputs = {
        "directory": index_dir,
        "svd_model": index_dir / "truncated_svd.joblib",
        "hnsw_index": index_dir / "content_hnsw.bin",
        "embeddings_dir": index_dir / "embeddings",
        "index_map": index_dir / "content_index_map.parquet",
        "metadata": index_dir / "content_index_metadata.json",
    }

    index_dir.mkdir(parents=True, exist_ok=True)
    outputs["embeddings_dir"].mkdir(parents=True, exist_ok=True)
    return outputs


def shard_row_counts(shards: list[Path]) -> list[int]:
    counts: list[int] = []
    for path in shards:
        matrix = sparse.load_npz(path)
        counts.append(matrix.shape[0])
        del matrix
    return counts


def allocate_sample_counts(
    row_counts: list[int],
    requested: int | None,
) -> tuple[list[int], int, int]:
    total = sum(row_counts)
    if requested is None or requested >= total:
        return row_counts, total, total

    raw = [requested * rows / total for rows in row_counts]
    counts = [int(math.floor(value)) for value in raw]
    remainder = requested - sum(counts)

    order = sorted(
        range(len(raw)),
        key=lambda i: raw[i] - counts[i],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    counts = [min(count, rows) for count, rows in zip(counts, row_counts)]
    return counts, total, sum(counts)


def build_fit_sample(
    shards: list[Path],
    sample_counts: list[int],
    random_state: int,
) -> sparse.csr_matrix:
    print("\n" + "=" * 76)
    print("BUILDING SVD FIT SAMPLE")

    rng = np.random.default_rng(random_state)
    samples: list[sparse.csr_matrix] = []

    for shard_id, (path, count) in enumerate(zip(shards, sample_counts)):
        if count <= 0:
            continue

        matrix = sparse.load_npz(path).tocsr()
        if count >= matrix.shape[0]:
            selected = matrix
        else:
            indices = np.sort(
                rng.choice(matrix.shape[0], size=count, replace=False)
            )
            selected = matrix[indices]

        samples.append(selected)
        print(
            f"Shard {shard_id:05d} | "
            f"{selected.shape[0]:,} / {matrix.shape[0]:,} rows"
        )
        del matrix

    if not samples:
        raise RuntimeError("SVD sample is empty.")

    return sparse.vstack(samples, format="csr")


def fit_svd(
    sample: sparse.csr_matrix,
    output_path: Path,
    components: int,
    iterations: int,
    random_state: int,
) -> tuple[TruncatedSVD, dict[str, Any]]:
    if components >= min(sample.shape):
        raise ValueError(
            f"--svd-components={components} is too large for sample shape "
            f"{sample.shape}."
        )

    print("\n" + "=" * 76)
    print("FITTING TRUNCATED SVD")
    started = time.perf_counter()

    model = TruncatedSVD(
        n_components=components,
        n_iter=iterations,
        algorithm="randomized",
        random_state=random_state,
    )
    model.fit(sample)
    joblib.dump(model, output_path, compress=3)

    elapsed = time.perf_counter() - started
    explained = float(model.explained_variance_ratio_.sum())

    print(f"Sample shape       : {sample.shape}")
    print(f"Latent dimensions  : {components:,}")
    print(f"Explained variance : {explained:.4%}")
    print(f"Completed in       : {format_elapsed(elapsed)}")

    return model, {
        "sample_rows": int(sample.shape[0]),
        "tfidf_features": int(sample.shape[1]),
        "components": components,
        "explained_variance_ratio": explained,
        "seconds": round(elapsed, 3),
    }


def build_hnsw_index(
    model: TruncatedSVD,
    shards: list[Path],
    row_map_path: Path,
    outputs: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n" + "=" * 76)
    print("BUILDING HNSW CONTENT INDEX")

    row_map = pd.read_parquet(row_map_path).sort_values(
        "global_row"
    ).reset_index(drop=True)
    total_rows = len(row_map)

    index = hnswlib.Index(space="cosine", dim=args.svd_components)
    index.init_index(
        max_elements=total_rows,
        ef_construction=args.hnsw_ef_construction,
        M=args.hnsw_m,
        random_seed=args.random_state,
    )
    index.set_num_threads(1)

    disk_dtype = (
        np.float16 if args.embedding_dtype == "float16" else np.float32
    )
    global_offset = 0
    embedding_bytes = 0
    started = time.perf_counter()

    for shard_id, path in enumerate(shards):
        matrix = sparse.load_npz(path).tocsr()
        embeddings = model.transform(matrix).astype(np.float32, copy=False)
        embeddings = normalize(embeddings, norm="l2", copy=False)

        rows = embeddings.shape[0]
        labels = np.arange(
            global_offset,
            global_offset + rows,
            dtype=np.int64,
        )
        index.add_items(embeddings, labels, num_threads=1)

        embedding_path = (
            outputs["embeddings_dir"] / f"part-{shard_id:05d}.npy"
        )
        np.save(
            embedding_path,
            embeddings.astype(disk_dtype, copy=False),
        )
        embedding_bytes += embedding_path.stat().st_size

        print(
            f"Shard {shard_id:05d} | {rows:,} rows | "
            f"labels {global_offset:,}–{global_offset + rows - 1:,}"
        )

        global_offset += rows
        del matrix
        del embeddings

    if global_offset != total_rows:
        raise RuntimeError(
            f"Indexed {global_offset:,} rows but row map contains "
            f"{total_rows:,}."
        )

    index.set_ef(args.hnsw_ef_search)
    index.save_index(str(outputs["hnsw_index"]))
    row_map.to_parquet(
        outputs["index_map"],
        index=False,
        compression="zstd",
    )

    elapsed = time.perf_counter() - started

    return {
        "rows": total_rows,
        "dimensions": args.svd_components,
        "shards": len(shards),
        "embedding_dtype": args.embedding_dtype,
        "embedding_size_bytes": embedding_bytes,
        "embedding_size_human": human_size(embedding_bytes),
        "index_size_bytes": outputs["hnsw_index"].stat().st_size,
        "index_size_human": human_size(
            outputs["hnsw_index"].stat().st_size
        ),
        "seconds": round(elapsed, 3),
    }


def validate_index(
    outputs: dict[str, Path],
    rows: int,
    dimensions: int,
) -> dict[str, Any]:
    map_rows = pq.ParquetFile(outputs["index_map"]).metadata.num_rows
    if map_rows != rows:
        raise RuntimeError("Index-map row count mismatch.")

    test_index = hnswlib.Index(space="cosine", dim=dimensions)
    test_index.load_index(str(outputs["hnsw_index"]), max_elements=rows)
    test_index.set_ef(50)

    first_embedding = sorted(
        outputs["embeddings_dir"].glob("part-*.npy")
    )[0]
    query = np.asarray(
        np.load(first_embedding, mmap_mode="r")[0],
        dtype=np.float32,
    ).reshape(1, -1)

    labels, distances = test_index.knn_query(query, k=min(5, rows))
    nearest_label = int(labels[0][0])
    nearest_distance = float(distances[0][0])

    if nearest_label != 0:
        raise RuntimeError(
            f"Self-query failed: expected label 0, got {nearest_label}."
        )

    return {
        "index_map_rows": map_rows,
        "self_query_nearest_label": nearest_label,
        "self_query_nearest_distance": nearest_distance,
    }


def main() -> None:
    args = parse_args()
    requested_sample = parse_sample_size(args.svd_fit_sample_size)

    if args.svd_components <= 1:
        raise ValueError("--svd-components must be greater than 1.")
    if args.svd_iterations <= 0:
        raise ValueError("--svd-iterations must be positive.")
    if args.hnsw_m <= 1:
        raise ValueError("--hnsw-m must be greater than 1.")

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.input_dir or config["paths"]["processed_dir"]
    )
    output_processed_dir = resolve_path(
        args.output_dir or config["paths"]["processed_dir"]
    )

    content_dir = processed_dir / "content"
    output_content_dir = output_processed_dir / "content"

    shards, row_map = require_inputs(content_dir)
    outputs = prepare_outputs(output_content_dir, args.overwrite)

    print("=" * 76)
    print("GOODREADS CONTENT INDEX BUILDER")
    print(f"Input content      : {content_dir}")
    print(f"Output index       : {outputs['directory']}")
    print(
        "SVD fit rows      : "
        + ("ALL" if requested_sample is None else f"{requested_sample:,}")
    )
    print(f"SVD dimensions    : {args.svd_components:,}")
    print(f"HNSW M            : {args.hnsw_m}")
    print(f"HNSW ef construct : {args.hnsw_ef_construction}")
    print(f"HNSW ef search    : {args.hnsw_ef_search}")
    print("=" * 76)

    total_started = time.perf_counter()

    counts = shard_row_counts(shards)
    sample_counts, total_rows, actual_sample = allocate_sample_counts(
        counts,
        requested_sample,
    )

    sample = build_fit_sample(
        shards,
        sample_counts,
        args.random_state,
    )

    model, svd_result = fit_svd(
        sample=sample,
        output_path=outputs["svd_model"],
        components=args.svd_components,
        iterations=args.svd_iterations,
        random_state=args.random_state,
    )
    del sample

    index_result = build_hnsw_index(
        model=model,
        shards=shards,
        row_map_path=row_map,
        outputs=outputs,
        args=args,
    )

    validation = validate_index(
        outputs=outputs,
        rows=index_result["rows"],
        dimensions=args.svd_components,
    )

    total_seconds = time.perf_counter() - total_started

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "tfidf_svd_hnsw_content_index",
        "input_content_directory": str(content_dir),
        "output_directory": str(outputs["directory"]),
        "settings": {
            "svd_components": args.svd_components,
            "svd_fit_sample_size_requested": args.svd_fit_sample_size,
            "svd_fit_sample_size_actual": actual_sample,
            "svd_fit_used_full_corpus": actual_sample == total_rows,
            "svd_iterations": args.svd_iterations,
            "embedding_dtype": args.embedding_dtype,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_construction": args.hnsw_ef_construction,
            "hnsw_ef_search": args.hnsw_ef_search,
            "random_state": args.random_state,
            "overwrite": args.overwrite,
        },
        "svd": svd_result,
        "index": index_result,
        "validation": validation,
        "total_seconds": round(total_seconds, 3),
    }

    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("CONTENT INDEX BUILD COMPLETE")
    print(f"Indexed books      : {index_result['rows']:,}")
    print(f"Latent dimensions : {index_result['dimensions']:,}")
    print(f"Embedding shards  : {index_result['shards']:,}")
    print(f"Embedding size    : {index_result['embedding_size_human']}")
    print(f"HNSW index size   : {index_result['index_size_human']}")
    print(
        f"Explained variance: "
        f"{svd_result['explained_variance_ratio']:.4%}"
    )
    print(f"Total elapsed     : {format_elapsed(total_seconds)}")
    print(f"Metadata          : {outputs['metadata']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
