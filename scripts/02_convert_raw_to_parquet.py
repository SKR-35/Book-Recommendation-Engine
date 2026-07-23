from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


JSON_DATASETS = {
    "books": "books.parquet",
    "authors": "authors.parquet",
    "genres": "genres.parquet",
    "works": "works.parquet",
}

INTEGER_COLUMNS = {
    "book_id", "work_id", "author_id", "best_book_id", "books_count",
    "ratings_count", "text_reviews_count", "original_publication_year",
    "original_publication_month", "original_publication_day",
    "publication_year", "publication_month", "publication_day", "num_pages",
}
FLOAT_COLUMNS = {"average_rating"}
BOOLEAN_COLUMNS = {"is_ebook"}

INTERACTION_DTYPES = {
    "user_id": "int32",
    "book_id": "int32",
    "is_read": "int8",
    "rating": "int8",
    "is_reviewed": "int8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw UCSD Goodreads files to RAM-friendly Parquet outputs."
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json-chunk-size", type=int, default=50_000)
    parser.add_argument("--csv-chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-hash", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:,.2f} {unit}"
        value /= 1024
    return f"{value:,.2f} TB"


def format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        if column in BOOLEAN_COLUMNS:
            normalized[column] = normalized[column].map(
                {
                    True: True, False: False, "true": True, "false": False,
                    "True": True, "False": False, 1: True, 0: False,
                }
            ).astype("boolean")
        elif column in INTEGER_COLUMNS:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            ).astype("Int64")
        elif column in FLOAT_COLUMNS:
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            ).astype("Float64")
        elif normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(json_safe).astype("string")
    return normalized


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays: list[pa.Array] = []
    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]
            if not column.type.equals(field.type):
                column = column.cast(field.type, safe=False)
        else:
            column = pa.nulls(table.num_rows, type=field.type)
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def iter_json_lines(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    buffer: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                buffer.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_number}: {exc}"
                ) from exc
            if len(buffer) >= chunk_size:
                yield pd.DataFrame.from_records(buffer)
                buffer.clear()
    if buffer:
        yield pd.DataFrame.from_records(buffer)


def prepare_output(path: Path, overwrite: bool) -> bool:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return True
    if not overwrite:
        print(f"SKIP    Output already exists: {path}")
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def convert_json_dataset(
    name: str,
    source: Path,
    destination: Path,
    chunk_size: int,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print(name.upper())
    print(f"Source : {source}")
    print(f"Output : {destination}")

    if not source.exists():
        print(f"WARNING Missing source file: {source}")
        return {"status": "missing", "rows": 0}
    if not prepare_output(destination, overwrite):
        return {"status": "skipped", "rows": None}

    started = time.perf_counter()
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows_written = 0
    chunks_written = 0

    try:
        progress = tqdm(
            iter_json_lines(source, chunk_size),
            desc=f"Converting {name}",
            unit="chunk",
        )
        for frame in progress:
            table = pa.Table.from_pandas(
                normalize_dataframe(frame), preserve_index=False
            )
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    destination,
                    schema=schema,
                    compression=None if compression == "none" else compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            else:
                assert schema is not None
                table = align_table_to_schema(table, schema)

            writer.write_table(table)
            rows_written += table.num_rows
            chunks_written += 1
            progress.set_postfix(rows=f"{rows_written:,}")
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"No rows were written for {name}.")

    validation_rows = pq.ParquetFile(destination).metadata.num_rows
    if validation_rows != rows_written:
        raise RuntimeError(
            f"Validation failed for {name}: wrote {rows_written:,}, "
            f"Parquet reports {validation_rows:,}."
        )

    elapsed = time.perf_counter() - started
    output_size = destination.stat().st_size
    reduction = 1 - (output_size / source.stat().st_size)

    print(f"Rows       : {rows_written:,}")
    print(f"Chunks     : {chunks_written:,}")
    print(f"Raw size   : {human_size(source.stat().st_size)}")
    print(f"Parquet    : {human_size(output_size)}")
    print(f"Reduction  : {reduction:.1%}")
    print(f"Elapsed    : {format_elapsed(elapsed)}")
    print("Validation : OK")

    return {
        "status": "converted",
        "rows": rows_written,
        "chunks": chunks_written,
        "output_size_bytes": output_size,
        "elapsed_seconds": round(elapsed, 3),
    }


def convert_book_id_map(
    source: Path,
    destination: Path,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("BOOK ID MAP")
    print(f"Source : {source}")
    print(f"Output : {destination}")

    if not source.exists():
        print(f"WARNING Missing source file: {source}")
        return {"status": "missing", "rows": 0}
    if not prepare_output(destination, overwrite):
        return {"status": "skipped", "rows": None}

    started = time.perf_counter()
    frame = pd.read_csv(source)
    for column in frame.columns:
        if column in {"book_id_csv", "book_id"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        elif frame[column].dtype == "object":
            frame[column] = frame[column].astype("string")

    frame.to_parquet(
        destination,
        index=False,
        compression=None if compression == "none" else compression,
    )
    rows_written = len(frame)
    validation_rows = pq.ParquetFile(destination).metadata.num_rows
    if validation_rows != rows_written:
        raise RuntimeError("Validation failed for book_id_map.")

    elapsed = time.perf_counter() - started
    output_size = destination.stat().st_size
    reduction = 1 - (output_size / source.stat().st_size)

    print(f"Rows       : {rows_written:,}")
    print(f"Raw size   : {human_size(source.stat().st_size)}")
    print(f"Parquet    : {human_size(output_size)}")
    print(f"Reduction  : {reduction:.1%}")
    print(f"Elapsed    : {format_elapsed(elapsed)}")
    print("Validation : OK")

    return {
        "status": "converted",
        "rows": rows_written,
        "chunks": 1,
        "output_size_bytes": output_size,
        "elapsed_seconds": round(elapsed, 3),
    }


def convert_interactions(
    source: Path,
    destination_dir: Path,
    chunk_size: int,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("INTERACTIONS")
    print(f"Source : {source}")
    print(f"Output : {destination_dir}")

    if not source.exists():
        print(f"WARNING Missing source file: {source}")
        return {"status": "missing", "rows": 0}
    if not prepare_output(destination_dir, overwrite):
        return {"status": "skipped", "rows": None}

    destination_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows_written = 0
    parts_written = 0

    reader = pd.read_csv(source, chunksize=chunk_size, dtype=INTERACTION_DTYPES)
    progress = tqdm(reader, desc="Converting interactions", unit="chunk")

    for part_number, frame in enumerate(progress):
        missing = [column for column in INTERACTION_DTYPES if column not in frame.columns]
        if missing:
            raise ValueError(f"Interactions file is missing columns: {missing}")

        frame = frame[list(INTERACTION_DTYPES)]
        output_path = destination_dir / f"part-{part_number:05d}.parquet"
        frame.to_parquet(
            output_path,
            index=False,
            compression=None if compression == "none" else compression,
        )
        rows_written += len(frame)
        parts_written += 1
        progress.set_postfix(rows=f"{rows_written:,}", parts=parts_written)

    if parts_written == 0:
        shutil.rmtree(destination_dir, ignore_errors=True)
        raise RuntimeError("No interaction rows were written.")

    validation_rows = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in destination_dir.glob("part-*.parquet")
    )
    if validation_rows != rows_written:
        raise RuntimeError(
            f"Interaction validation failed: wrote {rows_written:,}, "
            f"Parquet reports {validation_rows:,}."
        )

    elapsed = time.perf_counter() - started
    output_size = sum(path.stat().st_size for path in destination_dir.glob("*.parquet"))
    reduction = 1 - (output_size / source.stat().st_size)

    print(f"Rows       : {rows_written:,}")
    print(f"Parts      : {parts_written:,}")
    print(f"Raw size   : {human_size(source.stat().st_size)}")
    print(f"Parquet    : {human_size(output_size)}")
    print(f"Reduction  : {reduction:.1%}")
    print(f"Elapsed    : {format_elapsed(elapsed)}")
    print("Validation : OK")

    return {
        "status": "converted",
        "rows": rows_written,
        "parts": parts_written,
        "output_size_bytes": output_size,
        "elapsed_seconds": round(elapsed, 3),
    }


def raw_file_metadata(path: Path, calculate_hash: bool) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    metadata: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_human": human_size(path.stat().st_size),
        "modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    if calculate_hash:
        print(f"Hashing {path.name}...")
        metadata["sha256"] = sha256_file(path)
    return metadata


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    raw_dir = resolve_path(args.raw_dir or config["paths"]["raw_dir"])
    output_dir = resolve_path(args.output_dir or config["paths"]["interim_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("UCSD GOODREADS RAW -> PARQUET")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Compression  : {args.compression}")
    print(f"Overwrite    : {args.overwrite}")
    print(f"SHA256       : {not args.skip_hash}")
    print("=" * 72)

    files = config["files"]
    results: dict[str, Any] = {}
    raw_metadata: dict[str, Any] = {}

    for logical_name, output_name in JSON_DATASETS.items():
        source = raw_dir / files[logical_name]
        results[logical_name] = convert_json_dataset(
            logical_name,
            source,
            output_dir / output_name,
            args.json_chunk_size,
            args.compression,
            args.overwrite,
        )
        raw_metadata[logical_name] = raw_file_metadata(source, not args.skip_hash)

    map_source = raw_dir / files["book_id_map"]
    results["book_id_map"] = convert_book_id_map(
        map_source,
        output_dir / "book_id_map.parquet",
        args.compression,
        args.overwrite,
    )
    raw_metadata["book_id_map"] = raw_file_metadata(map_source, not args.skip_hash)

    interactions_source = raw_dir / files["interactions"]
    results["interactions"] = convert_interactions(
        interactions_source,
        output_dir / "interactions",
        args.csv_chunk_size,
        args.compression,
        args.overwrite,
    )
    raw_metadata["interactions"] = raw_file_metadata(
        interactions_source, not args.skip_hash
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "raw_directory": str(raw_dir),
        "output_directory": str(output_dir),
        "settings": {
            "json_chunk_size": args.json_chunk_size,
            "csv_chunk_size": args.csv_chunk_size,
            "compression": args.compression,
            "overwrite": args.overwrite,
            "sha256_calculated": not args.skip_hash,
        },
        "raw_files": raw_metadata,
        "outputs": results,
    }

    metadata_path = output_dir / "conversion_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("CONVERSION COMPLETE")
    print(f"Metadata: {metadata_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
