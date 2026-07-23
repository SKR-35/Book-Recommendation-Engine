from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


REQUIRED_COLUMNS = {
    "Title",
    "Author",
    "Exclusive Shelf",
}

OPTIONAL_COLUMNS = {
    "Book Id",
    "Author l-f",
    "Additional Authors",
    "ISBN",
    "ISBN13",
    "My Rating",
    "Publisher",
    "Binding",
    "Number of Pages",
    "Year Published",
    "Original Publication Year",
    "Date Read",
    "Date Added",
    "Bookshelves",
    "Bookshelves with positions",
    "My Review",
    "Spoiler",
    "Private Notes",
    "Read Count",
    "Owned Copies",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a Goodreads library export, clean it, match books to the "
            "local catalog, optionally use Google Books as fallback, and write "
            "model-ready Parquet outputs."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--input", required=True, help="Goodreads export CSV.")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument(
        "--include-shelves",
        default="read,currently-reading,to-read",
        help="Comma-separated Exclusive Shelf values to import.",
    )
    parser.add_argument(
        "--profile-shelves",
        default="read,currently-reading",
        help="Shelves that contribute to the user preference profile.",
    )
    parser.add_argument(
        "--exclude-shelves",
        default="",
        help="Comma-separated custom Bookshelves values to exclude.",
    )

    parser.add_argument("--local-threshold", type=float, default=88.0)
    parser.add_argument("--google-threshold", type=float, default=90.0)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--google-max-results", type=int, default=5)
    parser.add_argument("--disable-google", action="store_true")
    parser.add_argument("--require-google-api-key", action="store_true")

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Commit matching checkpoints every N books.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the SQLite checkpoint database.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_isbn(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip()
    text = re.sub(r'^="\s*', "", text)
    text = re.sub(r'"\s*$', "", text)
    return "".join(
        character
        for character in text.upper()
        if character.isdigit() or character == "X"
    )


def parse_csv_values(value: str) -> set[str]:
    return {
        item.strip().casefold()
        for item in value.split(",")
        if item.strip()
    }


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(float(value))
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


def load_goodreads_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [column.strip() for column in frame.columns]

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Goodreads CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    return frame


def clean_library(
    frame: pd.DataFrame,
    include_shelves: set[str],
    profile_shelves: set[str],
    excluded_custom_shelves: set[str],
) -> pd.DataFrame:
    cleaned = frame.copy()

    cleaned["source_row"] = range(len(cleaned))
    cleaned["goodreads_book_id"] = cleaned["Book Id"].map(safe_int)
    cleaned["title"] = cleaned["Title"].astype(str).str.strip()
    cleaned["author"] = cleaned["Author"].astype(str).str.strip()
    cleaned["additional_authors"] = (
        cleaned["Additional Authors"].astype(str).str.strip()
    )
    cleaned["isbn"] = cleaned["ISBN"].map(normalize_isbn)
    cleaned["isbn13"] = cleaned["ISBN13"].map(normalize_isbn)
    cleaned["my_rating"] = cleaned["My Rating"].map(safe_float).fillna(0.0)
    cleaned["exclusive_shelf"] = (
        cleaned["Exclusive Shelf"].astype(str).str.strip().str.casefold()
    )
    cleaned["bookshelves"] = (
        cleaned["Bookshelves"].astype(str).str.strip()
    )
    cleaned["read_count"] = (
        cleaned["Read Count"].map(safe_int).fillna(0).astype(int)
    )
    cleaned["date_read"] = pd.to_datetime(
        cleaned["Date Read"],
        errors="coerce",
    )
    cleaned["date_added"] = pd.to_datetime(
        cleaned["Date Added"],
        errors="coerce",
    )
    cleaned["year_published"] = cleaned["Year Published"].map(safe_int)
    cleaned["original_publication_year"] = (
        cleaned["Original Publication Year"].map(safe_int)
    )

    cleaned = cleaned[
        cleaned["exclusive_shelf"].isin(include_shelves)
    ].copy()

    if excluded_custom_shelves:
        def has_excluded_shelf(value: str) -> bool:
            shelves = {
                shelf.strip().casefold()
                for shelf in str(value).split(",")
                if shelf.strip()
            }
            return bool(shelves & excluded_custom_shelves)

        cleaned = cleaned[
            ~cleaned["bookshelves"].map(has_excluded_shelf)
        ].copy()

    cleaned = cleaned[
        cleaned["title"].str.len().gt(0)
    ].copy()

    cleaned["profile_eligible"] = cleaned[
        "exclusive_shelf"
    ].isin(profile_shelves)

    cleaned["preference_weight"] = cleaned.apply(
        preference_weight,
        axis=1,
    )
    cleaned["exclusion_from_recommendations"] = (
        cleaned["exclusive_shelf"].isin(
            {"read", "currently-reading", "to-read"}
        )
    )

    cleaned["title_key"] = cleaned["title"].map(normalize_text)
    cleaned["author_key"] = cleaned["author"].map(normalize_text)

    cleaned = cleaned.sort_values(
        ["goodreads_book_id", "source_row"],
        na_position="last",
    )
    cleaned = cleaned.drop_duplicates(
        subset=["title_key", "author_key", "isbn13", "isbn"],
        keep="first",
    ).reset_index(drop=True)

    cleaned["library_row_id"] = range(len(cleaned))
    return cleaned


def preference_weight(row: pd.Series) -> float:
    rating = float(row.get("my_rating") or 0.0)
    shelf = str(row.get("exclusive_shelf") or "").casefold()
    read_count = int(row.get("read_count") or 0)

    if rating >= 5:
        weight = 1.00
    elif rating >= 4:
        weight = 0.80
    elif rating >= 3:
        weight = 0.45
    elif rating >= 2:
        weight = -0.45
    elif rating >= 1:
        weight = -1.00
    elif shelf == "currently-reading":
        weight = 0.30
    elif shelf == "read":
        weight = 0.25
    else:
        weight = 0.0

    if read_count > 1 and weight > 0:
        weight = min(weight + 0.05 * (read_count - 1), 1.25)

    return float(weight)


class CheckpointStore:
    def __init__(self, path: Path, reset: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if reset and self.path.exists():
            self.path.unlink()

        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    library_row_id INTEGER PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    completed_at_utc TEXT NOT NULL
                )
                """
            )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, library_row_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM matches
                WHERE library_row_id = ?
                """,
                (library_row_id,),
            ).fetchone()

        if row is None:
            return None
        return json.loads(row[0])

    def set(self, library_row_id: int, payload: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO matches (
                    library_row_id,
                    payload_json,
                    completed_at_utc
                )
                VALUES (?, ?, ?)
                ON CONFLICT(library_row_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    completed_at_utc = excluded.completed_at_utc
                """,
                (
                    library_row_id,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def load_matcher_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "book_input_matcher_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load matcher from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result_to_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    raise TypeError("Unsupported match result object.")


def match_one(
    matcher: Any,
    connection: duckdb.DuckDBPyConnection,
    matcher_inputs: dict[str, Path],
    processed_dir: Path,
    row: pd.Series,
    args: argparse.Namespace,
) -> dict[str, Any]:
    title = row["title"] or None
    author = row["author"] or None
    isbn = row["isbn13"] or row["isbn"] or None

    result = matcher.local_match(
        connection=connection,
        inputs=matcher_inputs,
        title=title,
        author=author,
        isbn=isbn,
        threshold=args.local_threshold,
        candidate_limit=args.candidate_limit,
    )

    notes: list[str] = []

    if result is None and not args.disable_google:
        google_books, google_notes = matcher.google_search(
            inputs=matcher_inputs,
            processed_dir=processed_dir,
            title=title,
            author=author,
            isbn=isbn,
            max_results=args.google_max_results,
            require_api_key=args.require_google_api_key,
        )
        notes.extend(google_notes)

        if google_books:
            result = matcher.map_google_to_local(
                connection=connection,
                inputs=matcher_inputs,
                google_books=google_books,
                threshold=args.google_threshold,
                candidate_limit=args.candidate_limit,
            )
            if result is None:
                result = matcher.cold_start_result(
                    google_books[0],
                    notes,
                )

    if result is None:
        result = matcher.unmatched_result(notes)

    return result_to_payload(result)


def flatten_match(
    library_row: pd.Series,
    payload: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "library_row_id": int(library_row["library_row_id"]),
        "goodreads_book_id": library_row["goodreads_book_id"],
        "input_title": library_row["title"],
        "input_author": library_row["author"],
        "input_isbn": library_row["isbn"],
        "input_isbn13": library_row["isbn13"],
        "exclusive_shelf": library_row["exclusive_shelf"],
        "bookshelves": library_row["bookshelves"],
        "my_rating": float(library_row["my_rating"]),
        "read_count": int(library_row["read_count"]),
        "date_read": library_row["date_read"],
        "date_added": library_row["date_added"],
        "profile_eligible": bool(library_row["profile_eligible"]),
        "preference_weight": float(library_row["preference_weight"]),
        "exclusion_from_recommendations": bool(
            library_row["exclusion_from_recommendations"]
        ),
    }

    flattened = {
        **base,
        "match_status": payload.get("status"),
        "match_source": payload.get("source"),
        "match_confidence": payload.get("confidence"),
        "match_method": payload.get("match_method"),
        "local_book_id": payload.get("local_book_id"),
        "work_id": payload.get("work_id"),
        "hybrid_index": payload.get("hybrid_index"),
        "content_global_row": payload.get("content_global_row"),
        "collaborative_item_index": payload.get(
            "collaborative_item_index"
        ),
        "matched_title": payload.get("title"),
        "matched_author": payload.get("primary_author"),
        "matched_isbn": payload.get("isbn"),
        "matched_isbn13": payload.get("isbn13"),
        "matched_publication_year": payload.get("publication_year"),
        "matched_primary_genre": payload.get("primary_genre"),
        "matched_image_url": payload.get("image_url"),
        "google_volume_id": payload.get("google_volume_id"),
        "google_language": payload.get("google_language"),
        "google_categories": json.dumps(
            payload.get("google_categories"),
            ensure_ascii=False,
        ),
        "cold_start_content_text": payload.get(
            "cold_start_content_text"
        ),
        "match_notes": json.dumps(
            payload.get("notes", []),
            ensure_ascii=False,
        ),
    }
    return flattened


def write_outputs(
    output_dir: Path,
    cleaned: pd.DataFrame,
    imported: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "cleaned_library": output_dir / "cleaned_library.parquet",
        "imported_library": output_dir / "imported_library.parquet",
        "matched": output_dir / "matched_books.parquet",
        "cold_start": output_dir / "cold_start_books.parquet",
        "unmatched": output_dir / "unmatched_books.parquet",
        "profile_books": output_dir / "profile_books.parquet",
        "exclusion_books": output_dir / "recommendation_exclusions.parquet",
        "report": output_dir / "import_report.json",
    }

    cleaned.to_parquet(
        outputs["cleaned_library"],
        index=False,
        compression="zstd",
    )
    imported.to_parquet(
        outputs["imported_library"],
        index=False,
        compression="zstd",
    )

    imported[
        imported["match_status"] == "matched"
    ].to_parquet(
        outputs["matched"],
        index=False,
        compression="zstd",
    )
    imported[
        imported["match_status"] == "cold_start"
    ].to_parquet(
        outputs["cold_start"],
        index=False,
        compression="zstd",
    )
    imported[
        imported["match_status"] == "unmatched"
    ].to_parquet(
        outputs["unmatched"],
        index=False,
        compression="zstd",
    )

    imported[
        imported["profile_eligible"]
        & imported["preference_weight"].ne(0)
    ].to_parquet(
        outputs["profile_books"],
        index=False,
        compression="zstd",
    )

    imported[
        imported["exclusion_from_recommendations"]
    ][
        [
            "library_row_id",
            "goodreads_book_id",
            "input_title",
            "input_author",
            "local_book_id",
            "work_id",
            "hybrid_index",
            "exclusive_shelf",
        ]
    ].to_parquet(
        outputs["exclusion_books"],
        index=False,
        compression="zstd",
    )

    return outputs


def build_report(
    input_path: Path,
    output_dir: Path,
    original_rows: int,
    cleaned: pd.DataFrame,
    imported: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    status_counts = (
        imported["match_status"]
        .value_counts(dropna=False)
        .to_dict()
    )
    method_counts = (
        imported["match_method"]
        .value_counts(dropna=False)
        .to_dict()
    )
    shelf_counts = (
        imported["exclusive_shelf"]
        .value_counts(dropna=False)
        .to_dict()
    )

    matched_count = int(
        (imported["match_status"] == "matched").sum()
    )
    total = len(imported)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned),
        "imported_rows": total,
        "matched_rows": matched_count,
        "match_rate": matched_count / total if total else 0.0,
        "cold_start_rows": int(
            (imported["match_status"] == "cold_start").sum()
        ),
        "unmatched_rows": int(
            (imported["match_status"] == "unmatched").sum()
        ),
        "profile_eligible_rows": int(
            imported["profile_eligible"].sum()
        ),
        "recommendation_exclusion_rows": int(
            imported["exclusion_from_recommendations"].sum()
        ),
        "status_counts": status_counts,
        "method_counts": method_counts,
        "shelf_counts": shelf_counts,
        "settings": {
            "include_shelves": args.include_shelves,
            "profile_shelves": args.profile_shelves,
            "exclude_shelves": args.exclude_shelves,
            "local_threshold": args.local_threshold,
            "google_threshold": args.google_threshold,
            "candidate_limit": args.candidate_limit,
            "google_max_results": args.google_max_results,
            "disable_google": args.disable_google,
            "require_google_api_key": args.require_google_api_key,
            "resume": args.resume,
        },
    }


def main() -> None:
    args = parse_args()

    if not 0 <= args.local_threshold <= 100:
        raise ValueError("--local-threshold must be between 0 and 100.")
    if not 0 <= args.google_threshold <= 100:
        raise ValueError("--google-threshold must be between 0 and 100.")
    if args.candidate_limit <= 0:
        raise ValueError("--candidate-limit must be positive.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")

    config = load_config(args.config)
    input_path = resolve_path(args.input)
    processed_dir = resolve_path(
        args.processed_dir or config["paths"]["processed_dir"]
    )
    output_dir = resolve_path(
        args.output_dir
        or (processed_dir / "user_library")
    )

    if output_dir.exists() and args.overwrite and not args.resume:
        import shutil
        shutil.rmtree(output_dir)

    if output_dir.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{output_dir} already exists. Use --overwrite or --resume."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    include_shelves = parse_csv_values(args.include_shelves)
    profile_shelves = parse_csv_values(args.profile_shelves)
    excluded_shelves = parse_csv_values(args.exclude_shelves)

    raw = load_goodreads_csv(input_path)
    cleaned = clean_library(
        raw,
        include_shelves,
        profile_shelves,
        excluded_shelves,
    )

    matcher_path = PROJECT_ROOT / "scripts" / "11_match_book_input.py"
    matcher = load_matcher_module(matcher_path)
    matcher_inputs = matcher.require_inputs(processed_dir)

    checkpoint = CheckpointStore(
        output_dir / "import_checkpoint.sqlite",
        reset=not args.resume,
    )

    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("SET memory_limit = '2GB'")
    connection.execute("SET preserve_insertion_order = false")

    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        for _, row in tqdm(
            cleaned.iterrows(),
            total=len(cleaned),
            desc="Matching Goodreads books",
            unit="book",
        ):
            library_row_id = int(row["library_row_id"])
            payload = (
                checkpoint.get(library_row_id)
                if args.resume
                else None
            )

            if payload is None:
                payload = match_one(
                    matcher=matcher,
                    connection=connection,
                    matcher_inputs=matcher_inputs,
                    processed_dir=processed_dir,
                    row=row,
                    args=args,
                )
                checkpoint.set(library_row_id, payload)

            results.append(flatten_match(row, payload))

    finally:
        connection.close()

    imported = pd.DataFrame(results)
    outputs = write_outputs(output_dir, cleaned, imported)

    report = build_report(
        input_path=input_path,
        output_dir=output_dir,
        original_rows=len(raw),
        cleaned=cleaned,
        imported=imported,
        args=args,
    )
    report["elapsed_seconds"] = round(
        time.perf_counter() - started,
        3,
    )

    outputs["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("GOODREADS LIBRARY IMPORT COMPLETE")
    print(f"Original rows      : {report['original_rows']:,}")
    print(f"Imported rows      : {report['imported_rows']:,}")
    print(f"Matched locally    : {report['matched_rows']:,}")
    print(f"Cold-start books   : {report['cold_start_rows']:,}")
    print(f"Unmatched books    : {report['unmatched_rows']:,}")
    print(f"Match rate         : {report['match_rate']:.2%}")
    print(f"Profile books      : {report['profile_eligible_rows']:,}")
    print(
        f"Recommendation exclusions: "
        f"{report['recommendation_exclusion_rows']:,}"
    )
    print(f"Output directory   : {output_dir}")
    print(f"Report             : {outputs['report']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
