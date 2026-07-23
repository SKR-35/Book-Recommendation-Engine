from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


@dataclass(frozen=True)
class MatchResult:
    status: str
    source: str
    confidence: float
    match_method: str
    local_book_id: int | None
    work_id: int | None
    hybrid_index: int | None
    content_global_row: int | None
    collaborative_item_index: int | None
    title: str | None
    primary_author: str | None
    isbn: str | None
    isbn13: str | None
    publication_year: int | None
    primary_genre: str | None
    description: str | None
    image_url: str | None
    google_volume_id: str | None
    google_categories: list[str] | None
    google_language: str | None
    cold_start_content_text: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match a user-supplied book to the local Goodreads catalog, "
            "with optional Google Books fallback."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--isbn", default=None)
    parser.add_argument("--local-threshold", type=float, default=88.0)
    parser.add_argument("--google-threshold", type=float, default=90.0)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--google-max-results", type=int, default=5)
    parser.add_argument("--disable-google", action="store_true")
    parser.add_argument("--require-google-api-key", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character for character in value if not unicodedata.combining(character)
    )
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_isbn(value: str | None) -> str:
    if not value:
        return ""
    return "".join(
        c for c in value.upper() if c.isdigit() or c == "X"
    )


def parse_year(value: Any) -> int | None:
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


def validate_args(args: argparse.Namespace) -> None:
    if not any([args.title, args.author, args.isbn]):
        raise ValueError("Supply at least one of --title, --author, or --isbn.")
    if not 0 <= args.local_threshold <= 100:
        raise ValueError("--local-threshold must be between 0 and 100.")
    if not 0 <= args.google_threshold <= 100:
        raise ValueError("--google-threshold must be between 0 and 100.")
    if args.candidate_limit <= 0:
        raise ValueError("--candidate-limit must be positive.")
    if not 1 <= args.google_max_results <= 40:
        raise ValueError("--google-max-results must be between 1 and 40.")


def require_inputs(processed_dir: Path) -> dict[str, Path]:
    inputs = {
        "catalog": processed_dir / "catalog",
        "hybrid_map": processed_dir / "hybrid" / "hybrid_item_map.parquet",
        "google_client": PROJECT_ROOT / "scripts" / "10_google_books_client.py",
    }
    missing = []
    if not inputs["catalog"].exists():
        missing.append(str(inputs["catalog"]))
    if not inputs["hybrid_map"].exists():
        missing.append(str(inputs["hybrid_map"]))
    if missing:
        raise FileNotFoundError(
            "Required matching inputs are missing:\n  - " + "\n  - ".join(missing)
        )
    return inputs


def load_google_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("google_books_client_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Google Books client from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def isbn_local_match(
    connection: duckdb.DuckDBPyConnection,
    catalog_glob: str,
    isbn: str,
) -> pd.DataFrame:
    normalized = normalize_isbn(isbn)
    if not normalized:
        return pd.DataFrame()

    return connection.execute(
        f"""
        SELECT
            book_id, work_id, title, primary_author, isbn, isbn13,
            publication_year, primary_genre, description, image_url,
            ratings_count, bayesian_rating
        FROM read_parquet('{catalog_glob}')
        WHERE REGEXP_REPLACE(UPPER(COALESCE(isbn, '')), '[^0-9X]', '', 'g') = ?
           OR REGEXP_REPLACE(UPPER(COALESCE(isbn13, '')), '[^0-9X]', '', 'g') = ?
        ORDER BY
            is_canonical_edition DESC,
            ratings_count DESC NULLS LAST,
            book_id
        LIMIT 10
        """,
        [normalized, normalized],
    ).fetchdf()


def title_candidates(
    connection: duckdb.DuckDBPyConnection,
    catalog_glob: str,
    title: str,
    author: str | None,
    limit: int,
) -> pd.DataFrame:
    title_tokens = [
        token for token in normalize_text(title).split() if len(token) >= 3
    ][:4]
    author_tokens = [
        token for token in normalize_text(author).split() if len(token) >= 3
    ][:2]

    where_parts = ["is_canonical_edition = TRUE"]
    parameters: list[Any] = []

    if title_tokens:
        parts = []
        for token in title_tokens:
            parts.append("LOWER(COALESCE(title, '')) LIKE ?")
            parameters.append(f"%{token}%")
        where_parts.append("(" + " OR ".join(parts) + ")")

    if author_tokens:
        parts = []
        for token in author_tokens:
            parts.append("LOWER(COALESCE(primary_author, '')) LIKE ?")
            parameters.append(f"%{token}%")
        where_parts.append("(" + " OR ".join(parts) + ")")

    query = f"""
    SELECT
        book_id, work_id, title, primary_author, isbn, isbn13,
        publication_year, primary_genre, description, image_url,
        ratings_count, bayesian_rating
    FROM read_parquet('{catalog_glob}')
    WHERE {' AND '.join(where_parts)}
    ORDER BY
        ratings_count DESC NULLS LAST,
        bayesian_rating DESC NULLS LAST,
        book_id
    LIMIT {limit}
    """
    return connection.execute(query, parameters).fetchdf()


def score_candidate(
    query_title: str,
    query_author: str | None,
    candidate_title: str | None,
    candidate_author: str | None,
) -> tuple[float, dict[str, float]]:
    q_title = normalize_text(query_title)
    c_title = normalize_text(candidate_title)
    title_score = max(
        float(fuzz.WRatio(q_title, c_title)),
        float(fuzz.token_set_ratio(q_title, c_title)),
    )

    q_author = normalize_text(query_author)
    c_author = normalize_text(candidate_author)
    if q_author:
        author_score = float(fuzz.token_set_ratio(q_author, c_author))
        final = 0.78 * title_score + 0.22 * author_score
    else:
        author_score = 0.0
        final = title_score

    return final, {
        "title_score": title_score,
        "author_score": author_score,
    }


def best_fuzzy_match(
    candidates: pd.DataFrame,
    title: str,
    author: str | None,
) -> tuple[pd.Series | None, float, dict[str, float]]:
    best_row = None
    best_score = -1.0
    best_components: dict[str, float] = {}

    for _, row in candidates.iterrows():
        score, components = score_candidate(
            title,
            author,
            row.get("title"),
            row.get("primary_author"),
        )
        if score > best_score:
            best_row = row
            best_score = score
            best_components = components

    return best_row, best_score, best_components


def hybrid_lookup(
    connection: duckdb.DuckDBPyConnection,
    hybrid_map_path: Path,
    book_id: int,
) -> dict[str, int | None]:
    row = connection.execute(
        f"""
        SELECT hybrid_index, content_global_row, collaborative_item_index
        FROM read_parquet('{sql_path(hybrid_map_path)}')
        WHERE TRY_CAST(book_id AS BIGINT) = ?
        LIMIT 1
        """,
        [book_id],
    ).fetchone()

    if row is None:
        return {
            "hybrid_index": None,
            "content_global_row": None,
            "collaborative_item_index": None,
        }

    return {
        "hybrid_index": int(row[0]),
        "content_global_row": int(row[1]),
        "collaborative_item_index": int(row[2]),
    }


def local_result_from_row(
    connection: duckdb.DuckDBPyConnection,
    hybrid_map_path: Path,
    row: pd.Series,
    confidence: float,
    method: str,
    notes: list[str],
) -> MatchResult:
    book_id = int(row["book_id"])
    indices = hybrid_lookup(connection, hybrid_map_path, book_id)

    return MatchResult(
        status="matched",
        source="local_catalog",
        confidence=round(confidence, 4),
        match_method=method,
        local_book_id=book_id,
        work_id=int(row["work_id"]) if pd.notna(row.get("work_id")) else None,
        hybrid_index=indices["hybrid_index"],
        content_global_row=indices["content_global_row"],
        collaborative_item_index=indices["collaborative_item_index"],
        title=row.get("title"),
        primary_author=row.get("primary_author"),
        isbn=row.get("isbn"),
        isbn13=row.get("isbn13"),
        publication_year=parse_year(row.get("publication_year")),
        primary_genre=row.get("primary_genre"),
        description=row.get("description"),
        image_url=row.get("image_url"),
        google_volume_id=None,
        google_categories=None,
        google_language=None,
        cold_start_content_text=None,
        notes=notes,
    )


def local_match(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    title: str | None,
    author: str | None,
    isbn: str | None,
    threshold: float,
    candidate_limit: int,
) -> MatchResult | None:
    catalog_glob = sql_path(inputs["catalog"] / "*.parquet")

    if isbn:
        candidates = isbn_local_match(connection, catalog_glob, isbn)
        if not candidates.empty:
            return local_result_from_row(
                connection,
                inputs["hybrid_map"],
                candidates.iloc[0],
                100.0,
                "isbn_exact",
                ["Exact ISBN match in local catalog."],
            )

    if title:
        candidates = title_candidates(
            connection,
            catalog_glob,
            title,
            author,
            candidate_limit,
        )
        row, score, components = best_fuzzy_match(candidates, title, author)
        if row is not None and score >= threshold:
            return local_result_from_row(
                connection,
                inputs["hybrid_map"],
                row,
                score,
                "title_author_fuzzy",
                [
                    f"Fuzzy title score: {components.get('title_score', 0):.2f}",
                    f"Fuzzy author score: {components.get('author_score', 0):.2f}",
                ],
            )
    return None


def google_content_text(book: Any) -> str:
    parts = [
        book.title or "",
        book.title or "",
        book.title or "",
        " ".join(book.authors),
        " ".join(book.authors),
        " | ".join(book.categories),
        " | ".join(book.categories),
        book.description or "",
        f"language_{book.language}" if book.language else "",
    ]
    return " ".join(part for part in parts if part).strip()


def google_search(
    inputs: dict[str, Path],
    processed_dir: Path,
    title: str | None,
    author: str | None,
    isbn: str | None,
    max_results: int,
    require_api_key: bool,
) -> tuple[list[Any], list[str]]:
    notes: list[str] = []

    if require_api_key and not os.getenv("GOOGLE_BOOKS_API_KEY"):
        notes.append(
            "Google Books skipped because GOOGLE_BOOKS_API_KEY is not configured."
        )
        return [], notes

    try:
        module = load_google_module(inputs["google_client"])
        client = module.GoogleBooksClient(
            api_key=os.getenv("GOOGLE_BOOKS_API_KEY"),
            cache_path=(
                processed_dir / "google_books" / "google_books_cache.sqlite"
            ),
        )
        response = client.search_book(
            title=title,
            author=author,
            isbn=isbn,
            max_results=max_results,
            use_cache=True,
        )
        notes.append(f"Google Books returned {len(response.books)} result(s).")
        if response.from_cache:
            notes.append("Google Books response came from SQLite cache.")
        return response.books, notes
    except Exception as exc:
        notes.append(
            f"Google Books unavailable; local-only mode retained: "
            f"{type(exc).__name__}: {exc}"
        )
        return [], notes


def map_google_to_local(
    connection: duckdb.DuckDBPyConnection,
    inputs: dict[str, Path],
    google_books: list[Any],
    threshold: float,
    candidate_limit: int,
) -> MatchResult | None:
    catalog_glob = sql_path(inputs["catalog"] / "*.parquet")

    for book in google_books:
        google_isbn = book.isbn13 or book.isbn10
        if google_isbn:
            candidates = isbn_local_match(connection, catalog_glob, google_isbn)
            if not candidates.empty:
                return local_result_from_row(
                    connection,
                    inputs["hybrid_map"],
                    candidates.iloc[0],
                    100.0,
                    "google_isbn_to_local_exact",
                    [
                        "Input was resolved with Google Books.",
                        "Google ISBN matched a local catalog edition.",
                    ],
                )

        if book.title:
            google_author = book.authors[0] if book.authors else None
            candidates = title_candidates(
                connection,
                catalog_glob,
                book.title,
                google_author,
                candidate_limit,
            )
            row, score, components = best_fuzzy_match(
                candidates,
                book.title,
                google_author,
            )
            if row is not None and score >= threshold:
                return local_result_from_row(
                    connection,
                    inputs["hybrid_map"],
                    row,
                    score,
                    "google_title_author_to_local_fuzzy",
                    [
                        "Input was resolved with Google Books.",
                        f"Google-to-local title score: "
                        f"{components.get('title_score', 0):.2f}",
                        f"Google-to-local author score: "
                        f"{components.get('author_score', 0):.2f}",
                    ],
                )
    return None


def cold_start_result(book: Any, notes: list[str]) -> MatchResult:
    return MatchResult(
        status="cold_start",
        source="google_books",
        confidence=100.0,
        match_method="google_metadata_only",
        local_book_id=None,
        work_id=None,
        hybrid_index=None,
        content_global_row=None,
        collaborative_item_index=None,
        title=book.title,
        primary_author=", ".join(book.authors) if book.authors else None,
        isbn=book.isbn10,
        isbn13=book.isbn13,
        publication_year=parse_year(
            str(book.published_date)[:4] if book.published_date else None
        ),
        primary_genre=book.categories[0] if book.categories else None,
        description=book.description,
        image_url=book.thumbnail_url,
        google_volume_id=book.google_volume_id,
        google_categories=list(book.categories),
        google_language=book.language,
        cold_start_content_text=google_content_text(book),
        notes=notes
        + [
            "No reliable local catalog match was found.",
            "Use content-only retrieval, then popularity/novelty reranking.",
        ],
    )


def unmatched_result(notes: list[str]) -> MatchResult:
    return MatchResult(
        status="unmatched",
        source="none",
        confidence=0.0,
        match_method="none",
        local_book_id=None,
        work_id=None,
        hybrid_index=None,
        content_global_row=None,
        collaborative_item_index=None,
        title=None,
        primary_author=None,
        isbn=None,
        isbn13=None,
        publication_year=None,
        primary_genre=None,
        description=None,
        image_url=None,
        google_volume_id=None,
        google_categories=None,
        google_language=None,
        cold_start_content_text=None,
        notes=notes + ["No reliable local or Google Books result was available."],
    )


def print_result(result: MatchResult) -> None:
    print("=" * 76)
    print("BOOK INPUT MATCH")
    print(f"Status              : {result.status}")
    print(f"Source              : {result.source}")
    print(f"Method              : {result.match_method}")
    print(f"Confidence          : {result.confidence:.2f}")
    print(f"Title               : {result.title or 'Unknown'}")
    print(f"Author              : {result.primary_author or 'Unknown'}")
    print(f"Local book ID       : {result.local_book_id}")
    print(f"Hybrid index        : {result.hybrid_index}")
    print(f"Content row         : {result.content_global_row}")
    print(f"Collaborative index : {result.collaborative_item_index}")
    print(f"Google volume ID    : {result.google_volume_id}")
    print("=" * 76)

    if result.notes:
        print("Notes:")
        for note in result.notes:
            print(f"- {note}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    config = load_config(args.config)
    processed_dir = resolve_path(
        args.processed_dir or config["paths"]["processed_dir"]
    )
    inputs = require_inputs(processed_dir)

    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("SET memory_limit = '2GB'")
    connection.execute("SET preserve_insertion_order = false")

    notes: list[str] = []

    try:
        result = local_match(
            connection=connection,
            inputs=inputs,
            title=args.title,
            author=args.author,
            isbn=args.isbn,
            threshold=args.local_threshold,
            candidate_limit=args.candidate_limit,
        )

        if result is None and not args.disable_google:
            google_books, google_notes = google_search(
                inputs=inputs,
                processed_dir=processed_dir,
                title=args.title,
                author=args.author,
                isbn=args.isbn,
                max_results=args.google_max_results,
                require_api_key=args.require_google_api_key,
            )
            notes.extend(google_notes)

            if google_books:
                result = map_google_to_local(
                    connection=connection,
                    inputs=inputs,
                    google_books=google_books,
                    threshold=args.google_threshold,
                    candidate_limit=args.candidate_limit,
                )
                if result is None:
                    result = cold_start_result(google_books[0], notes)

        if result is None:
            result = unmatched_result(notes)
    finally:
        connection.close()

    print_result(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
