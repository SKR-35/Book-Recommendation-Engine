from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book_recommender.config import load_config  # noqa: E402


GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1"
DEFAULT_USER_AGENT = "book-recommendation-engine/0.1"


@dataclass(frozen=True)
class GoogleBook:
    google_volume_id: str
    title: str | None
    subtitle: str | None
    authors: list[str]
    publisher: str | None
    published_date: str | None
    description: str | None
    isbn10: str | None
    isbn13: str | None
    other_identifiers: dict[str, str]
    categories: list[str]
    language: str | None
    page_count: int | None
    average_rating: float | None
    ratings_count: int | None
    maturity_rating: str | None
    print_type: str | None
    canonical_volume_link: str | None
    info_link: str | None
    preview_link: str | None
    thumbnail_url: str | None
    small_thumbnail_url: str | None
    text_reading_modes: bool | None
    image_reading_modes: bool | None
    saleability: str | None
    is_ebook: bool | None
    country: str | None
    raw_query: str | None = None
    fetched_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResponse:
    query: str
    total_items: int
    books: list[GoogleBook]
    from_cache: bool
    cache_key: str


class GoogleBooksError(RuntimeError):
    """Base error for Google Books client failures."""


class GoogleBooksHTTPError(GoogleBooksError):
    """Raised when Google Books returns a non-success HTTP response."""


class GoogleBooksRateLimitError(GoogleBooksHTTPError):
    """Raised when Google Books returns HTTP 429."""


class SQLiteCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS google_books_cache (
                    cache_key TEXT PRIMARY KEY,
                    request_type TEXT NOT NULL,
                    request_value TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    fetched_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_google_books_expiry
                ON google_books_cache (expires_at_utc)
                """
            )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_json
                FROM google_books_cache
                WHERE cache_key = ?
                  AND expires_at_utc > ?
                """,
                (cache_key, now),
            ).fetchone()

        if row is None:
            return None

        return json.loads(row["response_json"])

    def set(
        self,
        cache_key: str,
        request_type: str,
        request_value: str,
        response: dict[str, Any],
        status_code: int,
        ttl_days: int,
    ) -> None:
        fetched_at = datetime.now(timezone.utc)
        expires_at = fetched_at + timedelta(days=ttl_days)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO google_books_cache (
                    cache_key,
                    request_type,
                    request_value,
                    response_json,
                    status_code,
                    fetched_at_utc,
                    expires_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    request_type = excluded.request_type,
                    request_value = excluded.request_value,
                    response_json = excluded.response_json,
                    status_code = excluded.status_code,
                    fetched_at_utc = excluded.fetched_at_utc,
                    expires_at_utc = excluded.expires_at_utc
                """,
                (
                    cache_key,
                    request_type,
                    request_value,
                    json.dumps(response, ensure_ascii=False),
                    status_code,
                    fetched_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )

    def delete_expired(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM google_books_cache
                WHERE expires_at_utc <= ?
                """,
                (now,),
            )
            return int(cursor.rowcount)

    def clear(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM google_books_cache")
            return int(cursor.rowcount)


class GoogleBooksClient:
    """
    Thin Google Books API client with optional API key and SQLite caching.

    Public volume search and retrieval can be used without OAuth. An API key is
    optional and included only when supplied explicitly or through the
    GOOGLE_BOOKS_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        cache_ttl_days: int = 30,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_BOOKS_API_KEY")
        self.cache_ttl_days = cache_ttl_days
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        default_cache = (
            PROJECT_ROOT
            / "data"
            / "cache"
            / "google_books_cache.sqlite"
        )
        self.cache = SQLiteCache(cache_path or default_cache)

    @staticmethod
    def build_query(
        title: str | None = None,
        author: str | None = None,
        isbn: str | None = None,
        subject: str | None = None,
        publisher: str | None = None,
        free_text: str | None = None,
    ) -> str:
        parts: list[str] = []

        if free_text and free_text.strip():
            parts.append(free_text.strip())
        if title and title.strip():
            parts.append(f'intitle:"{title.strip()}"')
        if author and author.strip():
            parts.append(f'inauthor:"{author.strip()}"')
        if isbn and isbn.strip():
            normalized_isbn = "".join(
                character
                for character in isbn
                if character.isdigit() or character.upper() == "X"
            )
            parts.append(f"isbn:{normalized_isbn}")
        if subject and subject.strip():
            parts.append(f'insubject:"{subject.strip()}"')
        if publisher and publisher.strip():
            parts.append(f'inpublisher:"{publisher.strip()}"')

        query = " ".join(parts).strip()
        if not query:
            raise ValueError(
                "At least one of title, author, isbn, subject, publisher, "
                "or free_text must be supplied."
            )

        return query

    @staticmethod
    def _cache_key(request_type: str, value: str, params: dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "request_type": request_type,
                "value": value,
                "params": params,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _request(
        self,
        path: str,
        params: dict[str, Any],
        request_type: str,
        request_value: str,
        use_cache: bool,
    ) -> tuple[dict[str, Any], bool, str]:
        request_params = {
            key: value
            for key, value in params.items()
            if value is not None
        }

        if self.api_key:
            request_params["key"] = self.api_key

        cache_params = {
            key: value
            for key, value in request_params.items()
            if key != "key"
        }
        cache_key = self._cache_key(
            request_type=request_type,
            value=request_value,
            params=cache_params,
        )

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached, True, cache_key

        url = f"{GOOGLE_BOOKS_BASE_URL}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=request_params,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise GoogleBooksError(
                        f"Google Books request failed after "
                        f"{attempt + 1} attempts: {exc}"
                    ) from exc

                time.sleep(
                    self.retry_backoff_seconds * (2**attempt)
                )
                continue

            if response.status_code == 429:
                if attempt >= self.max_retries:
                    raise GoogleBooksRateLimitError(
                        "Google Books rate limit exceeded after retries."
                    )

                retry_after = response.headers.get("Retry-After")
                wait_seconds = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else self.retry_backoff_seconds * (2**attempt)
                )
                time.sleep(wait_seconds)
                continue

            if 500 <= response.status_code < 600:
                if attempt >= self.max_retries:
                    raise GoogleBooksHTTPError(
                        f"Google Books server error "
                        f"{response.status_code}: {response.text[:300]}"
                    )

                time.sleep(
                    self.retry_backoff_seconds * (2**attempt)
                )
                continue

            if not response.ok:
                raise GoogleBooksHTTPError(
                    f"Google Books returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            payload = response.json()

            if use_cache:
                self.cache.set(
                    cache_key=cache_key,
                    request_type=request_type,
                    request_value=request_value,
                    response=payload,
                    status_code=response.status_code,
                    ttl_days=self.cache_ttl_days,
                )

            return payload, False, cache_key

        raise GoogleBooksError(
            f"Google Books request failed: {last_error}"
        )

    @staticmethod
    def _extract_identifiers(
        volume_info: dict[str, Any],
    ) -> tuple[str | None, str | None, dict[str, str]]:
        isbn10: str | None = None
        isbn13: str | None = None
        other: dict[str, str] = {}

        for identifier in volume_info.get(
            "industryIdentifiers", []
        ):
            identifier_type = str(
                identifier.get("type", "")
            ).upper()
            value = str(identifier.get("identifier", "")).strip()

            if not value:
                continue
            if identifier_type == "ISBN_10":
                isbn10 = value
            elif identifier_type == "ISBN_13":
                isbn13 = value
            else:
                other[identifier_type] = value

        return isbn10, isbn13, other

    @staticmethod
    def _preferred_image(
        image_links: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        thumbnail = (
            image_links.get("extraLarge")
            or image_links.get("large")
            or image_links.get("medium")
            or image_links.get("small")
            or image_links.get("thumbnail")
        )
        small_thumbnail = image_links.get("smallThumbnail")

        def secure(url: str | None) -> str | None:
            if not url:
                return None
            return str(url).replace("http://", "https://", 1)

        return secure(thumbnail), secure(small_thumbnail)

    @classmethod
    def parse_volume(
        cls,
        item: dict[str, Any],
        raw_query: str | None = None,
    ) -> GoogleBook:
        volume_info = item.get("volumeInfo", {})
        sale_info = item.get("saleInfo", {})
        reading_modes = volume_info.get("readingModes", {})
        image_links = volume_info.get("imageLinks", {})

        isbn10, isbn13, other_identifiers = cls._extract_identifiers(
            volume_info
        )
        thumbnail, small_thumbnail = cls._preferred_image(
            image_links
        )

        page_count = volume_info.get("pageCount")
        ratings_count = volume_info.get("ratingsCount")
        average_rating = volume_info.get("averageRating")

        return GoogleBook(
            google_volume_id=str(item.get("id", "")),
            title=volume_info.get("title"),
            subtitle=volume_info.get("subtitle"),
            authors=[
                str(author)
                for author in volume_info.get("authors", [])
            ],
            publisher=volume_info.get("publisher"),
            published_date=volume_info.get("publishedDate"),
            description=volume_info.get("description"),
            isbn10=isbn10,
            isbn13=isbn13,
            other_identifiers=other_identifiers,
            categories=[
                str(category)
                for category in volume_info.get("categories", [])
            ],
            language=volume_info.get("language"),
            page_count=(
                int(page_count)
                if isinstance(page_count, (int, float))
                else None
            ),
            average_rating=(
                float(average_rating)
                if isinstance(average_rating, (int, float))
                else None
            ),
            ratings_count=(
                int(ratings_count)
                if isinstance(ratings_count, (int, float))
                else None
            ),
            maturity_rating=volume_info.get("maturityRating"),
            print_type=volume_info.get("printType"),
            canonical_volume_link=volume_info.get(
                "canonicalVolumeLink"
            ),
            info_link=volume_info.get("infoLink"),
            preview_link=volume_info.get("previewLink"),
            thumbnail_url=thumbnail,
            small_thumbnail_url=small_thumbnail,
            text_reading_modes=reading_modes.get("text"),
            image_reading_modes=reading_modes.get("image"),
            saleability=sale_info.get("saleability"),
            is_ebook=sale_info.get("isEbook"),
            country=sale_info.get("country"),
            raw_query=raw_query,
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def search(
        self,
        query: str,
        max_results: int = 10,
        start_index: int = 0,
        language: str | None = None,
        order_by: str = "relevance",
        print_type: str = "books",
        filter_name: str | None = None,
        projection: str = "full",
        use_cache: bool = True,
    ) -> SearchResponse:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if not 1 <= max_results <= 40:
            raise ValueError("max_results must be between 1 and 40.")
        if start_index < 0:
            raise ValueError("start_index cannot be negative.")
        if order_by not in {"relevance", "newest"}:
            raise ValueError(
                "order_by must be 'relevance' or 'newest'."
            )
        if print_type not in {"all", "books", "magazines"}:
            raise ValueError(
                "print_type must be 'all', 'books', or 'magazines'."
            )
        if projection not in {"full", "lite"}:
            raise ValueError("projection must be 'full' or 'lite'.")

        params = {
            "q": query,
            "maxResults": max_results,
            "startIndex": start_index,
            "langRestrict": language,
            "orderBy": order_by,
            "printType": print_type,
            "filter": filter_name,
            "projection": projection,
        }

        payload, from_cache, cache_key = self._request(
            path="volumes",
            params=params,
            request_type="search",
            request_value=query,
            use_cache=use_cache,
        )

        books = [
            self.parse_volume(item, raw_query=query)
            for item in payload.get("items", [])
        ]

        return SearchResponse(
            query=query,
            total_items=int(payload.get("totalItems", 0)),
            books=books,
            from_cache=from_cache,
            cache_key=cache_key,
        )

    def search_book(
        self,
        title: str | None = None,
        author: str | None = None,
        isbn: str | None = None,
        subject: str | None = None,
        publisher: str | None = None,
        free_text: str | None = None,
        max_results: int = 10,
        language: str | None = None,
        order_by: str = "relevance",
        use_cache: bool = True,
    ) -> SearchResponse:
        query = self.build_query(
            title=title,
            author=author,
            isbn=isbn,
            subject=subject,
            publisher=publisher,
            free_text=free_text,
        )
        return self.search(
            query=query,
            max_results=max_results,
            language=language,
            order_by=order_by,
            use_cache=use_cache,
        )

    def get_volume(
        self,
        volume_id: str,
        projection: str = "full",
        use_cache: bool = True,
    ) -> GoogleBook:
        if not volume_id.strip():
            raise ValueError("volume_id cannot be empty.")
        if projection not in {"full", "lite"}:
            raise ValueError("projection must be 'full' or 'lite'.")

        payload, _, _ = self._request(
            path=f"volumes/{volume_id.strip()}",
            params={"projection": projection},
            request_type="get",
            request_value=volume_id.strip(),
            use_cache=use_cache,
        )
        return self.parse_volume(payload)

    def search_isbn(
        self,
        isbn: str,
        max_results: int = 5,
        use_cache: bool = True,
    ) -> SearchResponse:
        return self.search_book(
            isbn=isbn,
            max_results=max_results,
            use_cache=use_cache,
        )

    def clear_cache(self) -> int:
        return self.cache.clear()

    def delete_expired_cache(self) -> int:
        return self.cache.delete_expired()


def books_to_dataframe(books: Iterable[GoogleBook]) -> Any:
    import pandas as pd

    rows = []
    for book in books:
        row = book.to_dict()
        row["authors"] = " | ".join(book.authors)
        row["categories"] = " | ".join(book.categories)
        row["other_identifiers"] = json.dumps(
            book.other_identifiers,
            ensure_ascii=False,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def resolve_cache_path(
    config_path: str,
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    try:
        config = load_config(config_path)
        processed_dir = Path(
            config["paths"]["processed_dir"]
        ).expanduser()
        if not processed_dir.is_absolute():
            processed_dir = PROJECT_ROOT / processed_dir
        return (
            processed_dir.resolve()
            / "google_books"
            / "google_books_cache.sqlite"
        )
    except (FileNotFoundError, KeyError, TypeError):
        return (
            PROJECT_ROOT
            / "data"
            / "cache"
            / "google_books_cache.sqlite"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search Google Books with optional API key and SQLite cache."
        )
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--cache-ttl-days", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=3)

    search_group = parser.add_argument_group("search")
    search_group.add_argument("--query", default=None)
    search_group.add_argument("--title", default=None)
    search_group.add_argument("--author", default=None)
    search_group.add_argument("--isbn", default=None)
    search_group.add_argument("--subject", default=None)
    search_group.add_argument("--publisher", default=None)
    search_group.add_argument("--language", default=None)
    search_group.add_argument("--max-results", type=int, default=10)
    search_group.add_argument(
        "--order-by",
        choices=["relevance", "newest"],
        default="relevance",
    )
    search_group.add_argument("--no-cache", action="store_true")
    search_group.add_argument("--output-json", default=None)
    search_group.add_argument("--output-csv", default=None)

    maintenance_group = parser.add_argument_group("cache maintenance")
    maintenance_group.add_argument("--clear-cache", action="store_true")
    maintenance_group.add_argument(
        "--delete-expired-cache",
        action="store_true",
    )

    return parser.parse_args()


def print_results(response: SearchResponse) -> None:
    print("=" * 76)
    print("GOOGLE BOOKS SEARCH")
    print(f"Query       : {response.query}")
    print(f"Total items : {response.total_items:,}")
    print(f"Returned    : {len(response.books):,}")
    print(f"From cache  : {response.from_cache}")
    print("=" * 76)

    if not response.books:
        print("No results found.")
        return

    for rank, book in enumerate(response.books, start=1):
        authors = ", ".join(book.authors) or "Unknown author"
        identifier = book.isbn13 or book.isbn10 or "No ISBN"
        print(
            f"{rank:>2}. {book.title or 'Untitled'}\n"
            f"    Authors : {authors}\n"
            f"    Date    : {book.published_date or 'Unknown'}\n"
            f"    ISBN    : {identifier}\n"
            f"    Volume  : {book.google_volume_id}\n"
        )


def main() -> None:
    args = parse_args()

    if args.cache_ttl_days <= 0:
        raise ValueError("--cache-ttl-days must be positive.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative.")

    cache_path = resolve_cache_path(
        config_path=args.config,
        explicit_path=args.cache_path,
    )

    client = GoogleBooksClient(
        api_key=args.api_key,
        cache_path=cache_path,
        cache_ttl_days=args.cache_ttl_days,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )

    if args.clear_cache:
        deleted = client.clear_cache()
        print(f"Cleared {deleted:,} cached responses.")
        return

    if args.delete_expired_cache:
        deleted = client.delete_expired_cache()
        print(f"Deleted {deleted:,} expired cached responses.")
        return

    query_supplied = any(
        [
            args.query,
            args.title,
            args.author,
            args.isbn,
            args.subject,
            args.publisher,
        ]
    )
    if not query_supplied:
        raise ValueError(
            "Supply --query, --title, --author, --isbn, --subject, "
            "or --publisher."
        )

    if args.query:
        response = client.search(
            query=args.query,
            max_results=args.max_results,
            language=args.language,
            order_by=args.order_by,
            use_cache=not args.no_cache,
        )
    else:
        response = client.search_book(
            title=args.title,
            author=args.author,
            isbn=args.isbn,
            subject=args.subject,
            publisher=args.publisher,
            max_results=args.max_results,
            language=args.language,
            order_by=args.order_by,
            use_cache=not args.no_cache,
        )

    print_results(response)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "query": response.query,
                    "total_items": response.total_items,
                    "from_cache": response.from_cache,
                    "books": [
                        book.to_dict()
                        for book in response.books
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"JSON saved to: {output_path.resolve()}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        books_to_dataframe(response.books).to_csv(
            output_path,
            index=False,
        )
        print(f"CSV saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
