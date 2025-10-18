#!/usr/bin/env python3


import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# PostgreSQL imports
import psycopg
from psycopg.rows import dict_row
try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None


API_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "speedlimit-display-name-updater/1.0 (admin@example.com)"

# Lock for thread-safe logging and rate limiting
api_call_lock = Lock()
last_api_call_time = 0.0

RowData = Tuple[int, Optional[float], Optional[float]]
UpdateData = Tuple[int, str]

# Default database URL - can be overridden by SPEEDLIMIT_DATABASE_URL environment variable
# For GitHub Actions, set DATABASE_URL secret and it will be used automatically
DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

# Database connection pool (for multi-threaded mode)
_POOL: Optional["ConnectionPool"] = None
_POOL_LOCK: Lock = Lock()


# ============================================================================
# DATABASE CONNECTION FUNCTIONS (copied from db_config.py)
# ============================================================================

def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable value or the default if invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _build_connect_kwargs():
    """Construct keyword arguments for psycopg.connect."""
    database_url = os.getenv("SPEEDLIMIT_DATABASE_URL")
    if database_url:
        return {"conninfo": database_url}
    else:
        # Fallback: use DEFAULT_DATABASE_URL if SPEEDLIMIT_DATABASE_URL is not set
        return {"conninfo": DEFAULT_DATABASE_URL}


def connect() -> psycopg.Connection:
    """Return a psycopg connection to the configured PostgreSQL database."""
    return psycopg.connect(**_build_connect_kwargs())


def get_pool() -> "ConnectionPool":
    """Initialise (once) and return a psycopg pooled connection object."""
    if ConnectionPool is None:
        raise RuntimeError(
            "psycopg_pool is required for connection pooling. "
            "Install the 'psycopg-pool' package or set SPEEDLIMIT_DISABLE_POOLING=1."
        )

    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                min_size = max(1, _env_int("SPEEDLIMIT_POOL_MIN_SIZE", 1))
                max_size = max(min_size, _env_int("SPEEDLIMIT_POOL_MAX_SIZE", 5))
                connect_kwargs = _build_connect_kwargs()
                conninfo = connect_kwargs.pop("conninfo", None)
                pool_kwargs = {"min_size": min_size, "max_size": max_size}

                if conninfo:
                    if connect_kwargs:
                        pool_kwargs["kwargs"] = connect_kwargs
                    _POOL = ConnectionPool(conninfo, **pool_kwargs)
                else:
                    pool_kwargs["kwargs"] = connect_kwargs or None
                    _POOL = ConnectionPool(**pool_kwargs)
    return _POOL


@contextmanager
def connection_scope(*, dict_rows: bool = False) -> Iterator[psycopg.Connection]:
    """Context manager that handles commit/rollback automatically."""
    disable_pooling = os.getenv("SPEEDLIMIT_DISABLE_POOLING", "0") == "1"

    if disable_pooling or ConnectionPool is None:
        conn = connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    pool = get_pool()
    with pool.connection() as conn:
        previous_row_factory = conn.row_factory
        if dict_rows:
            conn.row_factory = dict_row
        try:
            yield conn
            if not conn.autocommit:
                conn.commit()
        except Exception:
            if not conn.autocommit:
                conn.rollback()
            raise
        finally:
            conn.row_factory = previous_row_factory


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================


def configure_logging(log_file: Optional[Path]) -> None:
    """Configure logging to stdout and optional file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def configure_default_database_if_needed() -> None:
    """Ensure PostgreSQL connection settings default to the migrated database."""
    if not os.getenv("SPEEDLIMIT_DATABASE_URL"):
        os.environ["SPEEDLIMIT_DATABASE_URL"] = DEFAULT_DATABASE_URL
        logging.info("Using default database: %s", DEFAULT_DATABASE_URL.split('@')[1] if '@' in DEFAULT_DATABASE_URL else DEFAULT_DATABASE_URL)


def ensure_display_name_column() -> None:
    """Ensure the display_name column exists in PostgreSQL."""
    query = """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'coordinate_speed'
          AND column_name = 'display_name'
        LIMIT 1
    """
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            exists = cursor.fetchone()
            if exists:
                return
            logging.info("Adding display_name column to coordinate_speed table.")
            cursor.execute("ALTER TABLE coordinate_speed ADD COLUMN display_name TEXT;")


def create_performance_indexes() -> None:
    """Create indexes for better query performance if they don't exist."""
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            # Index for finding NULL display_names quickly
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_coordinate_speed_display_name_null 
                ON coordinate_speed (id DESC) 
                WHERE display_name IS NULL OR TRIM(display_name) = ''
            """)
            logging.info("Performance index created/verified for display_name updates.")
            conn.commit()


def fetch_next_batch(batch_size: int) -> List[RowData]:
    """Return the next batch of rows missing a display_name.
    
    Uses optimized query with index hint for faster retrieval.
    """
    query = """
        SELECT id, latitude, longitude
        FROM coordinate_speed
        WHERE display_name IS NULL OR TRIM(display_name) = ''
        ORDER BY id DESC
        LIMIT %s
    """
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (batch_size,))
            rows = cursor.fetchall()
    return rows


def apply_updates(updates: Sequence[UpdateData]) -> int:
    """Persist display_name values for the provided rows in a single transaction using optimized batch update."""
    if not updates:
        return 0

    # For very small batches, use simple UPDATE
    if len(updates) <= 5:
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    "UPDATE coordinate_speed SET display_name = %s WHERE id = %s",
                    [(display_name, row_id) for row_id, display_name in updates],
                )
                updated_rows = (
                    cursor.rowcount if cursor.rowcount not in (None, -1) else len(updates)
                )
        if updated_rows != len(updates):
            logging.warning(
                "Requested %s updates but only %s rows were modified; they may have been updated elsewhere.",
                len(updates),
                updated_rows,
            )
        return updated_rows
    
    # For larger batches, use temp table + UPDATE FROM for better performance
    with connection_scope() as conn:
        with conn.cursor() as cursor:
            # Create temporary table
            cursor.execute("""
                CREATE TEMP TABLE temp_display_updates (
                    id INTEGER,
                    display_name TEXT
                ) ON COMMIT DROP
            """)
            
            # Bulk insert into temp table using COPY for maximum speed
            with cursor.copy("COPY temp_display_updates (id, display_name) FROM STDIN") as copy:
                for row_id, display_name in updates:
                    # Escape special characters for COPY format
                    escaped_name = display_name.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    copy.write_row([row_id, escaped_name])
            
            # Batch update using JOIN - much faster than individual UPDATEs
            cursor.execute("""
                UPDATE coordinate_speed cs
                SET display_name = tdu.display_name
                FROM temp_display_updates tdu
                WHERE cs.id = tdu.id
            """)
            
            updated_rows = cursor.rowcount if cursor.rowcount not in (None, -1) else len(updates)
    
    if updated_rows != len(updates):
        logging.warning(
            "Requested %s updates but only %s rows were modified; they may have been updated elsewhere.",
            len(updates),
            updated_rows,
        )
    return updated_rows


def get_display_name(
    lat: float,
    lon: float,
    timeout: float,
    retries: int,
    backoff: float,
    sleep_seconds: float = 0.0,
) -> Optional[str]:
    """Call Nominatim to retrieve the display name with basic retry logic."""
    global last_api_call_time
    
    params = {
        "lat": f"{lat:.7f}",
        "lon": f"{lon:.7f}",
        "format": "json",
        "zoom": 18,
        "addressdetails": 0,
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    attempt = 0
    while True:
        attempt += 1
        
        # Rate limiting: ensure minimum delay between API calls (thread-safe)
        if sleep_seconds > 0:
            with api_call_lock:
                current_time = time.time()
                time_since_last_call = current_time - last_api_call_time
                if time_since_last_call < sleep_seconds:
                    time.sleep(sleep_seconds - time_since_last_call)
                last_api_call_time = time.time()
        
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            display_name = payload.get("display_name")
            if display_name:
                return display_name
            logging.warning(
                "display_name missing in response for lat=%s lon=%s", lat, lon
            )
            return None
        except HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt <= retries:
                wait_time = backoff * attempt
                logging.warning(
                    "HTTP %s from Nominatim. Retrying in %.1fs (attempt %s/%s).",
                    exc.code,
                    wait_time,
                    attempt,
                    retries,
                )
                time.sleep(wait_time)
                continue
            logging.error(
                "HTTP error from Nominatim for lat=%s lon=%s: %s", lat, lon, exc
            )
            return None
        except URLError as exc:
            if attempt <= retries:
                wait_time = backoff * attempt
                logging.warning(
                    "Network error contacting Nominatim (%s). Retrying in %.1fs (attempt %s/%s).",
                    exc.reason,
                    wait_time,
                    attempt,
                    retries,
                )
                time.sleep(wait_time)
                continue
            logging.error(
                "Network error contacting Nominatim for lat=%s lon=%s: %s",
                lat,
                lon,
                exc,
            )
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(
                "Unexpected error when fetching display_name for lat=%s lon=%s: %s",
                lat,
                lon,
                exc,
            )
            return None


def process_single_row(
    row_data: RowData,
    sleep_seconds: float,
    timeout: float,
    retries: int,
    backoff: float,
) -> Optional[UpdateData]:
    """Process a single row and return the update data if successful."""
    row_id, lat, lon = row_data
    
    if lat is None or lon is None:
        logging.info("Skipping row id=%s due to missing coordinates.", row_id)
        return None

    display_name = get_display_name(lat, lon, timeout, retries, backoff, sleep_seconds)
    if display_name:
        return (row_id, display_name)
    else:
        logging.info("No display_name fetched for id=%s; will retry later.", row_id)
        return None


def process_batch(
    rows: Sequence[RowData],
    sleep_seconds: float,
    timeout: float,
    retries: int,
    backoff: float,
    limit: Optional[int] = None,
    workers: int = 1,
) -> Tuple[int, int]:
    """Update display_name for a batch of rows using multi-threading.

    Returns a tuple of (updated_rows, attempted_rows).
    """
    updated = 0
    attempted = 0
    pending_updates: List[UpdateData] = []
    
    rows_to_process = rows if limit is None else rows[:limit]
    attempted = len(rows_to_process)
    
    if workers <= 1:
        # Single-threaded mode (original behavior)
        for row_data in rows_to_process:
            result = process_single_row(row_data, sleep_seconds, timeout, retries, backoff)
            if result:
                pending_updates.append(result)
    else:
        # Multi-threaded mode
        logging.info("Processing %s rows with %s worker threads.", len(rows_to_process), workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_row = {
                executor.submit(
                    process_single_row,
                    row_data,
                    sleep_seconds,
                    timeout,
                    retries,
                    backoff
                ): row_data for row_data in rows_to_process
            }
            
            completed_count = 0
            for future in as_completed(future_to_row):
                completed_count += 1
                try:
                    result = future.result()
                    if result:
                        pending_updates.append(result)
                    # Log progress every 20% for large batches
                    if len(rows_to_process) >= 20 and completed_count % max(1, len(rows_to_process) // 5) == 0:
                        logging.info("Progress: %s/%s rows completed (%.1f%%)", 
                                   completed_count, len(rows_to_process), 
                                   (completed_count / len(rows_to_process)) * 100)
                except Exception as exc:  # pylint: disable=broad-except
                    row_data = future_to_row[future]
                    logging.exception(
                        "Unexpected error processing row id=%s: %s",
                        row_data[0],
                        exc
                    )

    if pending_updates:
        logging.info("Applying %s updates to database...", len(pending_updates))
        updated = apply_updates(pending_updates)
        logging.info("Successfully updated %s/%s rows.", updated, len(pending_updates))
    else:
        logging.info("No rows were successfully fetched in this batch.")

    return updated, attempted


def run_once(
    batch_size: int,
    sleep_seconds: float,
    timeout: float,
    retries: int,
    backoff: float,
    max_rows: Optional[int] = None,
    workers: int = 1,
) -> int:
    """Process the table until no pending rows remain."""
    total_updated = 0
    total_attempted = 0
    batch_count = 0
    start_time = time.time()
    
    remaining = max_rows if max_rows is not None else None
    while True:
        if remaining is not None and remaining <= 0:
            break

        effective_batch_size = (
            batch_size if remaining is None else min(batch_size, remaining)
        )
        rows = fetch_next_batch(effective_batch_size)
        if not rows:
            break

        batch_count += 1
        batch_start = time.time()
        
        rows_limit = remaining if remaining is not None else None
        updated, attempted = process_batch(
            rows,
            sleep_seconds,
            timeout,
            retries,
            backoff,
            limit=rows_limit,
            workers=workers,
        )
        
        batch_time = time.time() - batch_start
        total_updated += updated
        total_attempted += attempted
        
        if updated > 0:
            rate = updated / batch_time if batch_time > 0 else 0
            logging.info("Batch #%s: %s rows updated in %.2fs (%.2f rows/sec)", 
                        batch_count, updated, batch_time, rate)
        
        if remaining is not None:
            remaining -= attempted
    
    total_time = time.time() - start_time
    if total_updated > 0 and total_time > 0:
        overall_rate = total_updated / total_time
        logging.info("Summary: %s total rows updated in %.2fs (%.2f rows/sec average)", 
                    total_updated, total_time, overall_rate)
    
    return total_updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate coordinate_speed.display_name using Nominatim reverse geocoding."
    )
    parser.add_argument(
        "--database-url",
        dest="database_url",
        help="Optional PostgreSQL connection URL. "
        "Defaults to environment variables or the built-in credentials.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Number of rows to read per batch. Larger batches = faster updates but more memory."
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to wait between API calls to respect Nominatim usage policy. Lower with more workers.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of worker threads for parallel API calls. "
        "WARNING: Nominatim rate limit is 1 req/sec. Use with caution! "
        "Recommended: 3-5 workers with --sleep 0.2-0.3 for optimal speed",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="Timeout for each HTTP request in seconds."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for recoverable HTTP errors.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Backoff multiplier for retries (seconds multiplied by attempt count).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep watching for new rows and rerun indefinitely.",
    )
    parser.add_argument(
        "--idle-sleep",
        type=float,
        default=60.0,
        help="When --loop is set, seconds to sleep before rescanning if no rows need updates.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Process at most this many rows per pass (useful for testing).",
    )
    parser.add_argument("--log-file", help="Optional log file path.")
    args = parser.parse_args()

    log_path = Path(args.log_file) if args.log_file else None
    configure_logging(log_path)

    if args.database_url:
        os.environ["SPEEDLIMIT_DATABASE_URL"] = args.database_url
    else:
        configure_default_database_if_needed()

    logging.info("Starting display name updater. loop=%s, workers=%s", args.loop, args.workers)
    
    if args.workers > 1:
        logging.warning(
            "Running with %s worker threads. "
            "Please ensure you respect Nominatim usage policy to avoid being blocked!",
            args.workers
        )

    os.environ.setdefault("SPEEDLIMIT_DISABLE_POOLING", "1")
    pool = None
    if os.getenv("SPEEDLIMIT_DISABLE_POOLING", "0") != "1":
        try:
            pool = get_pool()
            logging.info(
                "Connection pooling enabled (size %s-%s).",
                getattr(pool, "min_size", "unknown"),
                getattr(pool, "max_size", "unknown"),
            )
        except RuntimeError as exc:
            logging.warning(
                "Connection pooling unavailable (%s); falling back to direct connections.",
                exc,
            )

    ensure_display_name_column()
    create_performance_indexes()

    while True:
        updated = run_once(
            args.batch_size,
            args.sleep,
            args.timeout,
            args.retries,
            args.backoff,
            args.max_rows,
            args.workers,
        )
        if updated:
            logging.info("Updated %s rows in this pass.", updated)
        else:
            logging.info("No rows needed updates.")

        if not args.loop:
            break

        time.sleep(args.idle_sleep)

    logging.info("Display name updater finished.")


if __name__ == "__main__":
    main()
