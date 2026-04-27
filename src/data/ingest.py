"""CLI entry-point for the `ingest` DVC stage.

Environment overrides (intended for local / CI small runs; full ingest
leaves them unset):

    INGEST_SOURCES   comma-separated adapter names to run
                     (default: all adapters in ALL_ADAPTERS)
    INGEST_LIMIT     int; per-adapter row cap applied on first download
                     (default: unlimited)
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from src.data.sources import ALL_ADAPTERS

RAW_DIR = Path("data/raw")


def main() -> None:
    names_filter = os.getenv("INGEST_SOURCES")
    allow = set(names_filter.split(",")) if names_filter else None
    limit_env = os.getenv("INGEST_LIMIT")
    limit = int(limit_env) if limit_env else None

    for cls in ALL_ADAPTERS:
        if allow is not None and cls.name not in allow:
            logger.info(f"{cls.name}: skipped (not in INGEST_SOURCES)")
            continue
        adapter = cls(cache_dir=RAW_DIR / cls.name)
        df = adapter.load(limit=limit)
        logger.info(f"{cls.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
