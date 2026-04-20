"""
Minimal migration runner.

Usage:
    poetry run python -m app.migrate          # apply pending
    poetry run python -m app.migrate --status  # show status
"""

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

from app.infrastructure.config.settings import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

ENSURE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def _get_applied(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
    return {r["version"] for r in rows}


async def _apply(conn: asyncpg.Connection, version: str, sql: str) -> None:
    async with conn.transaction():
        await conn.execute(sql)
        await conn.execute(
            "INSERT INTO schema_migrations (version) VALUES ($1)", version
        )


def _discover() -> list[tuple[str, Path]]:
    if not MIGRATIONS_DIR.is_dir():
        return []
    return [(f.stem, f) for f in sorted(MIGRATIONS_DIR.glob("*.sql"))]


async def run(*, status_only: bool = False) -> None:
    settings = Settings()
    conn = await asyncpg.connect(str(settings.database_url))

    try:
        await conn.execute(ENSURE_TABLE)
        applied = await _get_applied(conn)
        migrations = _discover()

        if status_only:
            for version, _ in migrations:
                mark = "✓" if version in applied else "·"
                print(f"  {mark}  {version}")
            return

        pending = [(v, p) for v, p in migrations if v not in applied]
        if not pending:
            print("No pending migrations.")
            return

        for version, path in pending:
            sql = path.read_text()
            await _apply(conn, version, sql)
            print(f"  ✓  {version}")

        print(f"\nApplied {len(pending)} migration(s).")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--status", action="store_true", help="Show migration status without applying"
    )
    args = parser.parse_args()
    asyncio.run(run(status_only=args.status))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
