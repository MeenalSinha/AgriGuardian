"""Unit tests for SQLite to PostgreSQL migration tool (Phase 12)."""

import sqlite3
import tempfile
from pathlib import Path
import pytest

from scripts.migrate_sqlite_to_postgres import (
    compute_topological_table_order,
    generate_postgres_ddl,
    get_table_metadata,
    map_sqlite_type_to_postgres,
    run_dry_run,
)


@pytest.fixture
def mock_sqlite_db():
    """Create a temporary SQLite database with parent and child tables."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("""
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ
        );
    """)
    conn.execute("""
        CREATE TABLE farmer_profiles (
            farmer_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id),
            full_name TEXT NOT NULL,
            total_land_acres REAL
        );
    """)
    conn.execute("""
        CREATE TABLE farms (
            farm_id TEXT PRIMARY KEY,
            farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id),
            farm_name TEXT NOT NULL
        );
    """)
    conn.execute("INSERT INTO users VALUES ('u1', 'ramesh', '2026-01-01');")
    conn.execute("INSERT INTO farmer_profiles VALUES ('f1', 'u1', 'Ramesh Patel', 4.5);")
    conn.execute("INSERT INTO farms VALUES ('farm1', 'f1', 'North Field');")
    conn.commit()
    conn.close()

    yield db_path

    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_map_sqlite_type_to_postgres():
    """Verify accurate type mapping between SQLite and PostgreSQL."""
    assert map_sqlite_type_to_postgres("id", "INTEGER", True) == "BIGINT PRIMARY KEY"
    assert map_sqlite_type_to_postgres("count", "INTEGER", False) == "BIGINT"
    assert map_sqlite_type_to_postgres("acres", "REAL", False) == "DOUBLE PRECISION"
    assert map_sqlite_type_to_postgres("name", "TEXT", False) == "TEXT"
    assert map_sqlite_type_to_postgres("is_active", "BOOLEAN", False) == "BOOLEAN"
    assert map_sqlite_type_to_postgres("avatar", "BLOB", False) == "BYTEA"


def test_topological_ordering(mock_sqlite_db):
    """Verify that child tables are ordered strictly after their foreign key parents."""
    conn = sqlite3.connect(str(mock_sqlite_db))
    try:
        tables = ["farms", "farmer_profiles", "users"]
        tables_meta = {t: get_table_metadata(conn, t) for t in tables}
    finally:
        conn.close()

    ordered = compute_topological_table_order(tables_meta)

    # users must be created before farmer_profiles, and farmer_profiles before farms
    assert ordered.index("users") < ordered.index("farmer_profiles")
    assert ordered.index("farmer_profiles") < ordered.index("farms")


def test_generate_postgres_ddl(mock_sqlite_db):
    """Verify generation of valid PostgreSQL DDL statements."""
    conn = sqlite3.connect(str(mock_sqlite_db))
    try:
        meta = get_table_metadata(conn, "farmer_profiles")
    finally:
        conn.close()

    ddl = generate_postgres_ddl(meta)
    assert 'CREATE TABLE IF NOT EXISTS "farmer_profiles"' in ddl
    assert '"farmer_id" TEXT' in ddl
    assert '"total_land_acres" DOUBLE PRECISION' in ddl


def test_run_dry_run_non_destructive(mock_sqlite_db):
    """Verify that dry-run successfully inspects schema and records without writes."""
    result = run_dry_run(mock_sqlite_db, "postgresql://user:pass@localhost:5432/testdb")

    assert result["tables_count"] == 3
    assert result["total_records"] == 3
    assert "users" in result["ordered_tables"]
    assert "farmer_profiles" in result["ordered_tables"]
    assert "farms" in result["ordered_tables"]

