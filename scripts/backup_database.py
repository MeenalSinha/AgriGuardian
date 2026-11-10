#!/usr/bin/env python
"""Automated and consistent online database backup utility for AgriGuardian (Phase 12).

Supports consistent live snapshotting of SQLite using sqlite3.Connection.backup()
as well as PostgreSQL pg_dump when running against production PostgreSQL instances.
Computes cryptographic SHA-256 integrity checksums for every backup artifact.
"""

import argparse
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("DatabaseBackup")


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def backup_sqlite(source_path: Path, backup_dir: Path) -> Path:
    """Perform atomic online backup of SQLite database without blocking concurrent readers/writers."""
    if not source_path.exists():
        logger.info(f"Source database not found at {source_path}. Initializing database schema...")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        from backend.app.repositories.farm_db import FarmDatabase
        FarmDatabase(db_path=str(source_path))

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_filename = f"farm_guardian_backup_{timestamp_str}.db"
    dest_path = backup_dir / backup_filename

    logger.info(f"Initiating online atomic backup from {source_path} to {dest_path}...")
    source_conn = sqlite3.connect(str(source_path))
    dest_conn = sqlite3.connect(str(dest_path))

    with dest_conn:
        source_conn.backup(dest_conn, pages=250, sleep=0.01)

    dest_conn.close()
    source_conn.close()

    # Calculate SHA-256 checksum and save metadata
    sha256 = compute_sha256(dest_path)
    checksum_path = backup_dir / f"{backup_filename}.sha256"
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256}  {backup_filename}\n")

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    logger.info(f"Backup completed successfully: {dest_path} ({size_mb:.2f} MB, SHA-256: {sha256[:12]}...)")
    return dest_path


def backup_postgres(db_url: str, backup_dir: Path) -> Path:
    """Execute pg_dump for production PostgreSQL database."""
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_filename = f"postgres_backup_{timestamp_str}.sql"
    dest_path = backup_dir / backup_filename

    logger.info(f"Executing pg_dump to {dest_path}...")
    cmd = ["pg_dump", "--dbname", db_url, "-f", str(dest_path)]
    subprocess.run(cmd, check=True)

    sha256 = compute_sha256(dest_path)
    checksum_path = backup_dir / f"{backup_filename}.sha256"
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256}  {backup_filename}\n")

    return dest_path


def main():
    parser = argparse.ArgumentParser(description="AgriGuardian Database Backup Tool")
    parser.add_argument("--dest-dir", help="Target backup directory (defaults to settings.backup_dir)")
    args = parser.parse_args()

    settings = get_settings()
    backup_dir = Path(args.dest_dir or settings.backup_dir).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)

    if settings.database_url and settings.database_url.startswith("postgres"):
        backup_path = backup_postgres(settings.database_url, backup_dir)
    else:
        source_path = Path(settings.farm_db_path).resolve()
        backup_path = backup_sqlite(source_path, backup_dir)

    print(f"SUCCESS: Backup written to {backup_path}")


if __name__ == "__main__":
    main()

