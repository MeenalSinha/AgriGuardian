#!/usr/bin/env python
"""Database restoration and integrity verification utility for AgriGuardian (Phase 12).

Verifies cryptographic SHA-256 checksums, restores database state, executes PRAGMA integrity_check,
and validates table record counts to prove business continuity readiness.
"""

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("DatabaseRestore")


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sqlite_backup_integrity(backup_path: Path) -> bool:
    """Execute low-level SQLite integrity check on backup file."""
    conn = sqlite3.connect(str(backup_path))
    try:
        cursor = conn.execute("PRAGMA integrity_check;")
        rows = cursor.fetchall()
        if rows and rows[0][0] == "ok":
            logger.info("SQLite PRAGMA integrity_check passed: 'ok'")
            return True
        else:
            logger.error(f"SQLite PRAGMA integrity_check failed: {rows}")
            return False
    finally:
        conn.close()


def inspect_backup_record_counts(backup_path: Path) -> dict:
    """Inspect and report core table record counts from backup."""
    conn = sqlite3.connect(str(backup_path))
    counts = {}
    key_tables = [
        "users",
        "farmer_profiles",
        "farms",
        "farm_plots",
        "crop_cycles",
        "farm_expenses",
        "farm_harvest_records",
        "farm_sales",
        "sync_devices",
    ]
    try:
        for tbl in key_tables:
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {tbl};")
                counts[tbl] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                counts[tbl] = "N/A"
    finally:
        conn.close()
    return counts


def restore_sqlite(backup_path: Path, target_path: Path, verify_checksum: bool = True) -> bool:
    """Restore SQLite database from verified backup."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found at {backup_path}")

    # 1. Verify Checksum if present
    checksum_file = backup_path.parent / f"{backup_path.name}.sha256"
    if checksum_file.exists() and verify_checksum:
        with open(checksum_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            expected_hash = line.split()[0]
        actual_hash = compute_sha256(backup_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(f"Checksum mismatch: expected {expected_hash}, calculated {actual_hash}")
        logger.info(f"SHA-256 checksum verified: {actual_hash[:12]}...")

    # 2. Verify Internal Database Integrity
    if not verify_sqlite_backup_integrity(backup_path):
        raise RuntimeError("Corrupted backup file: PRAGMA integrity_check failed.")

    # 3. Restore via atomic SQLite backup API
    logger.info(f"Restoring from {backup_path} to {target_path}...")
    source_conn = sqlite3.connect(str(backup_path))
    target_conn = sqlite3.connect(str(target_path))

    with target_conn:
        source_conn.backup(target_conn, pages=250, sleep=0.01)

    target_conn.close()
    source_conn.close()

    # 4. Verify Restored Target
    if not verify_sqlite_backup_integrity(target_path):
        raise RuntimeError("Restoration validation failed: target database is corrupted.")

    counts = inspect_backup_record_counts(target_path)
    logger.info(f"Restoration verified successfully. Record counts: {counts}")
    return True


def main():
    parser = argparse.ArgumentParser(description="AgriGuardian Database Restore & Verification Tool")
    parser.add_argument("--backup-file", required=True, help="Path to backup .db file to restore")
    parser.add_argument("--target-path", help="Target database path (defaults to settings.farm_db_path)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify integrity and print stats without restoring")
    args = parser.parse_args()

    settings = get_settings()
    backup_path = Path(args.backup_file).resolve()
    target_path = Path(args.target_path or settings.farm_db_path).resolve()

    if args.verify_only:
        integrity_ok = verify_sqlite_backup_integrity(backup_path)
        counts = inspect_backup_record_counts(backup_path)
        print(f"VERIFY RESULT: Integrity={'OK' if integrity_ok else 'FAILED'}")
        print(f"RECORD COUNTS: {counts}")
        sys.exit(0 if integrity_ok else 1)

    success = restore_sqlite(backup_path, target_path)
    if success:
        print(f"SUCCESS: Database restored and verified at {target_path}")
    else:
        print("ERROR: Restore operation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

