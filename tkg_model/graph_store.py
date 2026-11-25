"""SQLite 中的犯罪 TKG：实体、四元组、更新日志与历史属性。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tkg_pipeline.config import OUTPUT_DIR

TKG_DIR = OUTPUT_DIR / "tkg"
DB_PATH = TKG_DIR / "crime_tkg.sqlite"
NEO4J_DIR = TKG_DIR / "neo4j"
CONFIG_PATH = TKG_DIR / "update_config.json"

QUAD_COLUMNS = (
    "entity1_id",
    "relation_type",
    "relation_subtype",
    "entity2_id",
    "iso8601_timestamp",
    "valid_from",
    "valid_to",
    "confidence",
    "strength",
    "source",
    "case_number",
    "other_case",
    "unix_ts",
    "window_hours",
    "batch_id",
    "increment_flag",
)


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_class TEXT NOT NULL,
            standard_code TEXT,
            name TEXT,
            level TEXT,
            description TEXT,
            attrs_json TEXT,
            batch_id TEXT,
            increment_flag INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS quads (
            quad_id INTEGER PRIMARY KEY,
            entity1_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            relation_subtype TEXT NOT NULL,
            entity2_id TEXT NOT NULL,
            iso8601_timestamp TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            confidence REAL,
            strength REAL,
            source TEXT,
            case_number TEXT DEFAULT '',
            other_case TEXT DEFAULT '',
            unix_ts INTEGER,
            window_hours REAL,
            batch_id TEXT,
            increment_flag INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS attribute_history (
            hist_id INTEGER PRIMARY KEY,
            batch_id TEXT,
            entity_id TEXT,
            attribute TEXT,
            old_value TEXT,
            new_value TEXT,
            changed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS update_log (
            batch_id TEXT PRIMARY KEY,
            started_at TEXT,
            finished_at TEXT,
            mode TEXT,
            source_file TEXT,
            new_entities INTEGER,
            new_quads INTEGER,
            duplicate_skipped INTEGER,
            conflicts INTEGER,
            note TEXT
        );
        CREATE TABLE IF NOT EXISTS conflicts (
            conflict_id INTEGER PRIMARY KEY,
            batch_id TEXT,
            conflict_type TEXT,
            entity1_id TEXT,
            relation_subtype TEXT,
            entity2_id TEXT,
            iso8601_timestamp TEXT,
            case_number TEXT,
            existing_value TEXT,
            incoming_value TEXT,
            resolution TEXT,
            created_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_quad_key ON quads (
            entity1_id, relation_subtype, entity2_id, iso8601_timestamp,
            case_number, other_case
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quad_e1 ON quads(entity1_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quad_e2 ON quads(entity2_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quad_rel ON quads(relation_type, relation_subtype)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quad_time ON quads(iso8601_timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quad_case ON quads(case_number)")
    conn.commit()


def insert_entities(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO entities (
            entity_id, entity_class, standard_code, name, level, description,
            attrs_json, batch_id, increment_flag
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def insert_quads(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    before = conn.total_changes
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO quads ({",".join(QUAD_COLUMNS)})
        VALUES ({",".join("?" for _ in QUAD_COLUMNS)})
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def reset_database(path: Path = DB_PATH) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    wal = Path(str(path) + "-wal")
    shm = Path(str(path) + "-shm")
    if wal.exists():
        wal.unlink()
    if shm.exists():
        shm.unlink()
    conn = connect(path)
    init_db(conn)
    return conn


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
