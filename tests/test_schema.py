"""Schema / migration round-trip and the schema_version guard."""

from __future__ import annotations

import pytest

from reader import schema
from reader.store import SchemaMismatch, Store, apply_migrations, check_schema, connect


def test_fresh_db_reaches_current_version(tmp_path):
    store = Store(tmp_path / "r.db")
    try:
        assert store.get_embedding_config() is not None
        check_schema(store.conn, schema.SCHEMA_VERSION)
    finally:
        store.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "r.db"
    conn = connect(path)
    v1 = apply_migrations(conn)
    v2 = apply_migrations(conn)
    assert v1 == v2 == schema.SCHEMA_VERSION
    # One row is maintained, not one per migration.
    n = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert n == 1
    conn.close()


def test_schema_guard_rejects_drift(tmp_path):
    store = Store(tmp_path / "r.db")
    try:
        with pytest.raises(SchemaMismatch):
            check_schema(store.conn, schema.SCHEMA_VERSION + 99)
    finally:
        store.close()


def test_pin_embedding_config(tmp_path):
    store = Store(tmp_path / "r.db")
    try:
        store.pin_embedding_config(2048, "cohere/embed-v5.0")
        cfg = store.get_embedding_config()
        assert cfg["embedding_dimensions"] == 2048
        assert cfg["embedding_model"] == "cohere/embed-v5.0"
        with pytest.raises(ValueError):
            store.pin_embedding_config(1024, "x")
    finally:
        store.close()
