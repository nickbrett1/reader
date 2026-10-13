#!/usr/bin/env bash
# Wait for the in-flight pro embed (pid 180138) to exit, then embed whatever
# chunks/concepts/images remain unstamped for pro, and rebuild the index.
set -u
cd /workspaces/reader
export READER_INGEST=/workspaces/reader/.ingest
export READER_DATA=/workspaces/reader/.reader-data
export LITELLM_BASE_URL=http://nas:4000
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-d432ff3fa496fb0e039915995422e9b05b881ba02cdb5d5c}"

echo "[watch] $(date -u +%H:%M:%S) waiting for pid 180138 ..."
while kill -0 180138 2>/dev/null; do sleep 10; done
echo "[watch] $(date -u +%H:%M:%S) pid 180138 exited"

.venv/bin/python -c "import sqlite3;c=sqlite3.connect('.reader-data/reader.db');print('[watch] stamped pro:',c.execute(\"select count(*) from chunk where embed_model_id='cohere/embed-v5.0'\").fetchone()[0],'/',c.execute('select count(*) from chunk').fetchone()[0])"

echo "[watch] $(date -u +%H:%M:%S) follow-up embed (pro, remaining chunks) ..."
.venv/bin/python -m reader embed --model cohere/embed-v5.0 --dimensions 2048
echo "[watch] $(date -u +%H:%M:%S) index rebuild ..."
.venv/bin/python -m reader index
echo "[watch] $(date -u +%H:%M:%S) DONE"
.venv/bin/python -c "import sqlite3;c=sqlite3.connect('.reader-data/reader.db');print('[watch] stamped pro final:',c.execute(\"select count(*) from chunk where embed_model_id='cohere/embed-v5.0'\").fetchone()[0],'/',c.execute('select count(*) from chunk').fetchone()[0])"
