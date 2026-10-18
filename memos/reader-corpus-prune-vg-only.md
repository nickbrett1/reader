# reader — prune the corpus back to video-game books

**Status:** 🟢 local corpus pruned 2026-10-18; NAS deployment still to apply.

## Why

`reader-corpus-widen-and-eval.md` pulled the whole NAS staging library to widen
the embedding eval. That included 35 off-topic epubs — 18 `Data Visualization/`
and 17 `Machine Learning, AI and Bots/` — which became books 4–38. The corpus
should be video-game books only, so they are being removed.

## What was done locally (devcontainer)

- Added `.ingestignore` support to `manifest.walk` (bare names match a path
  component anywhere; patterns with `/` match the relative path).
- Added a `reader prune` verb: deletes book/pages/chunks/images and orphaned
  concepts from SQLite in one transaction, unlinks image blobs, optionally
  unlinks sources (`--delete-sources`), then prunes and compacts the LanceDB
  index. `--dry-run` lists matches; `--yes` confirms.
- Ran `reader prune --books 4-38 --delete-sources --yes`:

| before | after |
|---|---|
| 44 books · 24 554 chunks · 3 063 images · 756 concepts | **9 books · 5 200 chunks · 3 063 images · 756 concepts** |
| index 28 373 rows | **index 9 019 rows** |

No images or concept mentions belonged to the removed books, so the concept
graph is untouched. Five thousand two hundred chunks cover books 1, 2, 3 and
39–44 (all video-game titles, including the BG3/`079b3a` artbooks and the
`NP_ArcadeDecades80s` single-page vs normal-layout editions).

Backup of the pre-prune local DB: `.reader-data/reader.db.pre-prune-backup`.

## NAS steps (deployment)

The container mounts `/volume1/homes/nick/Books:/ingest:ro`, so the scheduled
pipeline will re-add the two off-topic folders unless they are ignored.

1. Create the ignore file on the NAS host:

   ```bash
   cat > /volume1/homes/nick/Books/.ingestignore <<'EOF'
   Data Visualization
   Machine Learning, AI and Bots
   EOF
   ```

2. Push this commit so Buildkite publishes a new image; let Watchtower update
   the `reader` container.

3. Rehearse, then prune. Book ids should match the local DB; confirm the work
   keys in the dry run are the AI/Data-Viz tranche before deleting:

   ```bash
   docker exec reader reader prune --books 4-38 --dry-run
   docker exec reader reader prune --books 4-38 --yes
   ```

   Do **not** pass `--delete-sources` on the NAS: `/ingest` is read-only. The
   source library stays intact; `.ingestignore` keeps it out of the corpus.

4. Verify:

   ```bash
   docker exec reader reader index            # expect 9 019 rows
   docker exec reader reader manifest | tail -1  # expect 9 work(s)
   ```
