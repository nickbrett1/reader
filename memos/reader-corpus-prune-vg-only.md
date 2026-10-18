# reader — prune the corpus back to video-game books

**Status:** ✅ local and ryzen corpora pruned 2026-10-18. The live corpus now
has 8 video-game books.

## Why

`reader-corpus-widen-and-eval.md` pulled the whole staging library to widen the
embedding eval. That included 35 off-topic epubs — 18 `Data Visualization/` and
17 `Machine Learning, AI and Bots/` — which became books 4–38. A later
re-`manifest` after the staging tree was reorganised added 36 more rows (ids
45–80), duplicating the same AI/Data-Viz titles under prefixed work keys. The
corpus should be video-game books only.

Book 42 (`079b3a_…pdf`) is **Let's Go Dizzy!** by Chris Wilkins and Roger M.
Kean — on-topic, kept. Books 41 and 44 are two editions of *Arcade Decades:
80s*; **44** (normal layout) is kept and **41** (`(single)`) removed. The AI
concepts were extracted after the widening, so pruning them also removed 5,955
orphaned concepts.

## Code changes

- `.ingestignore` support in `manifest.walk`: one pattern per line at the root
  of `READER_INGEST`; a bare name matches a path component anywhere, a pattern
  with `/` matches the relative path.
- `reader prune` verb: deletes book/pages/chunks/images and orphaned concepts
  from SQLite in one transaction, unlinks image blobs, optionally unlinks
  sources (`--delete-sources`), then prunes and compacts the LanceDB index.
  `--dry-run` lists matches; `--yes` confirms.

## Local (devcontainer)

`reader prune --books 4-38,41 --delete-sources --yes` (41 first, then 4–38):

| before | after |
|---|---|
| 44 books · 24 554 chunks · 3 063 images · 756 concepts | **8 books · 4 444 chunks · 3 063 images · 756 concepts** |
| index 28 373 rows | **index 8 263 rows** |

Backup of the pre-prune local DB: `.reader-data/reader.db.pre-prune-backup`.
Local image/concept counts differ from ryzen because those passes had been run
further there; book ids and chunks match.

## Ryzen (live deployment)

Buildkite build 45 published the `prune` + `.ingestignore` image; the ryzen
container was pulled forward to it. `/srv/reader/Books/.ingestignore`:

```
Data Visualization
Machine Learning, AI and Bots
NP_ArcadeDecades80s_Book(single).pdf
```

`reader prune --books 4-38,41,45-80 --yes` (no `--delete-sources`: `/ingest` is
read-only), then `reader index --optimize`:

| | before | after |
|---|---:|---:|
| Books | 80 | **8** |
| Pages | 4,284 | **1,836** |
| Chunks | 44,478 | **4,444** |
| Images | 10,643 | **2,860** |
| Concepts | 8,536 | **2,581** |
| Index rows | 63,657 | **9,885** |

Surviving ids: 1, 2, 3, 39, 40, 42, 43, 44. `reader manifest` now reports
**8 work(s)**. The 290 MB `reader.db-wal` will be reclaimed on the next clean
container shutdown.

## If the staging library changes again

Keep `.ingestignore` in step with the library: any directory or file that
should stay out of the corpus goes there *before* the next scheduled pipeline
run. Removing an already-ingested book is `reader prune --books <ids> --yes`.
