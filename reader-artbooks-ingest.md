# reader — pull + ingest the three big video-game art/PDF books, then re-embed & re-eval

**Goal:** complete the corpus. Pull the three remaining on-topic NAS PDFs (the
image-heavy artbooks skipped last pass), ingest them end-to-end (manifest →
extract → images → chunk), run the DeepSeek off-peak concept pass over the new
books, re-embed everything, optimize the index, and re-run the 15-label
pro-vs-fast eval on the enlarged corpus.

**Status:** 🟡 IN PROGRESS — started 2026-10-13T00:20Z.

**Predecessor:** `reader-corpus-widen-and-eval.md` (complete: 41 books / 23 380
text chunks; pro == fast tie at recall@10 = 0.357 over 15 labels).

---

## Plan (the 5 steps this pass)

1. **Pull** `BG3_Artbook_High_Res.pdf` (301 586 309), `NP_ArcadeDecades80s_Book.pdf`
   (101 830 336), `079b3a_8538d8d669ee4df6b406d216ad3569eb.pdf` (353 272 478) over the
   `nas-goose` HTTP relay (`python3 -m http.server 8099` in `/volume1/homes/nick/Books`;
   kill it when done). Record size + sha256 each.
2. **Ingest** the 3 new books: `manifest --commit` → `extract` → `images` → `chunk`.
   Image-heavy ⇒ record image counts, dedupe collapse, and wall time per book.
3. **Concepts** — `reader concepts --llm-model deepseek-flash` for books 39–44
   (DeepSeek off-peak; use the gate if peak — `scripts/deepseek_offpeak.py`).
4. **Embed** the new text/chunks/images/concepts (`cohere/embed-v5.0`, 2048) then
   `reader index --optimize`.
5. **Eval** — re-run the 15-label eval; record pro vs fast and the new corpus size.

---

## Running log

- _00:20Z_ — note opened; inspecting current DB state before the pull.
- _00:26Z_ — pull done via the nas-goose relay (`python3 -m http.server 8099`):
  BG3 301 586 309 (sha256 4b68866c…), NP_Arcade 101 830 336 (b7c7abaf…),
  079b3a 353 272 478 (b1c104ce…). Relay killed.
- _00:28–00:40Z_ — manifest/extract/chunk/images for the 3 new books (42–44).
  Images: book 42 = 691 unique (851 raw), 43 = 1073 (1422 raw), 44 = 618
  (1098 raw) → 2 382 new image rows, **0 embedded**.
  Text: +1 174 chunks (total 24 554); pages: 236/97/162.
- _00:22–00:57Z_ — a **fast** re-embed (`reembed --model cohere/embed-v5.0-fast`)
  ran concurrently and **FAILED**: `gateway request failed: timed out`
  (httpx.ReadTimeout) during the IMAGE phase. It had already stamped all
  23 380 old chunks fast in SQLite, but `cmd_embed` only upserts the Lance index
  at the very end — so the run aborted **before** any index write. Net effect:
  SQLite provenance said "fast" while the Lance index still held **pro** vectors.
- _00:58Z_ — chained confirm eval printed `cohere/embed-v5.0-fast:
  recall@10 = 0.357 over 15 queries` (query-side only; corpus vectors were still pro).

## Fixes applied

- **`4ca1525`** — `EmbedClient` timeout 60 s → **180 s** + retry (×2, backoff) on
  transient `Timeout`/`Transport` errors; definite HTTP status errors still fail
  fast. Regression test: `test_post_retries_transient_timeout_then_succeeds`.
- Cleared provenance (`embed_model_id/dimensions/embed_config_version/embedded_at
  = NULL`) on chunk (24 554), image (3 063), concept (56) so the index is rebuilt
  **in fast**, not left as pro.

## Resume here / status

🟡 **IN PROGRESS** — recovery run launched _00:59Z_: `reader embed
--model cohere/embed-v5.0-fast --dimensions 2048` → `index --optimize` → `eval`,
chained, to `/tmp/reembed-fast2.log`. Expect ~27 673 index rows.

**Still to do after it lands:** the DeepSeek concept pass for books 42–44
(`reader concepts --llm-model deepseek-flash --books 42,43,44` — the `--books`
flag is uncommitted in `src/reader/__main__.py`), gated to off-peak
(01:00–04:00 / 06:00–10:00 UTC are peak). Then re-run the eval to include
concepts.
