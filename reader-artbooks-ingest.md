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

## Resume here / status

🟡 **IN PROGRESS** — see "Running log" above for the last completed stage.
