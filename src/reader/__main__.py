"""CLI entry point: argparse + verb dispatch.

The app owns its entry point (genproj emits no stub ``__main__.py`` when a
``command`` is set), so a later regeneration cannot clobber this file.

Verbs are the ``dagu`` task surface (§5.2):

    manifest  extract  images  chunk  concepts  embed  eval  reembed

Every stage is idempotent and resumable; re-running ``embed`` after a model
change is the normal path, not an exception.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from . import chunk as chunk_mod
from . import concepts as concepts_mod
from . import extract as extract_mod
from . import images as images_mod
from . import index as index_mod
from . import manifest as manifest_mod
from . import schema
from .config import Config
from .embed import EmbedClient
from .store import Store

#: Default embedding model id (Cohere via the gateway).  ``eval`` decides
#: pro-vs-fast; ``reembed --model`` overrides either.
DEFAULT_MODEL = "cohere/embed-v5.0-fast"  # pro==fast on recall (2026-10-12, 23k chunks); fast is cheaper
EVAL_MODELS = (
    "cohere/embed-v5.0",
    "cohere/embed-v5.0-fast",
    "gemini/gemini-embedding-2",
)

#: ~10 hand-written queries for the first §11.1 invocation, including
#: deliberate cross-modal probes (a text query whose best answer is an image,
#: and vice versa).  Relevance labels come from ``--labels`` (JSON).
EVAL_QUERIES: list[dict[str, str]] = [
    {"q": "maps of the game world", "modality": "text"},
    {"q": "concept art of the main character", "modality": "text"},
    {"q": "the chapter on combat mechanics", "modality": "text"},
    {"q": "publisher credits and copyright page", "modality": "text"},
    {"q": "a diagram of the skill tree", "modality": "text"},
    {"q": "the final boss's design sheet", "modality": "text"},
    {"q": "screenshots showing the inventory UI", "modality": "text"},
    {"q": "the foreword by the creative director", "modality": "text"},
    {"q": "a boss named Aldric the Undying", "modality": "image"},
    {"q": "the overworld region known as Emberfall", "modality": "image"},
]


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


@contextmanager
def recorded(store: Store, verb: str) -> Iterator[int]:
    """Record a run row around a verb, marking failure with the exception."""
    run_id = store.start_run(verb)
    try:
        yield run_id
        store.finish_run(run_id, "ok")
    except Exception as exc:  # noqa: BLE001 - recorded then re-raised
        store.finish_run(run_id, "error", str(exc))
        raise


def open_store(config: Config) -> Store:
    return Store(config.db_path)


def _work_source(work: manifest_mod.Work) -> manifest_mod.SourceFile:
    # A work has exactly one format (guarded in manifest.py); use its file.
    return work.files[0]


def _write_blob(config: Config, raw: images_mod.RawImage) -> str:
    path = config.image_path(raw.sha256, raw.ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw.data)
    return str(path)


# --------------------------------------------------------------------------- #
# verbs
# --------------------------------------------------------------------------- #


def cmd_manifest(config: Config, args: argparse.Namespace) -> int:
    works = manifest_mod.build(config.ingest_dir)
    if args.json:
        print(json.dumps([
            {"key": w.key, "format": w.format, "files": [f.relpath for f in w.files]}
            for w in works
        ], indent=2))
    else:
        for w in works:
            print(f"{w.format:4} {w.key}  ({len(w.files)} file(s))")
        print(f"{len(works)} work(s) under {config.ingest_dir}")
    if args.commit:
        with open_store(config) as store, recorded(store, "manifest"):
            for w in works:
                sf = _work_source(w)
                store.upsert_book(w.key, w.format, sf.relpath, sf.sha256, title=w.stem)
    return 0


def cmd_extract(config: Config, args: argparse.Namespace) -> int:
    works = manifest_mod.build(config.ingest_dir)
    if args.limit:
        works = works[: args.limit]
    with open_store(config) as store, recorded(store, "extract"):
        for w in works:
            sf = _work_source(w)
            book_id = store.upsert_book(
                w.key, w.format, sf.relpath, sf.sha256, title=w.stem
            )
            if store.book_extraction_current(w.key, sf.sha256):
                print(f"skipped {w.key} (unchanged, extractor {schema.EXTRACTOR_VERSION})")
                continue
            pages = extract_mod.extract(sf.path, w.format)
            for p in pages:
                store.upsert_page(
                    book_id, p.page_no, p.text, sf.sha256, scanned=p.scanned
                )
            # Stamp only after every page landed: a crash mid-book re-runs it.
            store.mark_book_extracted(book_id, sf.sha256)
            print(f"extracted {len(pages)} page(s) from {w.key}")
    return 0


def _book_id_filter(value: object) -> set[int] | None:
    """Parse a ``--books 39,40`` argument into a set of book ids (or None)."""
    if not value:
        return None
    return {int(x) for x in str(value).split(",") if x.strip()}


def cmd_images(config: Config, args: argparse.Namespace) -> int:
    works = manifest_mod.build(config.ingest_dir)
    if args.limit:
        works = works[: args.limit]
    only = _book_id_filter(getattr(args, "books", None))
    with open_store(config) as store, recorded(store, "images"):
        if only is not None:
            qmarks = ",".join("?" * len(only))
            keys = {
                r["work_key"]
                for r in store.conn.execute(
                    f"SELECT work_key FROM book WHERE id IN ({qmarks})", tuple(only)
                )
            }
            works = [w for w in works if w.key in keys]
        total = 0
        for w in works:
            sf = _work_source(w)
            book_id = store.upsert_book(w.key, w.format, sf.relpath, sf.sha256, title=w.stem)
            if store.book_images_current(w.key, sf.sha256):
                print(f"{w.key}: skipped (unchanged)")
                continue
            raw = images_mod.extract_embedded(sf.path, w.format)
            if w.format == "pdf":
                pages = extract_mod.pdf_pages(sf.path)
                scanned = [p.page_no for p in pages if p.scanned]
                if scanned:
                    raw += images_mod.render_scanned_pages(sf.path, scanned)
            deduped = images_mod.dedupe(raw)
            for d in deduped:
                img = d.image
                blob_path = _write_blob(config, img)
                thumb = images_mod.thumbnail(img.data)
                store.upsert_image(
                    img.sha256, blob_path, img.mime, img.size, img.kind, sf.sha256,
                    phash=d.phash, w=img.w, h=img.h, book_id=book_id,
                    page_no=img.page_no, caption=img.caption, thumb=thumb,
                )
            store.mark_book_images(book_id, sf.sha256)
            total += len(deduped)
            print(f"{w.key}: {len(raw)} raw -> {len(deduped)} unique image(s)")
        print(f"{total} image(s) stored")
    return 0


def cmd_chunk(config: Config, args: argparse.Namespace) -> int:
    with open_store(config) as store, recorded(store, "chunk"):
        rows = store.conn.execute(
            "SELECT id, text FROM document_page ORDER BY id"
        ).fetchall()
        n = 0
        for page in rows:
            existing = store.conn.execute(
                "SELECT COUNT(*) FROM chunk WHERE page_id = ?", (page["id"],)
            ).fetchone()[0]
            if existing and not args.force:
                continue
            ranges = chunk_mod.chunk_text(page["text"])
            store.set_chunks(
                page["id"],
                [(c.ordinal, c.start, c.end, page["text"][c.start : c.end]) for c in ranges],
            )
            n += len(ranges)
        print(f"wrote {n} chunk(s)")
    return 0


def cmd_concepts(config: Config, args: argparse.Namespace) -> int:
    # LLM extraction (typed concepts + descriptions) when a model is named;
    # otherwise the deterministic capitalised-phrase fallback.  The LLM pass is
    # what makes the §6.2.2 dedup type gate meaningful (§12.14: gate to
    # DeepSeek off-peak downstream, in dagu).
    model = args.llm_model or os.environ.get("READER_CONCEPT_MODEL")
    only = None
    if getattr(args, "books", None):
        only = {int(x) for x in str(args.books).split(",") if x.strip()}
    with open_store(config) as store, recorded(store, "concepts"):
        books = store.conn.execute(
            "SELECT id, work_key, source_sha256 FROM book ORDER BY id"
        ).fetchall()
        if only is not None:
            books = [b for b in books if b["id"] in only]
        created = 0
        for b in books:
            if not args.force and store.book_concepts_current(b["work_key"], b["source_sha256"]):
                print(f"{b['work_key']}: skipped (concepts current)")
                continue
            pages = store.conn.execute(
                "SELECT id, page_no, text FROM document_page WHERE book_id = ? ORDER BY page_no",
                (b["id"],),
            ).fetchall()
            full_text = "\n".join(p["text"] for p in pages)
            llm = None
            if model:
                known = [
                    r["name"]
                    for r in store.conn.execute("SELECT DISTINCT name FROM concept ORDER BY name")
                ]
                llm = concepts_mod.gateway_concept_llm(
                    config.litellm_base_url, config.litellm_master_key, model, known=known
                )
            candidates = concepts_mod.extract_book_concepts(
                full_text, llm=llm, book_id=b["id"], limit=args.limit or 40
            )
            for c in candidates:
                cid = store.ensure_concept(c.name, c.type, c.description or None)
                for surface, start, end in concepts_mod.surface_pass(c, full_text):
                    sid = store.add_surface(cid, surface)
                    # Re-anchor the mention to the page that contains the offset.
                    page = _page_for_offset(pages, start)
                    if page is not None:
                        store.add_mention(cid, sid, page["id"], start, end)
                created += 1
            # Stamp only after the book's mentions are written.
            store.mark_book_concepts(b["id"], b["source_sha256"])
            print(f"{b['work_key']}: {len(candidates)} concept(s)")
        if args.rebuild_derived:
            store.rebuild_derived()
        print(f"{created} concept mention group(s) processed")
    return 0


def _page_for_offset(pages: Sequence[object], offset: int):
    """Map a whole-book character offset back to the page containing it."""
    cursor = 0
    for p in pages:
        nxt = cursor + len(p["text"]) + 1  # +1 for the joining newline
        if offset < nxt:
            return p
        cursor = nxt
    return pages[-1] if pages else None


def cmd_embed(config: Config, args: argparse.Namespace) -> int:
    model = args.model or DEFAULT_MODEL
    with open_store(config) as store, recorded(store, "embed") as run_id:
        cfg = store.get_embedding_config()
        dimensions = args.dimensions or (cfg["embedding_dimensions"] if cfg else None)
        if dimensions is None:
            raise SystemExit(
                "embedding dimensionality is not pinned; run `reader eval --pin N` first"
            )
        client = EmbedClient(config.litellm_base_url, config.litellm_master_key, model, dimensions)
        idx = index_mod.create_index(
            config.index_backend,
            dimensions=dimensions,
            api_key=config.turbopuffer_api_key,
            index_dir=config.index_dir,
        )
        def publish(rows: list[dict], what: str) -> int:
            """Upsert rows to the index *before* provenance is stamped.

            Stamping first and upserting later (the old order) means a failure
            mid-run leaves the vector store stale while SQLite already claims
            the rows are embedded, so the next run skips them and the index
            never catches up (reader-artbooks-ingest, 2026-10-13).
            """
            if not rows:
                return 0
            if args.dry_run:
                print(f"[dry-run] would upsert {len(rows)} {what} row(s)")
                return len(rows)
            idx.upsert(rows)
            print(f"upserted {len(rows)} {what} row(s) to {config.index_backend} index")
            return len(rows)

        seen = 0

        chunks = store.chunks_to_embed(model)
        concepts = store.concepts_to_embed(model)
        images = store.images_to_embed(model)
        pending = len(chunks) + len(concepts) + len(images)
        if pending == 0:
            print("nothing to embed")
            return 0
        if args.dry_run:
            # A dry run must not call the paid API: report the queue and stop.
            print(
                f"[dry-run] {pending} row(s) pending "
                f"({len(chunks)} text, {len(concepts)} concept, {len(images)} image); "
                "no API calls made"
            )
            return 0

        slice_size = max(1, args.batch_items)
        spent = 0.0

        def measure(kind: str, part: Sequence[object], payload: Sequence[object]):
            if kind == "image":
                tok = 0
                for im in part:
                    w, h = im["w"], im["h"]
                    if not w or not h:
                        continue
                    ww, hh = w, h
                    if ww * hh > schema.IMAGE_MAX_PIXELS:
                        scale = (schema.IMAGE_MAX_PIXELS / (ww * hh)) ** 0.5
                        hh = max(1, int(hh * scale))
                        ww = max(1, round(ww * hh / h))
                    tok += math.ceil(ww / 32) * math.ceil(hh / 32) + 2
                return tok, tok / 1e6 * schema.COHERE_IMAGE_USD_PER_MTOK, False
            tok = sum(len(t) for t in payload) // 4
            return tok, tok / 1e6 * schema.COHERE_TEXT_USD_PER_MTOK, False

        def run_kind(kind, items, payload_fn, rows_fn, stamp_fn):
            """Embed one slice at a time: a crash loses at most batch_items."""
            nonlocal seen, spent
            for i in range(0, len(items), slice_size):
                if args.max_usd is not None and spent >= args.max_usd:
                    raise SystemExit(
                        f"embedding budget exhausted: ${spent:.4f} >= ${args.max_usd:.2f}"
                    )
                part = items[i : i + slice_size]
                payload = payload_fn(part)
                # Cost is measured *before* anything is published or stamped: if
                # this raises, no row can be left embedded-but-unrecorded.
                tokens, usd, measured = measure(kind, part, payload)
                res = (client.embed_images if kind == "image" else client.embed_texts)(payload)
                rows = rows_fn(part, res.vectors)
                seen += publish(rows, kind)
                for obj in part:
                    stamp_fn(obj, res.model, res.dimensions,
                             res.provenance()["embed_config_version"])
                usage = res.usage or {}
                billed = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
                if billed and kind != "image":
                    tokens, measured = billed, True
                    usd = billed / 1e6 * schema.COHERE_TEXT_USD_PER_MTOK
                spent += usd
                store.record_spend(run_id, kind, res.model, len(part), tokens, usd, measured)
                print(f"  {kind}: {len(part)} row(s), {tokens} tok, ${usd:.4f}"
                      + ("" if measured else " (estimated)"))

        run_kind(
            "text", chunks,
            lambda part: [c["text"] for c in part],
            lambda part, vecs: [
                index_mod.text_row(c["id"], v, c["text"],
                                   book_id=c["book_id"], page_id=c["page_id"])
                for c, v in zip(part, vecs, strict=True)
            ],
            lambda c, m, d, cv: store.stamp_chunk_embedding(c["id"], m, d, cv),
        )

        def concept_text(c) -> str:
            # §6.2.2: embed canonical_name + description, not the bare name.
            return f"{c['name']}. {c['description']}" if c["description"] else c["name"]

        run_kind(
            "concept", concepts,
            lambda part: [concept_text(c) for c in part],
            lambda part, vecs: [
                index_mod.concept_row(c["id"], v, concept_text(c))
                for c, v in zip(part, vecs, strict=True)
            ],
            lambda c, m, d, cv: store.stamp_concept_embedding(c["id"], m, d, cv),
        )

        run_kind(
            "image", images,
            lambda part: [(im["mime"], Path(im["path"]).read_bytes()) for im in part],
            lambda part, vecs: [
                index_mod.image_row(im["id"], v,
                                    book_id=im["book_id"], page_id=im["page_id"])
                for im, v in zip(part, vecs, strict=True)
            ],
            lambda im, m, d, cv: store.stamp_image_embedding(im["id"], m, d, cv),
        )
    return 0


def cmd_reembed(config: Config, args: argparse.Namespace) -> int:
    # Re-embedding is just `embed` under a new model; provenance columns make
    # the work queue scope itself to the new model.
    return cmd_embed(config, args)


def cmd_index(config: Config, args: argparse.Namespace) -> int:
    """Index maintenance (the local backend's housekeeping verb)."""
    with open_store(config) as store, recorded(store, "index"):
        cfg = store.get_embedding_config()
        dimensions = cfg["embedding_dimensions"] if cfg else None
        idx = index_mod.create_index(
            config.index_backend,
            dimensions=dimensions,
            api_key=config.turbopuffer_api_key,
            index_dir=config.index_dir,
        )
        if isinstance(idx, index_mod.LanceIndex):
            if args.optimize:
                idx.optimize(args.cleanup_older_than_days)
            print(f"backend={config.index_backend} table={idx.table_name} rows={idx.count()}")
        else:
            print(f"backend={config.index_backend} (no local maintenance)")
    return 0


def _index_id(value: object, default_kind: str = "text") -> str:
    """Normalise an eval label to an index row id.

    The index keys rows ``<kind>:<record id>`` (``index.text_id`` etc.).  A
    label may be given already-qualified (``"concept:3"``) or as a bare record
    id (``3``), which is read as a ``text`` chunk id.
    """
    s = str(value)
    return s if ":" in s else f"{default_kind}:{s}"


def cmd_eval(config: Config, args: argparse.Namespace) -> int:
    if args.pin:
        with open_store(config) as store:
            store.pin_embedding_config(args.pin, args.model or DEFAULT_MODEL)
        print(f"pinned embedding dimensionality {args.pin} with {args.model or DEFAULT_MODEL}")
        return 0

    labels_path = Path(args.labels) if args.labels else None
    if labels_path is None or not labels_path.exists():
        print(
            "eval needs a labelled set: pass --labels queries.json "
            "(a JSON list of {query, relevant:[row ids]}). "
            "Use `eval --pin N` to commit the dimensionality.",
            file=sys.stderr,
        )
        return 2

    labels = json.loads(labels_path.read_text())
    models = args.models.split(",") if args.models else list(EVAL_MODELS)
    if len(models) > 1:
        # The index has no model column, so every model is scored against the
        # corpus vectors left by whichever model last ran `embed`; a multi-model
        # eval therefore compares QUERY embeddings only (2026-10-12).
        print(
            "note: the index stores no model column, so a multi-model eval varies only the "
            "query embedding; the corpus vectors are whichever model last ran `embed`. "
            "To compare models against their own corpus, re-embed per model first.",
            file=sys.stderr,
        )
    with open_store(config) as store:
        cfg = store.get_embedding_config()
        dimensions = cfg["embedding_dimensions"] if cfg else None
        idx = index_mod.create_index(
            config.index_backend,
            dimensions=dimensions,
            api_key=config.turbopuffer_api_key,
            index_dir=config.index_dir,
        )
        for model in models:
            try:
                client = EmbedClient(
                    config.litellm_base_url, config.litellm_master_key, model, dimensions
                )
                recalls = []
                for item in labels:
                    res = client.embed_texts([item["query"]])
                    # Index row ids are ``<kind>:<record id>`` (e.g. ``text:12``,
                    # ``concept:3``); a bare label is read as a ``text`` chunk id.
                    # Retrieve within the kinds the labels name, so an answer of
                    # a different kind cannot crowd the ranking (§4, honest kind
                    # filter) — a text-chunk label set eval'ed over ``text`` only.
                    relevant = {_index_id(r) for r in item.get("relevant", [])}
                    kinds = sorted({r.split(":", 1)[0] for r in relevant}) or None
                    got = idx.hybrid_query(
                        vector=res.vectors[0],
                        bm25_text=item["query"],
                        top_k=10,
                        kinds=kinds,
                    )
                    ids = {str(r["id"]) for r in (got or [])}
                    recalls.append(len(ids & relevant) / len(relevant) if relevant else 0.0)
            except Exception as exc:  # noqa: BLE001 - skip models the gateway lacks
                print(f"{model}: SKIPPED ({exc})", file=sys.stderr)
                continue
            mean = sum(recalls) / len(recalls) if recalls else 0.0
            print(f"{model}: recall@10 = {mean:.3f} over {len(labels)} queries")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reader", description="vg-books ingestion")
    sub = p.add_subparsers(dest="verb", required=True)

    m = sub.add_parser("manifest", help="walk + group the staging tree into works")
    m.add_argument("--json", action="store_true")
    m.add_argument("--commit", action="store_true", help="record books in SQLite")
    m.set_defaults(func=cmd_manifest)

    e = sub.add_parser("extract", help="extract page text")
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(func=cmd_extract)

    i = sub.add_parser("images", help="extract / render / dedupe images")
    i.add_argument("--limit", type=int, default=0)
    i.add_argument(
        "--books",
        default=None,
        help="comma-separated book ids to restrict the pass (default: all books)",
    )
    i.set_defaults(func=cmd_images)

    c = sub.add_parser("chunk", help="cut page text into offset ranges")
    c.add_argument("--force", action="store_true")
    c.set_defaults(func=cmd_chunk)

    k = sub.add_parser("concepts", help="per-book + surface + merge passes")
    k.add_argument("--limit", type=int, default=0)
    k.add_argument(
        "--books",
        default=None,
        help="comma-separated book ids to restrict the pass (default: all books)",
    )
    k.add_argument(
        "--llm-model",
        default=None,
        help="gateway chat model for extraction (default $READER_CONCEPT_MODEL; "
        "absent -> deterministic heuristic)",
    )
    k.add_argument("--force", action="store_true", help="re-extract even if current")
    k.add_argument("--rebuild-derived", action="store_true")
    k.set_defaults(func=cmd_concepts)

    em = sub.add_parser("embed", help="embed text/images/concepts via the gateway")
    em.add_argument("--model", default=None)
    em.add_argument("--dimensions", type=int, default=None)
    em.add_argument("--dry-run", action="store_true", help="report the queue; no API calls")
    em.add_argument("--batch-items", type=int, default=256,
                    help="rows per checkpointed slice (default 256)")
    em.add_argument("--max-usd", type=float, default=None,
                    help="abort before a slice that would exceed this run's spend")
    em.set_defaults(func=cmd_embed)

    ev = sub.add_parser("eval", help="recall@10 across models; pin the dimensionality")
    ev.add_argument("--models", default=None, help="comma-separated model ids")
    ev.add_argument("--labels", default=None, help="JSON labelled query set")
    ev.add_argument("--pin", type=int, default=None, help="commit dimensionality and exit")
    ev.add_argument("--model", default=None)
    ev.set_defaults(func=cmd_eval)

    re = sub.add_parser("reembed", help="re-embed everything under a new model")
    re.add_argument("--model", required=True)
    re.add_argument("--dimensions", type=int, default=None)
    re.add_argument("--dry-run", action="store_true", help="report the queue; no API calls")
    re.add_argument("--batch-items", type=int, default=256)
    re.add_argument("--max-usd", type=float, default=None)
    re.set_defaults(func=cmd_reembed)

    ix = sub.add_parser("index", help="index maintenance: build, optimize, stats")
    ix.add_argument("--optimize", action="store_true", help="compact versions + rebuild indices")
    ix.add_argument("--cleanup-older-than-days", type=int, default=None)
    ix.set_defaults(func=cmd_index)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config.from_env()
    try:
        return int(args.func(config, args))
    except manifest_mod.ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
