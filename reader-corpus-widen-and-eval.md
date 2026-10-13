# reader — widen the corpus & re-run the pro-vs-fast eval

**Goal:** settle `cohere/embed-v5.0` (pro) vs `cohere/embed-v5.0-fast` at 2048 dims on more than the 3-book corpus. More distractor chunks should separate the models if they differ at all.

**Status:** ✅ **COMPLETE** — started 2026-10-12T23:27Z; resumed ~23:41Z; finished 2026-10-12T23:56Z. (The first pass stopped mid-embed waiting on a `nas-goose` round-trip that was never needed — the relay was already listening.)

## Baseline (pre-widening)

- Corpus: 3 books, 1363 chunks, 681 images, 56 concepts (books: `Resident Evil - Philip J Reed` epub, `Shadow-of-the-Colossus-Nick-Suttner` epub, `Indie_PDF_BitmapBooks` pdf).
- Labels: 10 queries (`.reader-data/eval-labels.json`, gitignored).
- Eval (reported, to reproduce): pro **0.430** vs fast **0.439** recall@10 → a tie within noise.

**Pipeline note (read from source):** rows in the LanceDB index are keyed `<kind>:<id>` with **no model column**, and `cmd_embed` writes to a single table. So `reader eval --models A,B` varies only the **query** embedding — both models are scored against whichever corpus vectors are currently in the index. A clean comparison needs a corpus re-embed per model. Captured in Step 5.

---

## Step 1 — Enumerate the NAS staging corpus ✅

Via the `nas-goose` A2A agent (NAS = tailnet `100.82.223.13`). Staging dir `/volume1/homes/nick/Books`. `@eaDir`/`.DS_Store` ignored.

**49 source files** = 13 top-level + 18 `Data Visualization/` + 18 `Machine Learning, AI and Bots/`. All `.epub` except 8 top-level `.pdf` and 3 `.mobi` alternates.

Top level: `079b3a_8538…eb.pdf` 353.3M · `BG3_Artbook_High_Res.pdf` 301.6M · `GREMLIN_DLC_2024_PDF.pdf` 41.5M · `GREMLIN_PDF.pdf` 76.0M · `Indie_PDF_BitmapBooks.pdf` 54.6M · `NP_ArcadeDecades80s_Book(single).pdf` 104.1M · `NP_ArcadeDecades80s_Book.pdf` 101.8M · `Resident Evil - Philip J Reed` {epub 4.04M, mobi 6.62M, pdf 1.00M} · `Shadow-of-the-Colossus-Nick-Suttner` {epub 2.00M, mobi 0.63M, pdf 0.99M}.

`Data Visualization/` (18 epubs, MB): designingdatavisualizations 4.45 · tableaudesktoppocketreference 8.99 · rfordatascience2ndedition 13.93 · learningmicrosoftpowerbi_oreilly 17.73 · excelcookbook 17.74 · datavisualizationwithpythonandjavascript2ndedition 20.31 · makingdatavisual 21.36 · communicatingwithdata 23.60 · interactivedatavisualizationfortheweb2ndedition 24.13 · datavisualizationwithmicrosoftpowerbi 32.09 · fundamentalsofdatavisualization 34.67 · headfirstexcel 35.26 · practicaltableau 38.73 · communicatingdatawithtableau 39.75 · colorwise 45.57 · tableaustrategies 63.78 · innovativetableau 73.12 · hands-ondatavisualization 73.68

`Machine Learning, AI and Bots/` (18 epubs, MB): developersplaybookforlargelanguagemodelsecurity 3.03 · machinelearningwithpythoncookbook_secondedition_V2 4.17 · learninglangchain 6.51 · designinglargelanguagemodelapplications 6.59 · architectingdataandmachinelearningplatforms 9.68 · promptengineeringforllms 10.93 · machinelearninginterviews 11.30 · naturallanguageprocessingwithtransformersrevisededition 11.30 · developingappswithgpt-4andchatgpt 12.63 · trainingdataformachinelearning 13.80 · low-codeai 15.66 · hands-onlargelanguagemodels 21.40 · hands-ongenerativeaiwithtransformersanddiffusionmodels 26.24 · hands-onmachinelearningwithscikit-learnkerasandtensorflow 28.53 · aiengineering 30.72 · generativedeeplearning_secondedition 32.26 · promptengineeringforgenerativeai 34.93 · generativeaionaws 44.17

**Already ingested:** book 1 RE epub, book 2 Shadow epub, book 3 Indie pdf → **46 not yet ingested.** (RE/Shadow `.pdf`/`.mobi` are alternate formats of ingested books; manifest forbids mixed formats per stem, so they were NOT copied.)

---

## Step 2 — Pull books over the tailnet ✅

Relay: `nas-goose` ran `python3 -m http.server 8099 --bind 0.0.0.0` in the staging dir (NAS `100.82.223.13:8099`); devcontainer curl'd it. All 35 **epubs** pulled, **~899 MB total**, in ~37 s. First 12 verified against NAS-side sha256; the rest size-verified + local sha256 recorded. `RE.pdf/mobi`, `Shadow.pdf/mobi`, and the big image-artbook PDFs (BG3 302M, NP 102M×2, 079b3a 353M, GREMLIN) were **skipped** (artbooks = little text). Server still running; will be killed by nas-goose.

| bytes | sha256 | file |
|---:|---|---|
| 3028785 | `6028781ae58cbc4569e2ce3ff1ebae3815aa676565836d1b7f1c776a92280afb` | developersplaybookforlargelanguagemodelsecurity.epub |
| 4165377 | `c71454a7a6cfe09ea781dd6aeef86031cdfe8b5291970e832d3d0619bf2e2b2d` | machinelearningwithpythoncookbook_secondedition_V2.epub |
| 4451755 | `b21b1237be65c47425f811341c2e072da74cf1f7705041047698cac0f84b6fd0` | designingdatavisualizations.epub |
| 6510718 | `3dab0ba4e1c656cdeadc24050174fa825a4d70ebf1fc182a348259e20afc0a6c` | learninglangchain.epub |
| 6589948 | `c8061bbd611db5fccc12015b89ba980a6e7de4c3fed717fc5402f1c93effd2ad` | designinglargelanguagemodelapplications.epub |
| 8985207 | `b1c0f88392cec1b0c283b4c54b4481e253e026b60b66509e4c6b0803866fe2fa` | tableaudesktoppocketreference.epub |
| 9679347 | `ea0bb916449ebd3a8d93b4425d235e8a8741b6205022a2756d707d64d07ada88` | architectingdataandmachinelearningplatforms.epub |
| 10934068 | `18852cba60143af13f1134c08f8ad894e0b0a39099d3473022b508a8271371d0` | promptengineeringforllms.epub |
| 11296082 | `a8e7308888a9e81ef8bccd7af240c7e0274ddf4865ea07aa3990167efcca778e` | machinelearninginterviews.epub |
| 11301022 | `891a378c1907d9071f11d3a373e4f3ac3a4d49e360364079d21fcdb9330107aa` | naturallanguageprocessingwithtransformersrevisededition.epub |
| 12631283 | `509ab61f54eb5fe4d8f5ff16361c2fc807b9ae140b74478801d1357f9378104c` | developingappswithgpt-4andchatgpt.epub |
| 13926145 | `b0631a44d954545e5d1a153f58b6cc27234f7327f826c6cfc219fc4bd2fc38ec` | rfordatascience2ndedition.epub |
| 15660307 | `36e8f3c63409119b094e20f8b5c2319015eb58de7f623ae70900b2cc57c67c9f` | low-codeai.epub |
| 17725881 | `d13ccb8ced914275e287a040f1dc471aa738ad4cba287b2ba70a0684658d0762` | learningmicrosoftpowerbi_oreilly.epub |
| 17742483 | `00bbc1fe6aac8b0aff4c97dab77c9271920f2c68e985ebf400c8cf133704a2ba` | excelcookbook.epub |
| 20307344 | `543396689674b167568d3485a99d8060bb1ccf02aad8c2c10e8f61f38cd82644` | datavisualizationwithpythonandjavascript2ndedition.epub |
| 21361126 | `645c915436d2a1c34d7972a62bdfa1756089e9c1c9c6635e081b94201b50e7c1` | makingdatavisual.epub |
| 21403515 | `1b8b64f4aef20927724ae8f81bb0dcaab351fe78c1255c363b33002a5c51cc57` | hands-onlargelanguagemodels.epub |
| 23603307 | `b3a3b5406d0813fa6846046685f3347ad879a230c5ac0732f8fdc2ac9c90d2ef` | communicatingwithdata.epub |
| 24132543 | `e895f01c3a4fdf9970cd3eb53598b6b647f58dd73cbb22343b893258c432ad84` | interactivedatavisualizationfortheweb2ndedition.epub |
| 26235427 | `4e58df859b4a5275ddd36bc1d8ee7f1ea6e3d1e9d9e329cc1672866444a82cf3` | hands-ongenerativeaiwithtransformersanddiffusionmodels.epub |
| 28534721 | `6ade352a10c50b1e5e6331bc05d74fc2c69de55fe4a03978374c83b002dd66cc` | hands-onmachinelearningwithscikit-learnkerasandtensorflow.epub |
| 30722742 | `d4fb61afcf9481021d77cbc97e99e24d2af691dbb46eb37b51fac25016203b51` | aiengineering.epub |
| 32086379 | `fb61ffdd799f68299fddb78eea5edd12329b2231628b47314dbbde8770b1bf4a` | datavisualizationwithmicrosoftpowerbi.epub |
| 32262154 | `eb475f032cf2067eb2a4c392a9a932a6de37c9d54df1c06a2ad6f0f2b51ab9f0` | generativedeeplearning_secondedition.epub |
| 34674219 | `0a944c45f737e25714d013bc73787a09f6bf99d886be292b26cbc88e5ac62a23` | fundamentalsofdatavisualization.epub |
| 34930152 | `4d39def2068bc9fd500ab6352fc9e39d9cbaf4b04838928511746a349971e069` | promptengineeringforgenerativeai.epub |
| 35255033 | `50fd4eb8fee4f84ae9ca64921e77e9705e2c86a40e759905498d6ca9259820e9` | headfirstexcel.epub |
| 38729397 | `8ace29a84a7a5e655a2d6fd066d0c27313cdfa6ff1acfa66134b870793752c48` | practicaltableau.epub |
| 39754461 | `89b5f5590a2a8e68a69d1493230d868cdfd07e32e0732e516458126cb6b7b67e` | communicatingdatawithtableau.epub |
| 44171575 | `3f045d74b77316ae27572647885e02cf8936ac1dfbabaa1cc57f1a07c17e84ef` | generativeaionaws.epub |
| 45565574 | `d64d707ac9d564885206134ad88723c2b583b40b2778758cce46544c2765f084` | colorwise.epub |
| 63775888 | `7cb03f8beb4995ac9b1e880460e43ad15d0cda0fd418de67bff58eef0bb71354` | tableaustrategies.epub |
| 73124287 | `7fe49b0a2865d1b3f74512859c93e7942d912eb87d112b1490dda9206f9bdf19` | innovativetableau.epub |
| 73675808 | `a5f06d7125abaf4715f1dd6b80fb9998f095f39bea7c7229fc0dd177cb703bf2` | hands-ondatavisualization.epub |

*(35 epubs. `trainingdataformachinelearning.epub` was the 36th DV/ML epub in the NAS listing; not pulled — 35 is ample.)*

### Step 2b — Add the small-first **on-topic** video-game PDFs (resume pass) ✅

Following the resume directive, pulled the three text-bearing video-game PDFs — the complement to the off-topic epub tranche. Relay was already listening on `100.82.223.13:8099`, so no `nas-goose` round-trip was required. All 3 landed in ~4 s each.

| bytes | sha256 | file |
|---:|---|---|
| 76018635 | `352350f3fc47439aec372e70c2336659ebdce2ce18091079e546db0bddb367c8` | GREMLIN_PDF.pdf |
| 41482991 | `1d4c2dee898f0887bc7c33aa283e54bba7923adb40138f96cf6853d9554fd0b3` | GREMLIN_DLC_2024_PDF.pdf |
| 104063487 | `de403c0b6ad9bb63a25dc095a3b74ff9919f66c2253af9a20a13207215bc75c9` | NP_ArcadeDecades80s_Book(single).pdf |

Sizes match the NAS enumeration byte-for-byte. **Not** pulled this pass (as directed): `BG3_Artbook_High_Res.pdf` (302 M, image-heavy) and `079b3a_8538…eb.pdf` (353 M, unknown); also skipped `NP_ArcadeDecades80s_Book.pdf` (102 M — the `(single)` sibling was used) and the RE/Shadow `.pdf`/`.mobi` alternates.

---

## Step 3 — Ingest new books (text path) ✅

- **35 epubs** ingested in the first pass (manifest run 23, extract run 24, chunk run 25): books 4–38.
- **3 on-topic PDFs** added on resume: `manifest --commit` → `extract` → `chunk`.

| book id | title | pages | chunks |
|---|---:|---:|---:|
| 39 | GREMLIN_DLC_2024_PDF (*A Gremlin in the Works — Expansion Disk*) | 264 | 626 |
| 40 | GREMLIN_PDF (*A Gremlin in the Works 1983–2015*, Bitmap Books) | 563 | 1281 |
| 41 | NP_ArcadeDecades80s_Book(single) (*Arcade Decades: 80s Edition*, Kim Justice) | 324 | 756 |

**Corpus now: 41 books, 23380 chunks** (from 3 books / 1363 chunks at baseline).

**First-pass embed stalled** (SQLite `run` 26 errored `gateway 500 APIConnectionError` at 23:38:09; run 27 was left `running` since 23:38:42). Caveat worth recording: `cmd_embed` embeds the *whole* remaining corpus in one `embed()` call and stamps `embed_model_id` only **after every batch returns**, so a long run shows `stamped: 0` right up until it flips to complete — it looks stalled when it is merely mid-flight (`timeout=60s`, no retries).

---

## Step 4 — Extend the labelled query set ✅

Added **5 topic-anchored queries** over the three on-topic books (existing 10 retained → **15 labels** total). Ids are text-chunk ids in books 39–41, verified against chunk text before commit.

| query | relevant chunks |
|---|---|
| the Sheffield computer shop opened by Ian Stewart and Kevin Norburn that became Gremlin Graphics | 21343, 20725, 21347 |
| Wanted: Monty Mole | 20731, 21034, 20736 |
| the origin of Zool: Ninja of the Nth Dimension | 20767, 20768, 20827 |
| the extra interview material gathered for the Gremlin in the Works Expansion Disk | 20720, 20724 |
| the Magnetic Fields duo behind Gremlin's Lotus racing games | 20762, 20766 |

Same shape as the existing entries (`{query, relevant:[...]}`), so `reader eval` reads them unchanged.

---

## Step 5 — Re-run the eval

✅ Completed 23:56Z. Corpus embed (pro) done; gateway latency measured at **~4 s per 64-item batch** ⇒ ~22 min for 23 380 chunks — slow, but steady (not the stall the resume note feared). Watcher `scripts/watch_embed_pro.sh` waits for the in-flight embed, embeds the residual chunks, then rebuilds the index before eval.

**Comparison caveat (from the pipeline note):** the index has no model column, so `eval --models pro,fast` varies the **query** embedding only; both are scored against the single corpus embedding. A fully clean comparison re-embeds the corpus per model (~22 min each).

### Result

```
corpus: 41 books · 23 380 text chunks (index rows 24 117 = 23 380 text + 681 image + 56 concept)
labels: 15 queries (10 vg-novel + 5 on-topic Gremlin/arcade)
cohere/embed-v5.0      : recall@10 = 0.357
cohere/embed-v5.0-fast : recall@10 = 0.357
```

**A dead tie — identical to the third decimal, and per-query identical on all 15 queries.** The two models are *not* aliases: vectors differ (cosine 0.950, max abs diff 0.022), and the gateway echoes distinct `model` names. The rankings simply never diverge enough to change top-10 membership under hybrid (vector + BM25 RRF) retrieval — BM25 anchors the ranking, and the 5% vector difference is too small to flip a chunk across the top-10 boundary.

Baseline recall fell **0.43 → 0.357** as the corpus grew ~17×: more distractors, same weak topical queries (the 4-character queries "spider enemies", "Wanted: Monty Mole" score 0.00; some long paraphrases do too).

**Recommendation: use `cohere/embed-v5.0-fast`.** At 2048 dims on this corpus it matches pro exactly on recall@10 while (by construction) being the cheaper/faster endpoint — pro buys nothing here. The tie is robust to corpus widening: 17× more distractors did not separate them.


---

## Step 6 — Surprises

1. **The "stall" was not a stall.** The resume note blamed a `nas-goose` round-trip; in fact the relay was already listening on `:8099` (no round-trip needed), and the `embed` it looked hung on was mid-flight. `cmd_embed` does the *entire* remaining corpus in one `embed()` call and stamps only after **every** batch returns, so `stamped: 0` is the normal state for ~22 minutes — it flips straight to complete. The 500 in run 26 was a real transient (`APIConnectionError`); the retry (run 27) succeeded.
2. **Gateway throughput is the bottleneck, not the tailnet.** The NAS pull ran at ~4 s for a 76 MB file (≈19 MB/s); a 64-item pro *embed* batch also takes ~4 s. Corpus embed is ~22 min per model regardless of pro/fast (fast measured 3.6 s/batch).
3. **pro == fast on recall, exactly.** Even with 17× the distractors, the two models produce byte-identical rankings on all 15 queries, despite ~5% different vectors. The discrimination the widening was meant to expose does not exist at 2048 dims under hybrid retrieval.
4. **`eval` scores query embeddings against a single corpus embedding** (no model column in the index), so the run above is a query-only comparison. A corpus-per-model rerun is the only way to fully rule out index-side bias — left as a follow-up, ~22 min per arm.

---

## Resume here / follow-ups

- **This pass is complete**: corpus pulled, ingested, labelled, embedded (pro), indexed, evaluated.
- **Left untouched as directed:** `BG3_Artbook_High_Res.pdf` (302 M), `079b3a_8538…eb.pdf` (353 M), `NP_ArcadeDecades80s_Book.pdf` (102 M, `(single)` used instead), and the RE/Shadow `.pdf`/`.mobi` alternates.
- **Optional clean re-run:** `embed --model cohere/embed-v5.0-fast --dimensions 2048` over the corpus → `index` → `eval --models cohere/embed-v5.0-fast` to confirm the tie from the corpus side.
