# fr-support-rag: one RAG system, built twice

The support assistant on [fruitlessresearch.com](https://fruitlessresearch.com) ("Ask a question" on any
page) is a retrieval-augmented system on Cloudflare: Workers AI embeddings, a Vectorize index, Claude
through AI Gateway. This repo is the **same system rebuilt on Databricks**: the same 219 chunks, the same
system prompt, the same 30-question evaluation set, so the two platforms can be compared on the one
question that matters for a support bot, *does it answer correctly from the documents and refuse
everything else*.

It is a proof of concept on a Databricks Free Edition workspace, built in a day. It is not the production
assistant; that stays on Cloudflare, where the site runs.

## What runs where

| Step | Cloudflare build (`fruitless-research/site`) | Databricks build (this repo) |
|---|---|---|
| Corpus | manual, built site pages, changelog, licensing FAQ, chunked per section by `scripts/support-index.mjs` from an **allowlist** | the same chunks, exported by that script's `--dump` flag, as `data/chunks.jsonl` |
| Storage | the chunk text travels inside the vector metadata | a Unity Catalog volume (`/Volumes/workspace/rag/corpus`) and a Delta table `workspace.rag.chunks` with Change Data Feed |
| Embeddings | Workers AI `bge-base-en-v1.5`, 768-dim, 512-token window; the script embeds and upserts | **Databricks-managed**: `databricks-gte-large-en`, 1024-dim, 8k-token window; the Delta Sync index embeds the `passage` column itself |
| Index | Vectorize, cosine, vector-only search | Vector Search Delta Sync index on a Standard endpoint, **hybrid** search (vector + BM25) |
| Licensing pin | a licensing-shaped question also pulls the top 3 FAQ entries | identical, via a `source = faq` filter |
| Generation | Claude Haiku 5.5 via AI Gateway, streamed; `show_form` is a tool call | a Databricks-hosted model via Model Serving (Llama 3.3 70B by default, `RAG_CHAT_MODEL` to change); `show_form` is a tagged last line, so any hosted model can use it |
| Evaluation | `support-eval.mjs` against a local `wrangler dev`: regex rules per case | `src/evaluate.py`: the same rules as MLflow scorers, logged to an MLflow experiment with every answer and its retrieved passages |
| Deployment | `wrangler deploy` of the site Worker | a Declarative Automation Bundle (`databricks.yml`): one serverless job, SQL task + Python task |
| Public front door | the site widget: Turnstile, sessions, rate limits, streaming | none yet (see below) |

## Layout

```
data/chunks.jsonl        the corpus, exported from the site's chunker (make export)
data/eval.jsonl          the 30 evaluation cases, copied verbatim
sql/chunks.sql           the source Delta table, from the volume
src/config.py            names: catalog, schema, endpoint, index, models
src/build_index.py       endpoint + Delta Sync index, sync, wait
src/ask.py               retrieve (hybrid + licensing pin) then answer from the passages only
src/evaluate.py          the eval set as MLflow GenAI scorers
databricks.yml           the bundle; resources/rag.job.yml is the job
Makefile                 the same steps by hand: export, upload, table, index, ask, eval
```

## Run it

Needs the Databricks CLI logged into a workspace as profile `free` (or `PROFILE=<name>`), `uv`, and the
sibling `fruitless-research` repo for `make export`.

```bash
make export upload table index      # corpus → volume → Delta table → Vector Search index
make ask Q="Where do I put the .frlicense file on a Mac?"
make eval                           # 30 cases, logged to MLflow
databricks bundle deploy -t dev && databricks bundle run rag_index -t dev   # the same as a job
```

## Results

_Filled in from the first evaluation run; see below._

## What the comparison taught

_Filled in after the first evaluation run._

## Not built

- **A public front door.** On Free Edition, a Databricks App is reachable only by workspace users, so a
  public demo would be a small Cloudflare Worker calling the Vector Search and Model Serving REST APIs
  with a workspace token. The retrieval and generation would still run here; the page would only be the
  door.
- **Claude as the generator.** The Free Edition hosts open-weight models only. Pointing `ask.py` at the
  Anthropic API would make the generation step identical to the Cloudflare build and isolate the
  retrieval difference; it is a few lines, not done because the point was to use the hosted stack.
