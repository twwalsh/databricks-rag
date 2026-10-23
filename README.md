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

First full run, 2026-10-23, Llama 3.3 70B as the generator (MLflow run `4e1fd250…` in the workspace
experiment `fr-support-rag`, every answer with its retrieved passages):

| Scorer | Databricks build | Cloudflare build (first run) |
|---|---|---|
| must-say rules met | 29/30 | 29/30 |
| must-not-say rules met | 30/30 | 30/30 |
| right self-service form offered | 30/30 | 30/30 |
| **strict pass (all rules of a case)** | **29/30** | **29/30** |

The Databricks miss: *"Does the bundle include plugins you release later?"* The answer was correct and
cited the right FAQ entry ("The bundle is the catalog on the day you buy it, and a plugin released later is
a separate purchase") but never said **no** to a yes/no question, which the case requires. The
Cloudflare miss on its first run was the test's own regex, fixed in the test. So: the same score, for a
different reason, and the Databricks one is the model's.

Hand checks: the off-topic question ("What is the capital of France?") is refused; licensing questions
pull the FAQ first; the Trammel SLOW-attack question lands on the exact manual section.

## What the comparison taught

- **Grounding is the prompt's job only when there is something to ground on.** Against the still-empty
  index, the hosted model answered a licensing question from general knowledge, invented a "LICENSE FILE
  control in the plugin's header", and offered a form, with the prompt's prohibition in front of it. Both
  builds now refuse to call the model when retrieval returns nothing. Neither had that guard before.
- **Managed embeddings remove a whole class of bugs.** The Cloudflare build embeds at index time and
  query time with the same model by discipline (a comment in two files). The Delta Sync index embeds
  both sides itself, so the mismatch cannot happen; and gte-large's 8k window means no chunk is
  truncated, where bge-base's 512 tokens forced a character cap in the chunker.
- **Hybrid search costs nothing to turn on here** and helps exactly where a support corpus needs it:
  control names in capitals, plugin names, `.frlicense`.
- **The hosted open model is a little too eager with the form.** Llama offered a self-service form on
  questions where none applied (a "complete-bundle" form on a question about an attack time). Claude
  did not. The eval does not penalise a spurious form, which is a gap in the eval, not a pass.
- **Free Edition is a real workspace with a few sharp edges.** One Vector Search endpoint; a new index
  can sit at "pending endpoint provisioning" for 40 minutes and forcing its pipeline early breaks it;
  `mlflow.genai.evaluate` deadlocks against the SDK at its default worker count. All recorded in the
  code comments where they bit.
- **What the Databricks side gives that the Worker does not:** the table, the index, the job run, every
  evaluated answer and the registered model versions are all inspectable in one place, with lineage.
  The Cloudflare build's equivalents are a manifest file, AI Gateway logs and a terminal.

## Model lifecycle

`src/release.py` is the half of the POC the Cloudflare build has no equivalent for:

1. **Version:** every run logs the assistant *from code* (`src/agent.py`, an MLflow `ResponsesAgent`
   around `ask.py`) and registers it as a new version of `workspace.rag.support_assistant` in Unity
   Catalog, with the index and the chat endpoint declared as resources so a serving container gets
   pass-through auth to them.
2. **Evaluate:** the same 30 cases run against the version, linked to it in MLflow, so each version
   carries its own scores.
3. **Promote:** the `champion` alias moves only if strict pass ≥ 80%; otherwise the version stays
   registered, unaliased, with its failing scores attached. The previous champion keeps the alias
   `previous`, so a rollback is one alias call.
4. **Serve:** `--deploy` puts `models:/workspace.rag.support_assistant@champion` behind a Model Serving
   endpoint (`fr-support-assistant`).

## Not built

- **A public front door.** On Free Edition, a Databricks App is reachable only by workspace users, so a
  public demo would be a small Cloudflare Worker calling the Vector Search and Model Serving REST APIs
  with a workspace token. The retrieval and generation would still run here; the page would only be the
  door.
- **Claude as the generator.** The Free Edition hosts open-weight models only. Pointing `ask.py` at the
  Anthropic API would make the generation step identical to the Cloudflare build and isolate the
  retrieval difference; it is a few lines, not done because the point was to use the hosted stack.
