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

## The gateway

Generation does not call the model endpoint directly. `gateway/support_llm.json` defines a **Unity Gateway
model service**, `workspace.rag.support_llm`, and both `ask.py` and the public Worker call it through the
gateway's chat-completions API. The service owns:

- **Routing and fallback:** Llama 3.3 70B primary, Llama 3.1 8B fallback, tried automatically if the
  primary fails. Swapping the generator is an edit to the service, not to the code.
- **Rate limits:** 30 requests / 60k tokens per minute per caller; 600 requests / 1.5M tokens per hour
  for the service as a whole. A limit of zero would shut the demo off without a deploy.
- **An inference table:** every request lands in `workspace.rag.support_llm_payload` with the requester,
  destination model, latency, status and the full request and response, queryable with SQL. Combined
  with the eval traces in MLflow, every answer the demo has ever given is auditable.
- **Access as a Unity Catalog securable:** who may call the generator is a grant on the service, not a
  shared key.

The Cloudflare build uses Cloudflare's AI Gateway for the same reasons (logs, cost view, rate limits), as
a managed service in front of the Anthropic key. The difference here is that the service, its limits and
its log table are objects in the same catalog as the data, and the policy is a file in this repo.

## The same assistant as a LangGraph graph

`src/graph.py` is `ask.py` drawn as a **LangGraph state graph**, so the control flow is explicit and the
one place the two builds differ in mechanism is made real: `show_form` is a bound tool the model calls, as
in the Cloudflare build, rather than a tagged last line.

```
retrieve ──(no passages)──▶ refuse ──▶ END
    └────(passages)───────▶ generate ──(tool call)──▶ tools ──▶ generate ──▶ END
                                └──────(text)──────────────────────────────▶ END
```

The model is reached through the gateway with LangChain's OpenAI client (the gateway speaks the OpenAI
chat-completions API, so nothing in the graph knows which model is behind it). Two things the loop does
that a straight line cannot:

- **The tools node is a policy gate, not a passthrough.** A `show_form` call is honoured only when the
  question is shaped for it (a license or bundle form on a licensing-shaped question, a download form on a
  download-shaped one); otherwise the tool result says "no form applies" and the model writes text. An
  off-topic question cannot be answered with a license form however the model feels about it.
- **Llama 3.3 70B, when it calls a tool, emits the call and no text.** The graph feeds the tool result back
  and generates again, with no tools bound the second time so it writes instead of calling again.

Scored with the same 30 cases (`uv run python src/evaluate.py --graph`):

| Run | strict pass | note |
|---|---|---|
| graph, first run | 27/30 | the policy gate was licensing-only and threw away a correct `resend-download` call on "I lost my download link for the drum pack"; two second-pass answers lost the thread |
| graph, gate widened | **29/30** | same score as the straight-line path and the Cloudflare build |

The remaining miss is instructive: *"Where do the license files go on Windows?"* The model called
`show_form` first, and in the second pass, with the tool result in front of it, wrote "the passages do not
answer the question" when passage 2 held the `%APPDATA%` path. The straight-line path answers it. The
tool-call loop costs this model some attention; the form it gains is worth it on the licensing questions
and not on this one. The point of running both is to know that, with numbers.

## Not built

- **A public front door.** On Free Edition, a Databricks App is reachable only by workspace users, so a
  public demo would be a small Cloudflare Worker calling the Vector Search and Model Serving REST APIs
  with a workspace token. The retrieval and generation would still run here; the page would only be the
  door.
- **Claude as the generator.** The Free Edition's pay-per-token endpoints are open-weight models only;
  Claude is registered in `system.ai` but not served here. Pointing the gateway's primary destination at
  it on a paid workspace would make generation identical to the Cloudflare build and isolate the
  retrieval difference, with no code change.
- **A spend budget on gateway usage.** An account-level budget alert exists for Unity Gateway spend; it is
  an account-admin operation and was not set up on this Free Edition account.
