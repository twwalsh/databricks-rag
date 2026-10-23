# The whole pipeline, in order. Every target needs the CLI profile (default: free).
PROFILE ?= free
VOLUME   = dbfs:/Volumes/workspace/rag/corpus
CHUNKER  = ../fruitless-research/site

.PHONY: export upload table index ask eval all

## 1. chunk the public sources with the Cloudflare build's own chunker (same chunks, same allowlist)
export:
	cd $(CHUNKER) && npm run build >/dev/null && node scripts/support-index.mjs --dump $(CURDIR)/data/chunks.jsonl
	cp $(CHUNKER)/support/eval.jsonl data/eval.jsonl

## 2. corpus + eval set into the Unity Catalog volume
upload:
	databricks fs cp data/chunks.jsonl $(VOLUME)/chunks.jsonl --overwrite --profile $(PROFILE)
	databricks fs cp data/eval.jsonl   $(VOLUME)/eval.jsonl   --overwrite --profile $(PROFILE)

## 3. the Delta source table (Change Data Feed on, so Delta Sync indexes only what changed)
table:
	databricks experimental aitools tools query "$$(cat sql/chunks.sql)" --profile $(PROFILE)

## 4. endpoint + Delta Sync index with managed embeddings; waits for the sync
index:
	DATABRICKS_CONFIG_PROFILE=$(PROFILE) uv run python src/build_index.py

## ask one question:  make ask Q="Where do I put the .frlicense file on a Mac?"
ask:
	DATABRICKS_CONFIG_PROFILE=$(PROFILE) uv run python src/ask.py "$(Q)"

## 5. the 30-case eval set, logged to the Databricks MLflow experiment
eval:
	DATABRICKS_CONFIG_PROFILE=$(PROFILE) uv run python src/evaluate.py

all: export upload table index eval
