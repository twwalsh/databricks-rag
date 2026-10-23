-- The source table for the Vector Search index. Re-run after re-uploading chunks.jsonl to the
-- corpus volume (make upload), then `uv run python src/build_index.py` re-syncs the index.
--
-- `passage` = title + text, the exact string the Cloudflare build embeds (title first so a section
-- is findable by its name). Change Data Feed is what lets Delta Sync index only what changed.
CREATE OR REPLACE TABLE workspace.rag.chunks
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS SELECT
  id,
  source,
  title,
  url,
  text,
  concat(title, '\n\n', text) AS passage,
  length(text) AS chars
FROM read_files('/Volumes/workspace/rag/corpus/chunks.jsonl', format => 'json');
