"""Build (or refresh) the Vector Search index over workspace.rag.chunks.

    uv run python src/build_index.py            # create endpoint + index if missing, sync, wait

The source table is built by sql/chunks.sql from chunks.jsonl in the corpus volume. The index is a
Delta Sync index with Databricks-managed embeddings (databricks-gte-large-en, 1024-dim, 8k-token
window), so re-running the SQL and this script is the whole re-index: no embedding code here, which is
the first difference from the Cloudflare build (scripts/support-index.mjs there embeds each chunk itself
with Workers AI bge-base and upserts vectors). TRIGGERED sync: we call it, it does not watch the table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports when run as a job task
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

from config import EMBED_MODEL, ENDPOINT, INDEX, PROFILE, SOURCE_TABLE


def main() -> int:
    w = WorkspaceClient(profile=PROFILE)

    names = {e.name for e in w.vector_search_endpoints.list_endpoints()}
    if ENDPOINT not in names:
        print(f"creating endpoint {ENDPOINT}")
        w.vector_search_endpoints.create_endpoint(name=ENDPOINT, endpoint_type=EndpointType.STANDARD)
    while w.vector_search_endpoints.get_endpoint(ENDPOINT).endpoint_status.state.value != "ONLINE":
        time.sleep(10)
    print(f"endpoint {ENDPOINT} online")

    existing = {i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=ENDPOINT)}
    if INDEX not in existing:
        print(f"creating index {INDEX} from {SOURCE_TABLE}")
        w.vector_search_indexes.create_index(
            name=INDEX,
            endpoint_name=ENDPOINT,
            primary_key="id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=SOURCE_TABLE,
                pipeline_type=PipelineType.TRIGGERED,
                # `passage` is title + text, the same string the Cloudflare build embeds.
                embedding_source_columns=[EmbeddingSourceColumn(name="passage", embedding_model_endpoint_name=EMBED_MODEL)],
                columns_to_sync=["id", "source", "title", "url", "text"],
            ),
        )
    else:
        st = w.vector_search_indexes.get_index(index_name=INDEX).status
        if st and st.ready:
            print(f"index {INDEX} exists, syncing")
            w.vector_search_indexes.sync_index(index_name=INDEX)
        else:
            print(f"index {INDEX} exists but is not ready yet: {st.message if st else '?'}")

    # Wait for the sync. The first one also provisions the pipeline, so allow a while. The status
    # object has only ready / message / indexed_row_count (SDK 0.40+); the message carries the state.
    for _ in range(90):
        st = w.vector_search_indexes.get_index(index_name=INDEX).status
        ready = bool(st and st.ready)
        print(f"  ready={ready}  rows={st.indexed_row_count if st else None}  {st.message if st else ''}", flush=True)
        if ready:
            return 0
        if st and st.message and "fail" in st.message.lower():
            return 1
        time.sleep(20)
    print("timed out waiting for the index", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
