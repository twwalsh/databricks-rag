"""Names shared by every script. One place, so a rename is one edit."""
import os

# Local runs use the CLI profile; inside a Databricks job the runtime supplies auth, so no profile.
PROFILE = None if "DATABRICKS_RUNTIME_VERSION" in os.environ else os.environ.get("DATABRICKS_CONFIG_PROFILE", "free")

CATALOG = "workspace"
SCHEMA = "rag"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.chunks"
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/corpus"

ENDPOINT = "fr-support"
INDEX = f"{CATALOG}.{SCHEMA}.chunks_index"

# Databricks-hosted, pay-per-token, no keys. gte-large reads 8k tokens, so no chunk is truncated
# (the Cloudflare build's bge-base stops at 512 tokens and the chunker caps text for it).
EMBED_MODEL = "databricks-gte-large-en"
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "databricks-meta-llama-3-3-70b-instruct")

EXPERIMENT = "/Users/tw@twwalsh.com/fr-support-rag"
