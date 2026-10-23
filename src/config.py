"""Names shared by every script. One place, so a rename is one edit."""
import os

# Local runs use the CLI profile. Inside a Databricks job or a Model Serving container there is no
# ~/.databrickscfg and the runtime supplies auth, so PROFILE must be None there.
PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE") or (
    "free" if os.path.exists(os.path.expanduser("~/.databrickscfg")) and "DATABRICKS_RUNTIME_VERSION" not in os.environ else None
)

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

# The registered model (Unity Catalog) and its serving endpoint: the lifecycle half of the POC.
MODEL_NAME = f"{CATALOG}.{SCHEMA}.support_assistant"
MODEL_ALIAS = "champion"
SERVING_ENDPOINT = "fr-support-assistant"
EVAL_GATE = 0.80  # strict pass rate a version must reach to take the alias
