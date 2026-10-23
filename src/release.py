"""Model lifecycle: log a version of the assistant, evaluate it, promote it only if it passes, serve it.

    uv run python src/release.py              # log → register → eval gate → alias "champion" (or not)
    uv run python src/release.py --deploy     # the same, then deploy the champion to Model Serving
    uv run python src/release.py --deploy-only   # no new version: (re)deploy the current champion

What "lifecycle" means here, concretely:
  version    every run registers a new version of workspace.rag.support_assistant in Unity Catalog,
             logged from code (src/agent.py + ask.py + config.py), with the index and the chat endpoint
             declared as resources so a serving container gets pass-through auth to them.
  evaluate   the 30-case eval set (evaluate.py's scorers) runs against the version, linked to it in
             MLflow, so each version carries its own scores.
  promote    the alias `champion` moves to the new version only if strict pass rate ≥ EVAL_GATE;
             otherwise the version stays registered, unaliased, with the failing scores attached.
             The previous champion keeps the alias `previous` so a rollback is one alias call.
  serve      --deploy puts models:/<name>@champion behind a Model Serving endpoint
             (databricks.agents.deploy; ~15 min the first time, runs in this process).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
from mlflow import MlflowClient
from mlflow.genai import evaluate
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

from config import CHAT_MODEL, EVAL_GATE, EXPERIMENT, INDEX, MODEL_ALIAS, MODEL_NAME, PROFILE, SERVING_ENDPOINT
from evaluate import form_offered, load_cases, must_match, must_not_match, predict_fn, strict_pass

SRC = Path(__file__).resolve().parent


def deploy_champion(client: MlflowClient) -> int:
    from databricks import agents

    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    print(f"deploying {MODEL_NAME}@{MODEL_ALIAS} (version {version}) to serving endpoint {SERVING_ENDPOINT}; this takes a while")
    # scale_to_zero is REQUIRED on Free Edition ("Scale to zero must be enabled for this workspace"),
    # and right anyway for a demo endpoint that is idle most of the day.
    d = agents.deploy(MODEL_NAME, int(version), endpoint_name=SERVING_ENDPOINT, scale_to_zero=True)
    print(f"deployed: endpoint {d.endpoint_name}, query at {d.query_endpoint}")
    return 0


def main() -> int:
    deploy = "--deploy" in sys.argv
    mlflow.set_tracking_uri(f"databricks://{PROFILE}" if PROFILE else "databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    if "--deploy-only" in sys.argv:
        return deploy_champion(client)

    with mlflow.start_run(run_name="release") as run:
        info = mlflow.pyfunc.log_model(
            name="assistant",
            python_model=str(SRC / "agent.py"),
            code_paths=[str(SRC / "ask.py"), str(SRC / "config.py")],
            resources=[DatabricksServingEndpoint(endpoint_name=CHAT_MODEL), DatabricksVectorSearchIndex(index_name=INDEX)],
            input_example={"input": [{"role": "user", "content": "Where do I put the .frlicense file on a Mac?"}]},
            pip_requirements=[f"mlflow=={mlflow.__version__}", "databricks-sdk>=0.40"],
            registered_model_name=MODEL_NAME,
        )
        version = info.registered_model_version
        print(f"registered {MODEL_NAME} version {version}")

        # The eval, linked to this version. predict_fn is the same code the version was logged from.
        result = evaluate(data=load_cases(), predict_fn=predict_fn, scorers=[must_match, must_not_match, form_offered, strict_pass], model_id=info.model_id)
        rate = float(result.metrics.get("strict_pass/mean", 0.0))
        n = len(load_cases())
        mlflow.log_metric("strict_pass_rate", rate)
        print(json.dumps(result.metrics, indent=2))
        print(f"version {version}: {round(rate * n)}/{n} strict ({rate:.0%}); gate {EVAL_GATE:.0%}")

        client.set_model_version_tag(MODEL_NAME, version, "strict_pass_rate", f"{rate:.3f}")
        client.set_model_version_tag(MODEL_NAME, version, "eval_run_id", run.info.run_id)

    if rate < EVAL_GATE:
        print(f"below the gate: version {version} stays registered but NOT promoted; champion unchanged")
        return 1

    try:
        prev = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        if prev.version != version:
            client.set_registered_model_alias(MODEL_NAME, "previous", prev.version)
            print(f"previous champion was version {prev.version} (now alias 'previous')")
    except Exception:
        pass
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
    print(f"promoted: {MODEL_NAME}@{MODEL_ALIAS} -> version {version}")

    if deploy:
        return deploy_champion(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
