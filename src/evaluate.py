"""Run the support chat's eval set against the Databricks RAG and log it to MLflow.

    uv run python src/evaluate.py                 # all 30 cases, logged to the Databricks experiment
    uv run python src/evaluate.py --local         # same, but MLflow runs stay in ./mlruns
    uv run python src/evaluate.py --graph         # score the LangGraph path (graph.py) instead of ask.py

The cases are data/eval.jsonl, copied verbatim from the Cloudflare build (site/support/eval.jsonl
there): a question, `must` regexes the answer has to match, `mustNot` ones it must not, and sometimes
the self-service `form` it has to offer. The three scorers below are those rules, so a Databricks
score and a Cloudflare score mean the same thing. Each case is pass/fail on all of its rules, which is
how support-eval.mjs reports it; `strict_pass` is that number.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports when run as a job task

# The harness fans predict_fn out over many threads by default. With the Databricks SDK that
# deadlocked on 2026-10-23: 243 threads, every socket in CLOSE_WAIT, 0% CPU. Two workers is plenty
# for 30 questions and keeps the pay-per-token endpoints out of their rate limit too.
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")

import mlflow
from mlflow.genai import evaluate, scorer

from ask import ask, client
from graph import ask_graph
from config import EXPERIMENT, PROFILE

CASES = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"


def load_cases():
    rows = []
    for line in CASES.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        exp = {"must": c.get("must", []), "must_not": c.get("mustNot", [])}
        if c.get("form"):
            exp["form"] = c["form"]  # MLflow rejects an expectation whose value is None, so absent means "no form required"
        rows.append({"inputs": {"question": c["q"]}, "expectations": exp})
    return rows


ANSWERS: dict[str, dict] = {}  # question -> what predict_fn returned, so misses can be printed without a trace query


def predict_fn(question: str) -> dict:
    r = (ask_graph if "--graph" in sys.argv else ask)(question)
    ANSWERS[question] = r
    return {"answer": r["answer"], "form": r["form"], "sources": [p["title"] for p in r["passages"]]}


@scorer
def must_match(outputs, expectations) -> bool:
    text = outputs.get("answer", "")
    return all(re.search(p, text, re.I) for p in expectations.get("must", []))


@scorer
def must_not_match(outputs, expectations) -> bool:
    text = outputs.get("answer", "")
    return not any(re.search(p, text, re.I) for p in expectations.get("must_not", []))


@scorer
def form_offered(outputs, expectations) -> bool:
    want = expectations.get("form")
    return True if not want else outputs.get("form") == want


@scorer
def strict_pass(outputs, expectations) -> bool:
    """The Cloudflare runner's verdict: every rule of the case holds."""
    return must_match(outputs=outputs, expectations=expectations) and must_not_match(outputs=outputs, expectations=expectations) and form_offered(outputs=outputs, expectations=expectations)


def main() -> int:
    if "--local" not in sys.argv:
        mlflow.set_tracking_uri(f"databricks://{PROFILE}")
        mlflow.set_experiment(EXPERIMENT)
    rows = load_cases()
    client()  # build the SDK client once, on the main thread, before the workers start
    with mlflow.start_run(run_name="eval-graph" if "--graph" in sys.argv else "eval-ask"):
        result = evaluate(data=rows, predict_fn=predict_fn, scorers=[must_match, must_not_match, form_offered, strict_pass])
    m = result.metrics
    print(json.dumps(m, indent=2))
    passed = round(m.get("strict_pass/mean", 0) * len(rows))
    print(f"\n{passed}/{len(rows)} passed (strict)")
    for row in rows:
        q, exp = row["inputs"]["question"], row["expectations"]
        r = ANSWERS.get(q)
        if r is None:
            continue
        out = {"answer": r["answer"], "form": r["form"]}
        problems = [f"missing /{p}/" for p in exp["must"] if not re.search(p, out["answer"], re.I)]
        problems += [f"said /{p}/" for p in exp["must_not"] if re.search(p, out["answer"], re.I)]
        if exp.get("form") and out["form"] != exp["form"]:
            problems.append(f"form: wanted {exp['form']}, got {out['form']}")
        if problems:
            print(f"FAIL  {q}\n      {'; '.join(problems)}\n      form={out['form']} answer: {out['answer'][:300].replace(chr(10), ' ')}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
