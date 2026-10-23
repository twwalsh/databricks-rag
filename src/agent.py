"""The RAG as an MLflow ResponsesAgent, so it can be versioned, evaluated, aliased and served.

Logged "from code" (release.py passes this file as python_model), so nothing is pickled: the
registered version IS this file plus ask.py/config.py, and a diff between two versions is a diff of
the code. The agent does what ask.py does; this class only adapts it to the Responses API shape that
Model Serving, the AI Playground and MLflow tracing all speak.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from ask import ask


class SupportAssistant(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Last user turn is the question; earlier turns are not used (the site widget sends one question).
        question = ""
        for item in reversed(request.input):
            d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            if d.get("role") == "user":
                c = d.get("content")
                question = c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
                break
        r = ask(question)
        text = r["answer"]
        if r["form"]:
            text += f"\n\nFORM: {r['form']}"
        if r["passages"]:
            text += "\n\nSources: " + " · ".join(f"{p['title']}" + (f" ({p['url']})" if p.get("url") else "") for p in r["passages"][:3])
        return ResponsesAgentResponse(output=[self.create_text_output_item(text=text, id="msg_1")])


mlflow.models.set_model(SupportAssistant())
