"""The same assistant as a LangGraph state graph.

    uv run python src/graph.py "Where do I put the .frlicense file on a Mac?"
    uv run python src/graph.py --draw          # print the graph as Mermaid

ask.py is a straight line: retrieve, then generate, then parse. This is that line drawn as a graph so the
control flow is explicit and inspectable:

    retrieve ──(no passages)──▶ refuse ──▶ END
        └────(passages)───────▶ generate ──(tool call)──▶ tools ──▶ generate ──▶ END
                                    └──────(text)──────────────────────────────▶ END

and with one real difference: `show_form` is a bound TOOL the model calls, as it is in the Cloudflare
build (worker/support.ts), rather than the tagged last line ask.py uses so that any hosted model can
work. Two things the loop handles that a straight line cannot: Llama 3.3 70B, when it calls the tool,
emits the call and no answer text, so the graph feeds the tool result back and asks for the text; and
the tools node is a POLICY GATE, not a passthrough: it honours a show_form call only on a
licensing-shaped question (the same regex that pins the FAQ), so an off-topic question cannot be
answered with a license form however the model feels about it. The model is reached through the Unity Gateway service with LangChain's OpenAI client, because the
gateway speaks the OpenAI chat-completions API; nothing in this file knows which model is behind it.
Same system prompt, same limits, same empty-retrieval guard. evaluate.py --graph scores this path with the
same 30 cases, so the two orchestrations are compared on the same questions.
"""
import sys
from pathlib import Path
from typing import Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import END, StateGraph

from ask import FORMS, LICENSING, LIMITS, NOTHING, SYSTEM, _user_message, client, retrieve
from config import GATEWAY_SERVICE

# The prompt's form mechanism, in the tool-call form the Cloudflare build uses.
SYSTEM_TOOL = SYSTEM.replace(
    "by ending your answer with a final line of exactly `FORM: <name>`:",
    "by calling the show_form tool after your answer:",
).replace(
    "Add that line only when that action is the next step for this visitor, at most once per answer, and still write the answer first. Do not describe the form mechanism; just say \"use the form below\".",
    "Call it only when that action is the next step for this visitor, at most once per answer, and still write the answer first. Do not describe the tool; just say \"use the form below\".",
)


@tool
def show_form(form: Literal["resend-license", "resend-download", "complete-bundle"]) -> str:
    """Show the visitor one of the site's self-service forms below your answer. They enter their own email in it; you never see it."""
    return f"form {form} shown"


class State(TypedDict, total=False):
    question: str
    passages: list[dict]
    messages: list  # the running chat: system, user, assistant (tool call), tool result, assistant
    answer: str
    form: str | None
    model: str | None


def _llm() -> BaseChatOpenAI:
    """LangChain's OpenAI client pointed at the gateway: host + OAuth token from the SDK's own config.

    BaseChatOpenAI rather than ChatOpenAI: the subclass rewrites `max_tokens` to `max_completion_tokens`
    (OpenAI's 2024 rename), which the gateway's chat-completions API rejects as an unknown field.
    """
    cfg = client().config
    token = cfg.authenticate()["Authorization"].split(" ", 1)[1]
    return BaseChatOpenAI(base_url=f"{cfg.host.rstrip('/')}/ai-gateway/mlflow/v1", api_key=token, model=GATEWAY_SERVICE, temperature=0, max_tokens=LIMITS["maxTokens"])


def retrieve_node(state: State) -> State:
    return {"passages": retrieve(state["question"])}


def route(state: State) -> Literal["refuse", "generate"]:
    return "generate" if state.get("passages") else "refuse"


def refuse_node(state: State) -> State:
    # Nothing retrieved → no model call (the empty-index hallucination, 2026-10-23).
    return {"answer": NOTHING, "form": None, "model": None}


def generate_node(state: State) -> State:
    msgs = state.get("messages") or [SystemMessage(content=SYSTEM_TOOL), HumanMessage(content=_user_message(state["question"], state["passages"]))]
    # First pass: tools bound, so the model can ask for a form. Second pass (after the tools node): no
    # tools, so it writes the answer instead of calling again.
    llm = _llm() if any(isinstance(m, ToolMessage) for m in msgs) else _llm().bind_tools([show_form])
    msg = llm.invoke(msgs)

    def text_of(m) -> str:
        return (m.content if isinstance(m.content, str) else " ".join(x.get("text", "") for x in m.content if isinstance(x, dict))).strip()

    text = text_of(msg)
    if not text and not msg.tool_calls:
        # Llama sometimes answers a tool result with nothing at all; ask once more, in words.
        msgs = [*msgs, msg, HumanMessage(content="Write the answer now, as text.")]
        msg = llm.invoke(msgs)
        text = text_of(msg)
    return {"messages": [*msgs, msg], "answer": text, "model": GATEWAY_SERVICE}


def after_generate(state: State) -> Literal["tools", "done"]:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "done"


def tools_node(state: State) -> State:
    """Execute show_form, under policy: a form is shown only for a licensing-shaped question."""
    last: AIMessage = state["messages"][-1]
    out, form = [], state.get("form")
    for call in last.tool_calls:
        want = call.get("args", {}).get("form")
        if call.get("name") == "show_form" and want in FORMS and LICENSING.search(state["question"]) and form is None:
            form = want
            result = f"The {want} form will be shown below your answer. Now write the answer text."
        else:
            result = "No form applies to this question; answer in text only."
        out.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": [*state["messages"], *out], "form": form}


def build():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("refuse", refuse_node)
    g.add_node("generate", generate_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route, {"refuse": "refuse", "generate": "generate"})
    g.add_conditional_edges("generate", after_generate, {"tools": "tools", "done": END})
    g.add_edge("tools", "generate")
    g.add_edge("refuse", END)
    return g.compile()


GRAPH = build()


def ask_graph(question: str) -> dict:
    out = GRAPH.invoke({"question": question})
    return {"question": question, "answer": out["answer"], "form": out.get("form"), "model": out.get("model"), "passages": out.get("passages", [])}


if __name__ == "__main__":
    if "--draw" in sys.argv:
        print(GRAPH.get_graph().draw_mermaid())
        sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(2)
    r = ask_graph(" ".join(args))
    print(r["answer"])
    if r["form"]:
        print(f"\n[form: {r['form']}]")
    print("\nsources:")
    for p in r["passages"][:3]:
        print(f"  {p['score']:.3f}  {p['title']}  {p['url']}")
