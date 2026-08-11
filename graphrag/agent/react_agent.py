"""
LangGraph ReAct Agent
---------------------
State machine with 4 nodes:
  reason   → LLM decides: call a tool OR produce final answer
  act      → execute the chosen tool
  validate → self-critique: is the answer grounded in retrieved context?
  END

Flow:
  START → reason → act → reason → ... → reason → validate → END
                                              ↑_______________↓ (retry if not grounded)

Usage:
    agent = build_agent()
    result = agent.invoke({"query": "patient has fever and difficulty breathing"})
    print(result["final_answer"])
"""

import operator
import json
import uuid
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from graphrag.agent.tools import TOOLS, find_known_drugs
from graphrag.config import GEMINI_MODEL, GOOGLE_API_KEY

MAX_RETRIES = 3
MAX_TOOL_CALLS = 3

SYSTEM_PROMPT = """You are a medical diagnostic assistant. Your job is to help identify possible
diagnoses and treatments based on patient-reported symptoms.

You have access to three tools:
- retrieve_vector: semantic search over clinical guidelines
- retrieve_graph: structured knowledge graph lookup (symptoms → diseases → treatments)
- check_contraindications: drug safety check for a patient's conditions

Strategy:
1. First use retrieve_graph with the patient's symptoms for structured diagnostic paths
2. Then use retrieve_vector for additional clinical context
3. If a treatment involves drugs, use check_contraindications before recommending
4. Synthesize a clear, grounded answer citing the retrieved information

IMPORTANT: Never fabricate diagnoses or medications. Only recommend what the retrieved
documents explicitly support. Always note if more clinical evaluation is needed."""

VALIDATE_PROMPT = """Review this medical answer:

QUERY: {query}
ANSWER: {answer}
RETRIEVED CONTEXT: {context}

Is this answer:
1. Grounded in the retrieved context (not fabricated)?
2. Safe (no unsupported drug recommendations)?
3. Complete (addresses the patient's query)?

Reply with ONLY one of:
- "APPROVED: <brief reason>"
- "RETRY: <specific issue to fix>"
"""


# ---------- State -------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    # Benchmarks can keep answer options out of retrieval while still showing
    # them to the generator. Normal interactive calls omit this field.
    retrieval_query: NotRequired[str]
    messages: Annotated[list, add_messages]
    retrieved_context: list[str]
    retry_count: int
    final_answer: str
    candidate_answers: Annotated[list[str], operator.add]
    validation_verdicts: Annotated[list[str], operator.add]
    status: str


# ---------- Node functions ----------------------------------------------------

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _used_tool_names(state: AgentState) -> list[str]:
    return [
        msg.name
        for msg in state.get("messages", [])
        if isinstance(msg, ToolMessage) and msg.name
    ]


def _retrieval_query(state: AgentState) -> str:
    return (state.get("retrieval_query") or state["query"]).strip()


def _next_step_instruction(state: AgentState) -> str:
    """Keep the ReAct loop bounded: graph once, vector once, then answer."""
    used_tools = _used_tool_names(state)
    if "retrieve_graph" not in used_tools:
        return (
            "Next step: call retrieve_graph exactly once. Extract the patient's "
            "specific symptoms, diagnoses, or conditions as the JSON list argument."
        )
    if "retrieve_vector" not in used_tools:
        return (
            "Next step: call retrieve_vector exactly once using the full patient "
            "query. This is required even if the graph result was empty."
        )
    if "check_contraindications" not in used_tools:
        return (
            "If you will recommend a named drug, call check_contraindications at "
            "most once. Otherwise stop using tools and write the final answer."
        )
    return "Stop using tools and write the final answer grounded only in retrieved context."


def _mandatory_retrieval_call(state: AgentState) -> AIMessage | None:
    """Enforce grounding tools in code instead of relying on prompt compliance."""
    used_tools = _used_tool_names(state)
    retrieval_query = _retrieval_query(state)
    if "retrieve_graph" not in used_tools:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_graph",
                    "args": {"symptoms": json.dumps([retrieval_query])},
                    "id": f"graph-{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            ],
        )
    if "retrieve_vector" not in used_tools:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_vector",
                    "args": {"query": retrieval_query},
                    "id": f"vector-{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            ],
        )
    return None


def _mandatory_safety_call(state: AgentState, draft: AIMessage) -> AIMessage | None:
    """Force one contraindication check when a known drug appears in the draft."""
    if draft.tool_calls or "check_contraindications" in _used_tool_names(state):
        return None
    text = _retrieval_query(state) + "\n" + _content_to_text(draft.content)
    drugs = find_known_drugs(text)
    if not drugs:
        return None
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_contraindications",
                "args": {
                    "drug": drugs[0],
                    "conditions": json.dumps([state["query"]]),
                },
                "id": f"safety-{uuid.uuid4().hex}",
                "type": "tool_call",
            }
        ],
    )


def reason_node(state: AgentState) -> dict:
    """LLM decides: call a tool or produce final answer."""
    mandatory_call = _mandatory_retrieval_call(state)
    if mandatory_call is not None:
        return {"messages": [mandatory_call], "status": "retrieving"}

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    ).bind_tools(TOOLS)

    phase_instruction = _next_step_instruction(state)
    messages = [
        HumanMessage(
            content=(
                SYSTEM_PROMPT
                + "\n\n"
                + phase_instruction
                + "\n\nPatient query: "
                + state["query"]
            )
        )
    ]
    messages.extend(state["messages"])
    response: AIMessage = llm.invoke(messages)
    safety_call = _mandatory_safety_call(state, response)
    if safety_call is not None:
        return {"messages": [safety_call], "status": "safety_checking"}
    used_count = len(_used_tool_names(state))
    if response.tool_calls and used_count + len(response.tool_calls) > MAX_TOOL_CALLS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not complete the answer within the bounded tool budget. "
                        "Please seek clinician review rather than relying on an unverified answer."
                    )
                )
            ],
            "status": "tool_limit_reached",
        }
    return {"messages": [response]}


def act_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    tool_node = ToolNode(
        TOOLS,
        handle_tool_errors=lambda error: (
            f"Tool unavailable ({type(error).__name__}). Continue using the other "
            "retrieved evidence and explicitly disclose the missing check."
        ),
    )
    result = tool_node.invoke(state)
    # Collect tool outputs as retrieved context
    new_context = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            new_context.append(_content_to_text(msg.content))
    return {
        "messages": result["messages"],
        "retrieved_context": state["retrieved_context"] + new_context,
    }


def validate_node(state: AgentState) -> dict:
    """Self-critique: check if final answer is grounded in retrieved context."""
    # Extract last AI text response as candidate answer
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    if last_ai is None:
        return {
            "final_answer": "Unable to generate a grounded answer from the available evidence.",
            "retry_count": MAX_RETRIES,
            "status": "generation_failed",
        }

    candidate = _content_to_text(last_ai.content)
    # Put the most recent safety evidence first and cap each tool independently;
    # otherwise one long vector result can truncate later contraindication output.
    context_summary = "\n---\n".join(
        context[:2500] for context in reversed(state["retrieved_context"][-6:])
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )
    validation_prompt = VALIDATE_PROMPT.format(
        query=state["query"],
        answer=candidate,
        context=context_summary[:8000],
    )
    validation_response = llm.invoke([HumanMessage(content=validation_prompt)])
    verdict = _content_to_text(validation_response.content).strip()

    validation_update = {
        "candidate_answers": [candidate],
        "validation_verdicts": [verdict],
    }
    if verdict.startswith("APPROVED"):
        return {
            **validation_update,
            "final_answer": candidate,
            "retry_count": state["retry_count"],
            "status": "approved",
        }

    next_retry = state["retry_count"] + 1
    if next_retry >= MAX_RETRIES:
        # Preserve the best available candidate instead of ending with an empty
        # answer. The status still exposes that validation did not approve it.
        return {
            **validation_update,
            "final_answer": candidate,
            "retry_count": next_retry,
            "status": "max_retries_unapproved",
        }

    critique = verdict.replace("RETRY:", "").strip()
    if not critique:
        critique = "The validator did not approve the answer; make it safer and fully grounded."
    return {
        **validation_update,
        "messages": [HumanMessage(content=f"Please revise: {critique}")],
        "retry_count": next_retry,
        "status": "retrying",
    }


# ---------- Routing -----------------------------------------------------------

def route_after_reason(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "act"
    return "validate"


def route_after_validate(state: AgentState) -> str:
    if state.get("final_answer") or state["retry_count"] >= MAX_RETRIES:
        return END
    return "reason"


# ---------- Graph assembly ----------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", route_after_reason, {"act": "act", "validate": "validate"})
    graph.add_edge("act", "reason")
    graph.add_conditional_edges("validate", route_after_validate, {"reason": "reason", END: END})

    return graph.compile()


# ---------- CLI ---------------------------------------------------------------

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "Patient presents with high fever, cough, and chest pain."
    agent = build_agent()
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "retrieved_context": [],
        "retry_count": 0,
        "final_answer": "",
        "candidate_answers": [],
        "validation_verdicts": [],
        "status": "running",
    }
    result = agent.invoke(initial_state)
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
