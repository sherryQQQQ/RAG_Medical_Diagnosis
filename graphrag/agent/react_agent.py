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

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graphrag.agent.tools import TOOLS
from graphrag.config import GEMINI_MODEL, GOOGLE_API_KEY

MAX_RETRIES = 3

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
    messages: Annotated[list, "append-only"]
    retrieved_context: list[str]
    retry_count: int
    final_answer: str


# ---------- Node functions ----------------------------------------------------

def reason_node(state: AgentState) -> dict:
    """LLM decides: call a tool or produce final answer."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    ).bind_tools(TOOLS)

    messages = [HumanMessage(content=SYSTEM_PROMPT + "\n\nPatient query: " + state["query"])]
    messages.extend(state["messages"])
    response: AIMessage = llm.invoke(messages)
    return {"messages": [response]}


def act_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    tool_node = ToolNode(TOOLS)
    result = tool_node.invoke(state)
    # Collect tool outputs as retrieved context
    new_context = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            new_context.append(msg.content)
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
        return {"final_answer": "Unable to generate answer.", "retry_count": MAX_RETRIES}

    candidate = last_ai.content
    context_summary = "\n---\n".join(state["retrieved_context"][-6:])  # last 6 chunks

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )
    validation_prompt = VALIDATE_PROMPT.format(
        query=state["query"],
        answer=candidate,
        context=context_summary[:3000],  # token budget
    )
    validation_response = llm.invoke([HumanMessage(content=validation_prompt)])
    verdict = validation_response.content.strip()

    if verdict.startswith("APPROVED"):
        return {"final_answer": candidate, "retry_count": state["retry_count"]}
    else:
        # Inject critique as a new human message to guide retry
        critique = verdict.replace("RETRY:", "").strip()
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Please revise: {critique}")],
            "retry_count": state["retry_count"] + 1,
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

def build_agent() -> StateGraph:
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
    }
    result = agent.invoke(initial_state)
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
