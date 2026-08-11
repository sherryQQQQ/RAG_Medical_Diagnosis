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
from typing import Annotated, TypedDict

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
    if "retrieve_graph" not in used_tools:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_graph",
                    "args": {"symptoms": json.dumps([state["query"]])},
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
                    "args": {"query": state["query"]},
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
    text = state["query"] + "\n" + _content_to_text(draft.content)
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
            new_context.append(_content_to_text(ms