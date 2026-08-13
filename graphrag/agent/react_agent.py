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
import re
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Callable, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from graphrag.agent.tools import TOOLS, find_known_drugs
from graphrag.config import (
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_REQUEST_TIMEOUT_S,
    GOOGLE_API_KEY,
)

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

POLICY_VALIDATE_PROMPT = """Review this medical answer against both the task contract and the evidence.

QUERY: {query}
ANSWER: {answer}
RETRIEVED CONTEXT: {context}
TASK CONTRACT: {task_policy}

Apply these rules in order:
1. Enforce the task contract. Do not treat weak retrieval as permission to change the task.
2. Distinguish missing retrieval evidence from decisive information missing in the user's query.
3. If the query itself lacks decisive patient-specific information, require an explicit abstention
   or request for that information. Do not fill it in from a generic guideline.
4. If a tool failed and the answer explicitly discloses the failure and safely declines an
   unsupported conclusion, that answer is complete and should be approved.
5. If retrieved sources conflict, require the answer to disclose the conflict rather than silently
   selecting one source.
6. Otherwise require the answer to be grounded, safe, and responsive.

Reply with ONLY one of:
- "APPROVED: <brief reason>"
- "RETRY: <specific issue to fix>"
"""


@dataclass(frozen=True)
class RetrievalStep:
    """A required retrieval tool call with deterministic benchmark-safe args."""

    tool_name: str
    build_args: Callable[[str], dict[str, Any]]
    instruction: str


DEFAULT_RETRIEVAL_STEPS = (
    RetrievalStep(
        tool_name="retrieve_graph",
        build_args=lambda query: {"symptoms": json.dumps([query])},
        instruction=(
            "Next step: call retrieve_graph exactly once. Extract the patient's "
            "specific symptoms, diagnoses, or conditions as the JSON list argument."
        ),
    ),
    RetrievalStep(
        tool_name="retrieve_vector",
        build_args=lambda query: {"query": query},
        instruction=(
            "Next step: call retrieve_vector exactly once using the full patient "
            "query. This is required even if the graph result was empty."
        ),
    ),
)


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
    # Per-example task contracts let the same graph distinguish forced-choice
    # benchmarks from open clinical questions without leaking answer keys.
    task_mode: NotRequired[str]
    task_policy: NotRequired[str]
    allowed_choices: NotRequired[list[str]]
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    total_tokens: NotRequired[int]
    provider_request_count: NotRequired[int]
    reflection_request_count: NotRequired[int]
    policy_approval_count: NotRequired[int]
    retrieved_ids: NotRequired[list[str]]
    retrieval_latency_s: NotRequired[float]
    context_count: NotRequired[int]


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


def _message_usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _usage_update(state: AgentState, message: Any) -> dict[str, int]:
    usage = _message_usage(message)
    return {
        key: int(state.get(key, 0)) + value for key, value in usage.items()
    }


def _explicit_answer_choice(candidate: str, allowed_choices: list[str]) -> str | None:
    """Return an explicit final choice, avoiding letters found only in reasoning."""
    allowed = {str(choice).strip().upper() for choice in allowed_choices}
    if not allowed:
        return None
    for match in re.finditer(r"\{[^{}]*\}", candidate, flags=re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for key in ("answer_choice", "prediction", "answer"):
            value = str(payload.get(key, "")).strip().upper()
            if value in allowed:
                return value
    patterns = (
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"(?:final[_\s-]*answer|answer[_\s-]*choice|answer|choice)"
        r"\s*[`*_\"']*\s*(?:is\s*)?(?::|=)?\s*[`*\"']*\(?([A-Z])\)?",
        r"(?:final\s+choice|chosen\s+answer)\s*[`*\"']*\s*[:=]\s*[`*\"'\[]*([A-Z])\b",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, candidate, flags=re.IGNORECASE))
    normalized = [match.upper() for match in matches if match.upper() in allowed]
    if normalized:
        return normalized[-1]
    compact = re.sub(r"[`*_\s]", "", candidate).upper()
    compact_match = re.fullmatch(r"\(?([A-Z])\)?[.)]?", compact)
    if compact_match and compact_match.group(1) in allowed:
        return compact_match.group(1)
    return None


def _is_safe_tool_failure_answer(candidate: str) -> bool:
    text = candidate.lower()
    disclosure_markers = ("tool", "retrieval", "source", "evidence", "unavailable", "failed")
    abstention_markers = (
        "cannot determine",
        "can't determine",
        "cannot answer",
        "can't answer",
        "cannot safely answer",
        "can't safely answer",
        "cannot be answered",
        "can't be answered",
        "insufficient",
        "not enough information",
        "unable to verify",
        "unable to confirm",
        "seek clinician",
    )
    return any(marker in text for marker in disclosure_markers) and any(
        marker in text for marker in abstention_markers
    )


def _task_policy_precheck(state: AgentState, candidate: str) -> str | None:
    """Approve contracts that can be checked deterministically before an LLM critique."""
    task_mode = state.get("task_mode", "default")
    if task_mode == "multiple_choice":
        choice = _explicit_answer_choice(candidate, state.get("allowed_choices", []))
        if choice:
            return (
                "APPROVED: deterministic task contract preserved explicit valid "
                f"choice {choice}"
            )
        return (
            "RETRY: deterministic task contract requires exactly one explicit "
            "valid choice from the supplied options; abstention is not allowed"
        )
    if task_mode == "open_medical":
        tool_failed = any(
            "tool unavailable" in context.lower()
            for context in state.get("retrieved_context", [])
        )
        if tool_failed and _is_safe_tool_failure_answer(candidate):
            return (
                "APPROVED: deterministic task contract accepted explicit safe "
                "tool-failure disclosure"
            )
    return None


def _used_tool_names(state: AgentState) -> list[str]:
    return [
        msg.name
        for msg in state.get("messages", [])
        if isinstance(msg, ToolMessage) and msg.name
    ]


def _retrieval_query(state: AgentState) -> str:
    return (state.get("retrieval_query") or state["query"]).strip()


def _next_step_instruction(
    state: AgentState,
    retrieval_steps: tuple[RetrievalStep, ...] = DEFAULT_RETRIEVAL_STEPS,
    enable_safety: bool = True,
) -> str:
    """Keep the ReAct loop bounded: required retrieval, then an answer."""
    used_tools = _used_tool_names(state)
    for step in retrieval_steps:
        if step.tool_name not in used_tools:
            return step.instruction
    if enable_safety and "check_contraindications" not in used_tools:
        return (
            "If you will recommend a named drug, call check_contraindications at "
            "most once. Otherwise stop using tools and write the final answer."
        )
    return "Stop using tools and write the final answer grounded only in retrieved context."


def _mandatory_retrieval_call(
    state: AgentState,
    retrieval_steps: tuple[RetrievalStep, ...] = DEFAULT_RETRIEVAL_STEPS,
) -> AIMessage | None:
    """Enforce grounding tools in code instead of relying on prompt compliance."""
    used_tools = _used_tool_names(state)
    retrieval_query = _retrieval_query(state)
    for step in retrieval_steps:
        if step.tool_name in used_tools:
            continue
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": step.tool_name,
                    "args": step.build_args(retrieval_query),
                    "id": f"{step.tool_name}-{uuid.uuid4().hex}",
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


def reason_node(
    state: AgentState,
    *,
    tools: list[Any] | None = None,
    retrieval_steps: tuple[RetrievalStep, ...] = DEFAULT_RETRIEVAL_STEPS,
    system_prompt: str = SYSTEM_PROMPT,
    enable_safety: bool = True,
    allow_optional_tool_calls: bool = True,
    max_tool_calls: int = MAX_TOOL_CALLS,
    model: str = GEMINI_MODEL,
    max_tokens: int | None = None,
) -> dict:
    """LLM decides: call a tool or produce final answer."""
    effective_tools = TOOLS if tools is None else tools
    mandatory_call = _mandatory_retrieval_call(state, retrieval_steps)
    if mandatory_call is not None:
        return {"messages": [mandatory_call], "status": "retrieving"}

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        request_timeout=GEMINI_REQUEST_TIMEOUT_S,
        retries=GEMINI_MAX_RETRIES,
        max_tokens=max_tokens,
    )
    if allow_optional_tool_calls:
        llm = llm.bind_tools(effective_tools)

    phase_instruction = _next_step_instruction(
        state, retrieval_steps, enable_safety
    )
    messages = [
        HumanMessage(
            content=(
                system_prompt
                + "\n\n"
                + phase_instruction
                + "\n\nPatient query: "
                + state["query"]
            )
        )
    ]
    messages.extend(state["messages"])
    response: AIMessage = llm.invoke(messages)
    usage_update = _usage_update(state, response)
    request_update = {
        "provider_request_count": int(state.get("provider_request_count", 0)) + 1
    }
    safety_call = (
        _mandatory_safety_call(state, response) if enable_safety else None
    )
    if safety_call is not None:
        return {
            "messages": [safety_call],
            "status": "safety_checking",
            **usage_update,
            **request_update,
        }
    used_count = len(_used_tool_names(state))
    if response.tool_calls and used_count + len(response.tool_calls) > max_tool_calls:
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
            **usage_update,
            **request_update,
        }
    return {"messages": [response], **usage_update, **request_update}


def act_node(state: AgentState, tools: list[Any] | None = None) -> dict:
    """Execute tool calls from the last AI message."""
    effective_tools = TOOLS if tools is None else tools
    tool_node = ToolNode(
        effective_tools,
        handle_tool_errors=lambda error: (
            f"Tool unavailable ({type(error).__name__}). Continue using the other "
            "retrieved evidence and explicitly disclose the missing check."
        ),
    )
    result = tool_node.invoke(state)
    # Collect tool outputs as retrieved context
    new_context = []
    retrieved_ids = list(state.get("retrieved_ids", []))
    retrieval_latency_s = float(state.get("retrieval_latency_s", 0.0))
    context_count = int(state.get("context_count", 0))
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            new_context.append(_content_to_text(msg.content))
            artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
            retrieved_ids.extend(
                str(value) for value in artifact.get("retrieved_ids", [])
            )
            retrieval_latency_s += float(artifact.get("retrieval_latency_s", 0.0))
            context_count += int(artifact.get("context_count", 0))
    return {
        "messages": result["messages"],
        "retrieved_context": state["retrieved_context"] + new_context,
        "retrieved_ids": retrieved_ids,
        "retrieval_latency_s": retrieval_latency_s,
        "context_count": context_count,
    }


def validate_node(
    state: AgentState,
    model: str = GEMINI_MODEL,
    *,
    validation_prompt_template: str = VALIDATE_PROMPT,
    enable_policy_precheck: bool = False,
    max_tokens: int | None = None,
) -> dict:
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
        context[:8000] for context in reversed(state["retrieved_context"][-6:])
    )

    if enable_policy_precheck:
        policy_verdict = _task_policy_precheck(state, candidate)
        if policy_verdict:
            policy_update = {
                "candidate_answers": [candidate],
                "validation_verdicts": [policy_verdict],
            }
            if policy_verdict.startswith("APPROVED"):
                return {
                    **policy_update,
                    "final_answer": candidate,
                    "retry_count": state["retry_count"],
                    "status": "policy_approved",
                    "policy_approval_count": int(
                        state.get("policy_approval_count", 0)
                    )
                    + 1,
                }
            next_retry = state["retry_count"] + 1
            if next_retry >= MAX_RETRIES:
                return {
                    **policy_update,
                    "final_answer": candidate,
                    "retry_count": next_retry,
                    "status": "max_retries_unapproved",
                }
            critique = policy_verdict.replace("RETRY:", "").strip()
            return {
                **policy_update,
                "messages": [HumanMessage(content=f"Please revise: {critique}")],
                "retry_count": next_retry,
                "status": "retrying",
            }

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        request_timeout=GEMINI_REQUEST_TIMEOUT_S,
        retries=GEMINI_MAX_RETRIES,
        max_tokens=max_tokens,
    )
    validation_prompt = validation_prompt_template.format(
        query=state["query"],
        answer=candidate,
        context=context_summary[:8000],
        task_policy=state.get(
            "task_policy",
            "Give a grounded, safe, complete answer to the user's query.",
        ),
    )
    validation_response = llm.invoke([HumanMessage(content=validation_prompt)])
    verdict = _content_to_text(validation_response.content).strip()
    usage_update = _usage_update(state, validation_response)

    validation_update = {
        "candidate_answers": [candidate],
        "validation_verdicts": [verdict],
        "provider_request_count": int(state.get("provider_request_count", 0)) + 1,
        "reflection_request_count": int(state.get("reflection_request_count", 0)) + 1,
        **usage_update,
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

def build_agent(
    *,
    tools: list[Any] | None = None,
    retrieval_steps: tuple[RetrievalStep, ...] = DEFAULT_RETRIEVAL_STEPS,
    system_prompt: str = SYSTEM_PROMPT,
    enable_safety: bool = True,
    allow_optional_tool_calls: bool = True,
    max_tool_calls: int = MAX_TOOL_CALLS,
    model: str = GEMINI_MODEL,
    max_tokens: int | None = None,
    validation_prompt_template: str = VALIDATE_PROMPT,
    enable_policy_precheck: bool = False,
):
    effective_tools = TOOLS if tools is None else tools
    graph = StateGraph(AgentState)

    graph.add_node(
        "reason",
        lambda state: reason_node(
            state,
            tools=effective_tools,
            retrieval_steps=retrieval_steps,
            system_prompt=system_prompt,
            enable_safety=enable_safety,
            allow_optional_tool_calls=allow_optional_tool_calls,
            max_tool_calls=max_tool_calls,
            model=model,
            max_tokens=max_tokens,
        ),
    )
    graph.add_node("act", lambda state: act_node(state, effective_tools))
    graph.add_node(
        "validate",
        lambda state: validate_node(
            state,
            model,
            validation_prompt_template=validation_prompt_template,
            enable_policy_precheck=enable_policy_precheck,
            max_tokens=max_tokens,
        ),
    )

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
