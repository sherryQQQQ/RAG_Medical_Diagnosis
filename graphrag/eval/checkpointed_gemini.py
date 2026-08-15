"""Small checkpointed Gemini adapter with hard call, prompt, and cost guards."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from graphrag.eval.mirage_benchmark import MODEL_PRICING_USD_PER_MILLION


@dataclass(frozen=True)
class ProviderResponse:
    raw_output: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_s: float
    reused: bool


Provider = Callable[[str, str, str, Mapping[str, Any]], ProviderResponse]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, Mapping) else str(item)
            for item in content
        )
    return str(content or "")


def _usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


class CheckpointedGeminiProvider:
    """Invoke Gemini only when a content-addressed response is not checkpointed."""

    def __init__(
        self,
        *,
        model: str,
        checkpoint_path: Path,
        fingerprint: str,
        max_calls: int = 35,
        max_cost_usd: float = 0.25,
        max_prompt_chars: int = 14_000,
        max_output_tokens: int = 1_024,
        allow_new_calls: bool = True,
    ):
        self.model = model
        self.path = checkpoint_path
        self.fingerprint = fingerprint
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.max_prompt_chars = max_prompt_chars
        self.max_output_tokens = max_output_tokens
        self.allow_new_calls = allow_new_calls
        self.pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
        if not self.pricing:
            raise ValueError(f"Paid execution requires configured pricing for {model}")
        if self.path.exists():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                self.payload.get("model") != model
                or self.payload.get("dataset_fingerprint") != fingerprint
            ):
                raise ValueError("Provider checkpoint metadata does not match this pilot")
        else:
            self.payload = {
                "format_version": 1,
                "model": model,
                "dataset_fingerprint": fingerprint,
                "calls": {},
            }
        self._llms: dict[str, Any] = {}

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * float(self.pricing["input"])
            + output_tokens * float(self.pricing["output_including_thinking"])
        ) / 1_000_000

    @property
    def spent(self) -> float:
        return sum(
            self.estimate_cost(
                int(item["input_tokens"]), int(item["output_tokens"])
            )
            for item in self.payload["calls"].values()
        )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def __call__(
        self, call_id: str, stage: str, prompt: str, schema: Mapping[str, Any]
    ) -> ProviderResponse:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        schema_hash = _canonical_hash(schema)
        existing = self.payload["calls"].get(call_id)
        if existing is not None:
            if (
                existing["stage"] != stage
                or existing["prompt_sha256"] != prompt_hash
            ):
                raise ValueError(f"Checkpoint prompt changed for {call_id}")
            if (
                existing.get("schema_sha256") is not None
                and existing["schema_sha256"] != schema_hash
            ):
                raise ValueError(f"Checkpoint schema changed for {call_id}")
            return ProviderResponse(
                raw_output=existing["raw_output"],
                input_tokens=int(existing["input_tokens"]),
                output_tokens=int(existing["output_tokens"]),
                total_tokens=int(existing["total_tokens"]),
                latency_s=float(existing["latency_s"]),
                reused=True,
            )
        if not self.allow_new_calls:
            raise RuntimeError(f"Reuse-only mode has no checkpoint for {call_id}")
        if len(self.payload["calls"]) >= self.max_calls:
            raise RuntimeError(f"Provider call guard exceeded: {self.max_calls}")
        if len(prompt) > self.max_prompt_chars:
            raise ValueError(f"Prompt exceeds {self.max_prompt_chars} characters")
        conservative_next_cost = self.estimate_cost(
            len(prompt), self.max_output_tokens
        )
        if self.spent + conservative_next_cost > self.max_cost_usd:
            raise RuntimeError(
                f"Cost guard would be exceeded: ${self.spent:.4f} + "
                f"${conservative_next_cost:.4f} > ${self.max_cost_usd:.4f}"
            )

        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        from graphrag.config import (
            GEMINI_MAX_RETRIES,
            GEMINI_REQUEST_TIMEOUT_S,
            GOOGLE_API_KEY,
        )

        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        schema_key = schema_hash
        if schema_key not in self._llms:
            self._llms[schema_key] = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=GOOGLE_API_KEY,
                temperature=0,
                request_timeout=GEMINI_REQUEST_TIMEOUT_S,
                retries=GEMINI_MAX_RETRIES,
                max_tokens=self.max_output_tokens,
                thinking_budget=0,
                response_mime_type="application/json",
                response_schema=dict(schema),
            )
        started = time.perf_counter()
        response = self._llms[schema_key].invoke([HumanMessage(content=prompt)])
        latency = time.perf_counter() - started
        record = {
            "stage": stage,
            "prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
            "raw_output": _content_text(response),
            "latency_s": latency,
            **_usage(response),
        }
        self.payload["calls"][call_id] = record
        self._save()
        if self.spent > self.max_cost_usd:
            raise RuntimeError("Actual provider cost exceeded the configured guard")
        return ProviderResponse(
            raw_output=record["raw_output"],
            input_tokens=int(record["input_tokens"]),
            output_tokens=int(record["output_tokens"]),
            total_tokens=int(record["total_tokens"]),
            latency_s=float(record["latency_s"]),
            reused=False,
        )
