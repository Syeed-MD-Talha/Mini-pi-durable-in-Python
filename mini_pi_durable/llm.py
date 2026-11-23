"""
LLM providers for mini_pi_durable:
- `FauxLLM`: deterministic offline mock model for learning & tests.
- `OpenAICompatibleLLM`: real client using the official `openai` SDK, compatible
  with OpenAI, DeepSeek, OpenRouter, Groq, Together, Ollama, LM Studio, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from openai import OpenAI

if TYPE_CHECKING:
    from .core import ToolDef


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list["ToolDef"] | None = None,
    ) -> LLMResponse: ...


class FauxLLM:
    """Deterministic mock model so demos run offline with zero API keys."""

    def __init__(self) -> None:
        self._responder: Callable[[list[dict[str, Any]], str], LLMResponse] | None = None
        self._queue: list[LLMResponse] = []

    def set_responses(self, responses: list[LLMResponse]) -> None:
        self._queue = list(responses)

    def set_responder(self, fn: Callable[[list[dict[str, Any]], str], LLMResponse]) -> None:
        self._responder = fn

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list["ToolDef"] | None = None,
    ) -> LLMResponse:
        if self._queue:
            return self._queue.pop(0)
        if self._responder is not None:
            return self._responder(messages, system_prompt)
        last = messages[-1] if messages else {}
        return LLMResponse(text=f"Echo: {last.get('content', '')}", input_tokens=10, output_tokens=5)


class OpenAICompatibleLLM:
    """
    Real LLM client powered by the official `openai` SDK.
    Works with OpenAI and any OpenAI-compatible endpoint by setting `base_url`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.strip() if base_url and base_url.strip() else None
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)

    def generate(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list["ToolDef"] | None = None,
    ) -> LLMResponse:
        api_messages: list[dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            role = m["role"]
            if role == "user":
                api_messages.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                tcs = m.get("tool_calls") or []
                if tcs:
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for tc in tcs
                    ]
                api_messages.append(msg)
            elif role == "tool_result":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["call_id"],
                        "content": str(m["output"]),
                    }
                )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = response.usage

        parsed_calls: list[LLMToolCall] = []
        for tc in message.tool_calls or []:
            raw_args = tc.function.arguments or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            parsed_calls.append(
                LLMToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        return LLMResponse(
            text=message.content or "",
            tool_calls=parsed_calls,
            input_tokens=int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else 0,
            output_tokens=int(usage.completion_tokens) if usage and usage.completion_tokens is not None else 0,
        )
