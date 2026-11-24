"""
providers/llm/scripted.py
─────────────────────────
The scripted fake LLM adapter, for tests. Not registered in the provider
factory, so configuration can never select it: tests inject it, one instance
per agent, and the instance's agent name is how errors identify the agent.

Scripting:
  - Coding agents by file path: ``files={path: reply}``. The file(s) a request
    asks for are read from its coding request header (agents/coding_request.py).
    A single-file request gets that file's reply; a batch gets a JSON array of
    the files' replies in header order, each carrying its ``filepath``.
  - Single-call agents (requirements, architecture, planner, repair) by call:
    ``calls=[reply, ...]``, answered in order to requests without a header.

A reply is a dict or list (sent as JSON) or a str (sent as is; in a batch it
must be a JSON object). The fake is strict: a request it was not scripted for
raises UnscriptedRequestError naming the agent and the file (or the call).
Every request is recorded in ``requests`` and reports 10 input and 10 output
tokens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Mapping, Optional, Sequence

from agents.coding_request import parse_request_header
from providers.llm.base import BaseLLMProvider, LLMResponse

Reply = Any  # dict | list | str


class UnscriptedRequestError(LookupError):
    """The scripted fake received a request no script answers."""


@dataclass(frozen=True)
class ScriptedRequest:
    """One request the scripted fake received."""
    system_prompt: str
    user_message: str
    files: list[str] = field(default_factory=list)  # from the header; [] for a call


def _as_text(reply: Reply) -> str:
    return reply if isinstance(reply, str) else json.dumps(reply)


def _as_item(reply: Reply, path: str) -> dict:
    item = json.loads(reply) if isinstance(reply, str) else dict(reply)
    if not isinstance(item, dict):
        raise TypeError(f"a batched reply must be a JSON object, got {type(item).__name__} for {path}")
    return {**item, "filepath": path}


class ScriptedLLMProvider(BaseLLMProvider):
    """A strict fake LLM for one agent, answering by file path or by call."""

    def __init__(
        self,
        agent: str,
        *,
        files: Optional[Mapping[str, Reply]] = None,
        calls: Optional[Sequence[Reply]] = None,
    ) -> None:
        self.agent = agent
        self.files: dict[str, Reply] = dict(files or {})
        self.calls: list[Reply] = list(calls or [])
        self.requests: list[ScriptedRequest] = []
        self._calls_answered = 0
        self._lock = Lock()  # coding agents call from a thread pool

    @property
    def provider_name(self) -> str:
        return f"scripted({self.agent})"

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        stream: bool = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        paths = parse_request_header(user_message)
        with self._lock:
            self.requests.append(ScriptedRequest(system_prompt, user_message, list(paths)))
            text = self._file_reply(paths) if paths else self._call_reply()
        if stream and chunk_callback is not None:
            chunk_callback(text)
        return LLMResponse(
            text=text,
            input_tokens=10,
            output_tokens=10,
            model=model,
            provider=self.provider_name,
        )

    def _file_reply(self, paths: list[str]) -> str:
        missing = [p for p in paths if p not in self.files]
        if missing:
            raise UnscriptedRequestError(
                f"scripted fake for agent '{self.agent}' has no reply for file(s): {', '.join(missing)}"
            )
        if len(paths) == 1:
            return _as_text(self.files[paths[0]])
        return json.dumps([_as_item(self.files[p], p) for p in paths])

    def _call_reply(self) -> str:
        if self._calls_answered >= len(self.calls):
            raise UnscriptedRequestError(
                f"scripted fake for agent '{self.agent}' has no reply for call {self._calls_answered + 1} "
                f"(a request without a coding request header, so no file was named)"
            )
        reply = self.calls[self._calls_answered]
        self._calls_answered += 1
        return _as_text(reply)
