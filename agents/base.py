import json
import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Optional

from core.config import settings
from core.state import ProjectState, diff_state
from core.workspace import Workspace
from providers.llm.base import BaseLLMProvider, LLMResponse
from providers.llm.llmFactory import LLMFactory as llmFactory

logger = logging.getLogger(__name__)

DEFAULT_MODEL = settings.model_name
DEFAULT_MAX_TOKENS = settings.default_max_tokens or 8192
DEFAULT_RETRIES    = settings.default_retries or 3
DEFAULT_BACKOFF    = settings.default_backoff or 2.0

#: A JSON escape sequence still present in a string value (double-encoded output).
_ESCAPED_CHAR_RE = re.compile(r"\\(?:[\"\\/bfnrt]|u[0-9a-fA-F]{4})")


def deep_unescape(data: Any) -> Any:
    """
    Recursively unescape values that are themselves JSON-encoded strings
    (double-encoded LLM output, e.g. a `code` field containing
    `"from sqlalchemy import ...\\n..."` written as a JSON string).

    Two shapes are decoded:
      1. values still carrying escape sequences (`\\n`, `\\"`, ...) — inner
         JSON string, decoded with json.loads;
      2. values that are quote-wrapped with raw newlines inside — the outer
         parse already turned `\\n` into real newlines, leaving a string that
         is no longer valid JSON but clearly was a wrapped code blob.

    Plain code strings such as `"use strict"` are left untouched.
    """
    if isinstance(data, dict):
        return {k: deep_unescape(v) for k, v in data.items()}
    if isinstance(data, list):
        return [deep_unescape(v) for v in data]
    if isinstance(data, str):
        s = data
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            inner = s[1:-1]
            if _ESCAPED_CHAR_RE.search(s):
                try:
                    value = json.loads(s)
                    if isinstance(value, str):
                        return deep_unescape(value)
                except Exception:
                    pass
            elif "\n" in inner or re.search(r"=|;|[{}()]", inner):
                # Already-outer-parsed wrapped string: raw newlines inside, or a
                # single line that clearly holds code (`=`, braces, semicolons).
                return deep_unescape(inner)
        return s
    return data


####################BaseAgent##########################
class BaseAgent(ABC):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRIES,
        retries: int | None = None,
        backoff: float = DEFAULT_BACKOFF,
        temperature: float = 0.2,
        provider: BaseLLMProvider | str | None = None,
        workspace: Workspace | None = None,
    ):
        """Initialize the base agent. *workspace* is the output root generated
        files are written under; it defaults to the real project directory."""
        self.workspace = workspace if workspace is not None else Workspace()
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = retries if retries is not None else max_retries
        self.backoff = backoff
        self.temperature = temperature
        
        if isinstance(provider, str):
            provider_name = provider.lower().strip()
            self.provider = llmFactory.get(provider_name)
        elif provider is not None:
            self.provider = provider
        else:
            provider_name = self.model.split("/")[0] if "/" in self.model else settings.default_provider
            if provider_name not in llmFactory.available():
                provider_name = settings.default_provider
            self.provider = llmFactory.get(provider_name)

    @property
    def name(self)->str:
        """Get the name of the agent."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def system_prompt(self)->str:
        """Get the system prompt of the agent."""
        ...

    def _build_prompt(self, state: ProjectState) -> str:
        """Build the prompt for single-call agents using the default _work(). Override in subclass if using the default _work()."""
        raise NotImplementedError(f"[{self.name}] _build_prompt() is not implemented.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        """Parse response for single-call agents using the default _work(). Override in subclass if using the default _work()."""
        raise NotImplementedError(f"[{self.name}] _parse_response() is not implemented.")
    


    ##LLM CALL
    def llm_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> LLMResponse:
        

        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        prefix = agent_name if agent_name is not None else self.name

        printer = LineBufferedPrinter(prefix)
        response = self.provider.complete(
            system_prompt = sys_prompt,
            user_message  = prompt,
            model         = self.model,
            max_tokens    = self.max_tokens,
            temperature   = self.temperature,
            stream        = True,
            chunk_callback = printer,
        )
        printer.flush()
        return response

    def _call_with_system(
        self,
        system: str,
        user: str,
        agent_name: str = "",
        state: Optional[ProjectState] = None,
        lock: Optional[Any] = None,
    ) -> str:
        """
        Variant of llm_call() that accepts a dedicated system prompt.
        Retries up to self.max_retries with exponential back-off.
        Returns raw text and thread-safely updates state metrics if state is provided.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(
                    prompt=user,
                    system_prompt=system,
                    agent_name=agent_name or self.name,
                )
                if state is not None:
                    if lock is not None:
                        with lock:
                            state.metrics.total_tokens_used += response.total_tokens
                            state.metrics.total_llm_calls += 1
                    else:
                        state.metrics.total_tokens_used += response.total_tokens
                        state.metrics.total_llm_calls += 1
                return response.text
            except Exception as exc:
                logger.warning(
                    "[%s] LLM attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)
        raise RuntimeError("Unreachable")

    def _parse_batch_response(self, raw: str) -> list[dict]:
        """
        Parse raw text response from a batch LLM call into a list of dictionaries.
        Strips markdown code fences and extracts JSON array or object fallback.
        """
        text = self._strip_fences(raw).strip()

        # 1. Try parsing JSON array
        try:
            raw_arr = self._extract_raw_json(text, "[")
            items = self._try_parse(raw_arr, list)
            if isinstance(items, list):
                return deep_unescape(items)
        except Exception:
            pass

        # 2. Try parsing single JSON object and wrapping in list
        try:
            raw_obj = self._extract_raw_json(text, "{")
            items = self._try_parse(raw_obj, dict)
            if isinstance(items, dict):
                return deep_unescape([items])
        except Exception:
            pass

        # 3. Fallback using json_repair
        import json_repair
        try:
            res = json_repair.loads(text)
            if isinstance(res, list):
                return deep_unescape(res)
            elif isinstance(res, dict):
                return deep_unescape([res])
        except Exception:
            pass

        raise ValueError(f"Batch JSON parse failed: {raw[:300]}")


    def run(self, state: dict | ProjectState) -> dict[str, Any]:
        """
        The run skeleton shared by every agent.
        Accepts the full state (dict or ProjectState) as input: snapshots it,
        marks the agent's start, times the work step, records its duration in
        metrics and returns ONLY the changes, through the state-update module.
        Agents supply their work in _work(); they do not override run().
        """
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)

        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        self._work(ps)

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)
        logger.info("[%s] finished in %.2fs via %s", self.name, duration, self.provider.provider_name)
        return diff_state(before, ps)

    def _work(self, ps: ProjectState) -> None:
        """
        The agent's work step: update the state in place; returning early is fine.
        Default: one LLM call with retries, built by _build_prompt() and parsed
        by _parse_response(). Agents with other work override this.
        """
        prompt = self._build_prompt(ps)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(prompt)
                ps.metrics.total_tokens_used += response.total_tokens
                ps.metrics.total_llm_calls   += 1

                self._parse_response(ps, response.text)
                break

            except Exception as exc:
                logger.warning(
                    "[%s] attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    logger.error("[%s] all retries exhausted.", self.name)
                    ps.log_error(self.name, str(exc))
                else:
                    time.sleep(self.backoff * attempt)


    # ── JSON helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) and return inner content."""
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        return fenced.group(1) if fenced else text

    @staticmethod
    def _extract_raw_json(text: str, bracket: str = "{") -> str:
        """
        Extract the first balanced JSON object or array from *text*.

        Uses a bracket-counting scan instead of a greedy regex so that
        `{"a":1} some trailing text {"b":2}` correctly returns `{"a":1}`.
        """
        open_b  = bracket
        close_b = "}" if bracket == "{" else "]"
        start   = text.find(open_b)
        if start == -1:
            kind = "object" if bracket == "{" else "array"
            raise ValueError(f"No JSON {kind} found in response:\n{text[:300]}")

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_b:
                depth += 1
            elif ch == close_b:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # Unbalanced — return from start to end and let the parser try
        return text[start:]

    @staticmethod
    def _try_parse(raw: str, expect_type: type):
        """
        Try json.loads first, then json_repair.loads.
        Returns the parsed object or raises ValueError.
        """
        import json_repair

        # 1. Strict parse
        try:
            result = json.loads(raw)
            if isinstance(result, expect_type):
                return result
        except Exception:
            pass

        # 2. json_repair
        try:
            result = json_repair.loads(raw)
            if isinstance(result, expect_type):
                return result
        except Exception:
            pass

        type_name = "dict" if expect_type is dict else "list"
        raise ValueError(f"Failed to parse JSON as {type_name}:\n{raw[:300]}")

    # ── Public extraction methods ─────────────────────────────────────

    @classmethod
    def extract_json(cls, text: str) -> dict[str, Any]:
        """
        Safely extract the first JSON object from a model response.
        Handles markdown code fences (```json ... ```) gracefully.
        """
        text = cls._strip_fences(text)
        raw  = cls._extract_raw_json(text, "{")
        return deep_unescape(cls._try_parse(raw, dict))

    @classmethod
    def extract_json_list(cls, text: str) -> list[Any]:
        """
        Safely extract the first JSON array from a model response.
        """
        text = cls._strip_fences(text)
        raw  = cls._extract_raw_json(text, "[")
        return deep_unescape(cls._try_parse(raw, list))

    # ── Validation helpers ────────────────────────────────────────────

    @staticmethod
    def _validate_data(data: Any, schema_cls: Any, is_list: bool) -> Any:
        """
        Run Pydantic validation on parsed data.
        Returns the validated/coerced Python object(s).
        """
        if is_list:
            if not isinstance(data, list):
                raise ValueError("Expected a list, got " + str(type(data)))
            validated = []
            for item in data:
                if hasattr(schema_cls, "model_validate"):
                    obj = schema_cls.model_validate(item)
                else:
                    obj = schema_cls(**item)
                validated.append(obj.model_dump() if hasattr(obj, "model_dump") else item)
            return validated
        else:
            if not isinstance(data, dict):
                raise ValueError("Expected a dictionary, got " + str(type(data)))
            if hasattr(schema_cls, "model_validate"):
                obj = schema_cls.model_validate(data)
            else:
                obj = schema_cls(**data)
            return obj.model_dump() if hasattr(obj, "model_dump") else data

    def _repair_json_via_llm(
        self,
        json_fragment: str,
        schema_cls: Any,
        state: Optional[ProjectState] = None,
        lock: Optional[Any] = None,
    ) -> str:
        """
        Send **only** the JSON fragment (not the full LLM prose) to the
        model for repair, together with the expected schema so the model
        knows what structure to preserve.
        """
        schema_hint = ""
        if hasattr(schema_cls, "model_json_schema"):
            try:
                schema_hint = (
                    "\n\nExpected JSON schema:\n"
                    + json.dumps(schema_cls.model_json_schema(), indent=2)
                )
            except Exception:
                pass

        repair_prompt = (
            "The following JSON is invalid or does not match the expected schema.\n\n"
            "Repair it so it becomes valid JSON matching the schema.\n\n"
            "Rules:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT add new information.\n"
            "- Only fix syntax errors and missing/misnamed fields.\n"
            "- Return ONLY the corrected JSON, no explanation.\n"
            f"{schema_hint}\n\n"
            f"{json_fragment}"
        )

        logger.info("[%s] Sending JSON fragment to LLM for repair…", self.name)
        try:
            raw_text = self._call_with_system(
                system="You are a helpful assistant that repairs JSON syntax.",
                user=repair_prompt,
                agent_name=f"{self.name}-repair",
                state=state,
                lock=lock,
            )
            return raw_text
        except Exception as exc:
            logger.error("[%s] LLM JSON repair call failed: %s", self.name, exc)
            return json_fragment

    def extract_json_and_validate(
        self,
        text: str,
        schema_cls: Any,
        is_list: bool = False,
        state: Optional[ProjectState] = None,
        lock: Optional[Any] = None,
    ) -> Any:
        """
        Three-stage extraction pipeline:

        1. **Parse + validate** — json.loads / json_repair on the raw text.
        2. **Local repair + validate** — re-run json_repair on just the
           extracted substring (handles truncated brackets, trailing commas).
        3. **LLM repair + validate** — send the JSON fragment (not the full
           response) to the model with the target schema for correction.

        Returns the validated (and pydantic-coerced) data.
        """
        bracket   = "[" if is_list else "{"
        extractor = self.extract_json_list if is_list else self.extract_json

        # ── Stage 1: direct parse ────────────────────────────────────
        try:
            parsed = extractor(text)
            return self._validate_data(parsed, schema_cls, is_list)
        except Exception as stage1_exc:
            logger.warning(
                "[%s] Stage 1 (direct parse) failed: %s", self.name, stage1_exc
            )

        # ── Stage 2: local repair on raw substring ───────────────────
        raw_fragment = ""
        try:
            stripped = self._strip_fences(text)
            raw_fragment = self._extract_raw_json(stripped, bracket)
        except Exception:
            raw_fragment = text  # fallback — let json_repair try the whole thing

        import json_repair
        try:
            repaired_local = json_repair.loads(raw_fragment)
            if not isinstance(repaired_local, (dict, list)):
                raise ValueError("json_repair returned unexpected type")
            return self._validate_data(deep_unescape(repaired_local), schema_cls, is_list)
        except Exception as stage2_exc:
            logger.warning(
                "[%s] Stage 2 (local repair) failed: %s", self.name, stage2_exc
            )

        # ── Stage 3: LLM repair (send only the JSON fragment) ───────
        repaired_text = self._repair_json_via_llm(raw_fragment, schema_cls, state=state, lock=lock)
        try:
            parsed_repaired = extractor(repaired_text)
            result = self._validate_data(parsed_repaired, schema_cls, is_list)
            logger.info("[%s] Stage 3 (LLM repair) succeeded.", self.name)
            return result
        except Exception as stage3_exc:
            logger.error(
                "[%s] All 3 repair stages failed. Final error: %s",
                self.name, stage3_exc,
            )
            raise stage3_exc

########stream printer with thread safety########   
class LineBufferedPrinter:
            _print_lock = Lock()
            def __init__(self, prefix: str):
                self.prefix = prefix
                self.buffer = ""
            def __call__(self, chunk: str):
                self.buffer += chunk
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    with LineBufferedPrinter._print_lock:
                        sys.stdout.write(f"[{self.prefix}] {line}\n")
                        sys.stdout.flush()
            def flush(self):
                if self.buffer:
                    with LineBufferedPrinter._print_lock:
                        sys.stdout.write(f"[{self.prefix}] {self.buffer}\n")
                        sys.stdout.flush()
                    self.buffer = ""
