from providers.llm.base import LLMResponse
import time
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from providers.llm.base import BaseLLMProvider
from providers.llm.llmFactory import LLMFactory as llmFactory

from core.state import ProjectState
import os
from difflib import SequenceMatcher


logger = logging.getLogger(__name__)


from core.config import settings

DEFAULT_MODEL = settings.model_name
DEFAULT_MAX_TOKENS = settings.default_max_tokens or 8192
DEFAULT_RETRIES    = settings.default_retries or 3
DEFAULT_BACKOFF    = settings.default_backoff or 2.0 

####################Filters####################

def filter_manifest_by_dependencies(
    manifest: Any,
    filepath: str,
    task: Any = None,
) -> dict[str, Any]:
    """
    Filter CodeManifest to only include files that the given file/task actually depends on.
    """
    if not hasattr(manifest, "files") or not manifest.files:
        return {}

    fm = manifest.files.get(filepath)
    if not fm:
        return {}

    dep_task_ids = set(task.depends_on) if (task and hasattr(task, "depends_on") and task.depends_on) else set()

    filtered: dict[str, Any] = {}

    if dep_task_ids:
        for path, file_m in manifest.files.items():
            if path == filepath:
                continue
            if getattr(file_m, "task_id", None) in dep_task_ids or path in dep_task_ids:
                filtered[path] = file_m

    if not filtered:
        target_token = os.path.basename(filepath).split('.')[0].lower()
        for path, file_m in manifest.files.items():
            if path == filepath:
                continue
            path_token = os.path.basename(path).split('.')[0].lower()
            if getattr(file_m, "written", False) or (path_token and (path_token in target_token or target_token in path_token)):
                filtered[path] = file_m

    return filtered


def filter_api_contracts_by_relevance(
    contracts: list[Any],
    filepath: str,
    purpose: str = "",
    task_title: str = "",
    top_k: int = 3,
) -> list[Any]:
    """Filter and rank API contracts by relevance score."""
    
    if not contracts:
        return []
    
    if len(contracts) <= top_k:
        return contracts
    
    search_text = f"{filepath} {purpose} {task_title}".lower()
    
    def score_contract(contract: Any) -> float:
        if isinstance(contract, dict):
            c_text = f"{contract.get('path', '')} {contract.get('description', '')}".lower()
        else:
            c_text = f"{getattr(contract, 'path', '')} {getattr(contract, 'description', '')}".lower()
        
        # Use sequence matching for fuzzy similarity
        return SequenceMatcher(None, search_text, c_text).ratio()
    
    # Compute score once per contract
    scored = [(score_contract(c), c) for c in contracts]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K with score > threshold
    threshold = 0.35
    filtered = [c for score, c in scored if score >= threshold]
    
    return filtered[:top_k] if filtered else [c for _, c in scored[:top_k]]


# ── Trivial-file classifier ───────────────────────────────────────────────────

#: Filename patterns that are almost always trivial (< 20 lines, no bespoke logic)
_TRIVIAL_NAMES = frozenset({
    "__init__.py",
    "index.ts", "index.tsx", "index.js", "index.jsx",  # barrel / re-export files
    "types.ts", "types.tsx",                            # pure type declarations
    "constants.ts", "constants.js",                     # constant modules
    "config.py", "config.ts", "config.js",              # app config stubs
    "settings.py",
    "exceptions.py", "errors.py", "errors.ts",          # exception/error stubs
    "utils.py",                                         # small utility stubs
})

_TRIVIAL_PATTERNS = (
    "/__init__.py",
    "/index.",          # barrel files under any directory
    "/types.",
    "/constants.",
    "/exceptions.",
    "/errors.",
    "/config.",
    "/settings.",
)

TRIVIAL_BATCH_SIZE = 3  # max files per batch LLM call


def is_trivial_file(task: Any, manifest: Any) -> bool:
    """
    Returns True if a task maps to a file that is "trivial":
      - A known boilerplate pattern (init, barrel, config, types, constants, errors)
      - OR fewer than 3 planned exports in the manifest
    Trivial files will be batched together into a single LLM call.
    """
    if not hasattr(manifest, "files"):
        return False
    fm = getattr(manifest, "by_task", lambda _: None)(task.id)
    if fm is None:
        return False

    filepath = getattr(fm, "filepath", "")
    exports  = getattr(fm, "exports", []) or []
    basename = os.path.basename(filepath).lower()

    # Explicit name match
    if basename in _TRIVIAL_NAMES:
        return True
    # Suffix pattern match
    for pattern in _TRIVIAL_PATTERNS:
        if pattern in filepath.lower():
            return True
    # Export count
    if len(exports) < 3:
        return True
    return False


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
    ):
        """Initialize the base agent."""
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

    @abstractmethod
    def _build_prompt(self,state:ProjectState)->str:
        """Build the prompt for the agent."""
        ...

    @abstractmethod
    def _parse_response(self,state:ProjectState,raw:str)->ProjectState:
        """Parse the response from the agent."""
        ...
    


    ##LLM CALL
    def llm_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> LLMResponse:
        import sys
        from threading import Lock

        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        prefix = agent_name if agent_name is not None else self.name

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

        printer = LineBufferedPrinter(prefix)

        try:
            response = self.provider.complete(
                system_prompt = sys_prompt,
                user_message  = prompt,
                model         = self.model,
                max_tokens    = self.max_tokens,
                temperature   = self.temperature,
                stream        = True,
                chunk_callback = printer,
            )
        except TypeError as e:
            if "unexpected keyword argument" in str(e) or "complete() got" in str(e):
                response = self.provider.complete(
                    system_prompt = sys_prompt,
                    user_message  = prompt,
                    model         = self.model,
                    max_tokens    = self.max_tokens,
                    temperature   = self.temperature,
                )
            else:
                raise
        printer.flush()
        return response


    @staticmethod
    def _compute_diff(before: ProjectState, after: ProjectState) -> dict[str, Any]:
        """
        Compare two ProjectState snapshots and return only the fields that changed as a dict.

        For Annotated/reducer fields (metrics, errors, current_agent), returns
        only the delta so LangGraph reducers merge correctly.
        """
        from core.state import PerformanceMetrics

        before_dict = before.model_dump(mode="json")
        after_dict = after.model_dump(mode="json")

        diff: dict[str, Any] = {}
        for key in after_dict:
            if key == "metrics":
                bm = before_dict["metrics"]
                am = after_dict["metrics"]
                dt = am["total_tokens_used"] - bm["total_tokens_used"]
                dc = am["total_llm_calls"] - bm["total_llm_calls"]
                nd = {k: v for k, v in am["agent_durations"].items()
                      if bm["agent_durations"].get(k) != v}
                if dt or dc or nd:
                    diff["metrics"] = PerformanceMetrics(
                        total_tokens_used=dt,
                        total_llm_calls=dc,
                        agent_durations=nd,
                    ).model_dump(mode="json")
                continue

            if key == "errors":
                new_errors = after_dict["errors"][len(before_dict["errors"]):]
                if new_errors:
                    diff["errors"] = new_errors
                continue

            if key == "current_agent":
                if after_dict["current_agent"] != before_dict["current_agent"]:
                    diff["current_agent"] = after_dict["current_agent"]
                continue

            if key == "completed_task_ids":
                b_ids = set(before_dict.get("completed_task_ids") or [])
                a_ids = set(after_dict.get("completed_task_ids") or [])
                new_ids = a_ids - b_ids
                if new_ids:
                    diff["completed_task_ids"] = list(new_ids)
                continue

            if before_dict[key] != after_dict[key]:
                diff[key] = after_dict[key]

        return diff


    def run(self, state: dict | ProjectState) -> dict[str, Any]:
        """
        Run the agent.
        Accepts the full state (dict or ProjectState) as input.
        Returns a dict containing ONLY the changed/wanted fields.
        """
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)

        ps.mark_agent_start(self.name)
        prompt = self._build_prompt(ps)
        t0 = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(prompt)
                ps.metrics.total_tokens_used += response.total_tokens
                ps.metrics.total_llm_calls   += 1

                ps = self._parse_response(ps, response.text)
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

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)
        logger.info("[%s] finished in %.2fs via %s", self.name, duration, self.provider.provider_name)
        return self._compute_diff(before, ps)


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
        return cls._try_parse(raw, dict)

    @classmethod
    def extract_json_list(cls, text: str) -> list[Any]:
        """
        Safely extract the first JSON array from a model response.
        """
        text = cls._strip_fences(text)
        raw  = cls._extract_raw_json(text, "[")
        return cls._try_parse(raw, list)

    @staticmethod
    def extract_section(text: str, tag: str) -> str:
        """
        Extract content between <tag> ... </tag> in the response.
        Useful when the model is prompted to use XML-style delimiters.
        """
        pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
        match   = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"<{tag}> section not found in response.")
        return match.group(1).strip()

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

    def _repair_json_via_llm(self, json_fragment: str, schema_cls: Any) -> str:
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
            response = self.provider.complete(
                system_prompt="You are a helpful assistant that repairs JSON syntax.",
                user_message=repair_prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,
            )
            return response.text
        except Exception as exc:
            logger.error("[%s] LLM JSON repair call failed: %s", self.name, exc)
            return json_fragment

    def extract_json_and_validate(
        self, text: str, schema_cls: Any, is_list: bool = False
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
            return self._validate_data(repaired_local, schema_cls, is_list)
        except Exception as stage2_exc:
            logger.warning(
                "[%s] Stage 2 (local repair) failed: %s", self.name, stage2_exc
            )

        # ── Stage 3: LLM repair (send only the JSON fragment) ───────
        repaired_text = self._repair_json_via_llm(raw_fragment, schema_cls)
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
