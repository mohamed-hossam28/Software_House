"""LLM provider backed by the local Claude Code CLI (`claude -p`).

Runs each completion as a headless Claude Code call, so it uses the CLI's own
login (a Claude subscription) instead of an API key. Meant for local testing
of the pipeline; each call spawns a process, so expect a few seconds of
overhead per call.

The prompt goes in on stdin and the system prompt through a temp file, so
long prompts never hit the OS command-line length limit. Tools and setting
sources are disabled and the CLI runs in an empty temp directory, so the call
behaves like a plain completion: no file access, no CLAUDE.md, no hooks.
"""

import json
import os
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

from providers.llm.base import BaseLLMProvider, LLMResponse

DEFAULT_TIMEOUT = 900.0


class ClaudeCodeError(RuntimeError):
    pass


class ClaudeCodeProvider(BaseLLMProvider):

    def __init__(self, api_key: str | None = None, timeout: float | None = None):
        # api_key is accepted for factory compatibility and ignored: the CLI
        # authenticates with its own login.
        self.executable = os.environ.get("CLAUDE_CODE_BIN") or shutil.which("claude")
        if not self.executable:
            raise ClaudeCodeError(
                "Claude Code CLI not found on PATH. Install it or set CLAUDE_CODE_BIN."
            )
        self.timeout = timeout or float(os.environ.get("CLAUDE_CODE_TIMEOUT", DEFAULT_TIMEOUT))

    @property
    def provider_name(self) -> str:
        return "claude_code"

    def build_command(self, model: str, system_prompt_file: str) -> list[str]:
        return [
            self.executable, "-p",
            "--output-format", "json",
            "--model", model,
            "--system-prompt-file", system_prompt_file,
            "--tools", "",
            "--setting-sources", "",
        ]

    def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str   = "sonnet",
        max_tokens:    int   = 8192,
        temperature:   float = 0.2,
        stream:        bool  = False,
        chunk_callback: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        # max_tokens and temperature have no CLI equivalent and are ignored.
        with tempfile.TemporaryDirectory(prefix="claude_code_llm_") as workdir:
            prompt_file = os.path.join(workdir, "system_prompt.txt")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(system_prompt or "You are a helpful assistant.")

            try:
                proc = subprocess.run(
                    self.build_command(model, prompt_file),
                    input=user_message,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=workdir,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise ClaudeCodeError(f"claude -p timed out after {self.timeout}s") from e

        payload = self.parse_output(proc)
        text = payload.get("result") or ""
        if stream and chunk_callback and text:
            chunk_callback(text)

        usage = payload.get("usage") or {}
        model_usage = payload.get("modelUsage") or {}
        return LLMResponse(
            text          = text,
            input_tokens  = (usage.get("input_tokens", 0)
                             + usage.get("cache_creation_input_tokens", 0)
                             + usage.get("cache_read_input_tokens", 0)),
            output_tokens = usage.get("output_tokens", 0),
            model         = next(iter(model_usage), model),
            provider      = self.provider_name,
        )

    @staticmethod
    def parse_output(proc: subprocess.CompletedProcess) -> dict:
        try:
            payload = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError):
            detail = (proc.stderr or proc.stdout or "").strip()[:2000]
            raise ClaudeCodeError(
                f"claude -p exited {proc.returncode} without JSON output: {detail}"
            )
        if payload.get("is_error") or proc.returncode != 0:
            raise ClaudeCodeError(
                f"claude -p failed ({payload.get('subtype')}): {payload.get('result')}"
            )
        return payload
