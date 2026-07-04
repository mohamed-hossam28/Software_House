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

logger = logging.getLogger(__name__)


DEFAULT_MODEL="openai/gpt-oss-120b:free"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_RETRIES    = 3
DEFAULT_BACKOFF    = 2.0   # seconds, doubles each retry

class BaseAgent(ABC):
    def __init__(
        self,
        model:str=DEFAULT_MODEL,
        max_tokens:int=DEFAULT_MAX_TOKENS,
        max_retries:int=DEFAULT_RETRIES,
        backoff:float=DEFAULT_BACKOFF,
        temperature:float=0.2,
        provider:    BaseLLMProvider | None = None, 
        
    ):
        """Initialize the base agent."""
        self.model=model
        self.max_tokens=max_tokens
        self.max_retries=max_retries
        self.backoff=backoff
        self.temperature=temperature
        # Extract the provider prefix from the model string (e.g. "openai/gpt-4o" → "openai")
        provider_name    = model.split("/")[0]
        self.provider    = provider or llmFactory.get(provider_name)
        

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
    def llm_call(self, prompt: str) -> LLMResponse:

        return self.provider.complete(
            system_prompt = self.system_prompt,
            user_message  = prompt,
            model         = self.model,
            max_tokens    = self.max_tokens,
            temperature   = self.temperature,
        )


    def run(self,state:ProjectState)->ProjectState:
        """Run the agent."""
        state.mark_agent_start(self.name)
        prompt=self._build_prompt(state)
        t0=time.perf_counter()

        for attempt in range(1,self.max_retries+1):
            try:
                response =self.llm_call(prompt)
                state.metrics.total_tokens_used += response.total_tokens
                state.metrics.total_llm_calls   += 1
 
                state = self._parse_response(state, response.text)
                break

            except Exception as exc:
                logger.warning(
                    "[%s] attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt==self.max_retries:
                    logger.error("[%s] all retries exhausted.", self.name)
                    state.log_error(self.name,str(exc))
                    
                else:
                    time.sleep(self.backoff*attempt)

        duration = time.perf_counter() - t0
        state.metrics.agent_durations[self.name] = round(duration, 3)
        logger.info("[%s] finished in %.2fs via %s", self.name, duration, self.provider.provider_name)
        return state


        # ── Parsing helpers ────────────────────────
 
    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """
        Safely extract the first JSON object from a model response.
        Handles markdown code fences (```json ... ```) gracefully.
        """
        # Strip markdown fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fenced:
            text = fenced.group(1)
 
        # Find first {...}
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"No JSON object found in response:\n{text[:300]}")
 
        return json.loads(match.group())
 
    @staticmethod
    def extract_json_list(text: str) -> list[Any]:
        """
        Safely extract the first JSON array from a model response.
        """
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fenced:
            text = fenced.group(1)
 
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError(f"No JSON array found in response:\n{text[:300]}")
 
        return json.loads(match.group())
 
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
 

                    
