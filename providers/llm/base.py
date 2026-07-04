from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class LLMResponse:
    text: str                           # the model's reply
    input_tokens: int  = 0             # tokens in the prompt
    output_tokens: int = 0             # tokens in the reply
    model: str         = ""            # which model actually ran
    provider: str      = ""            # which provider was used
 
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message:  str,
        model:         str,
        max_tokens:    int   = 8192,
        temperature:   float = 0.2,
    )->LLMResponse:
        pass 