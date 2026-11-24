"""
The scripted fake LLM adapter: answers a Coding agent by file path (read from
the coding request header) and a single-call agent by call, and refuses
anything it was not scripted for.
"""
import json

import pytest

from agents.coding_request import request_header
from providers.llm.llmFactory import LLMFactory
from providers.llm.scripted import ScriptedLLMProvider, UnscriptedRequestError

ROUTER = "output/backend/app/api/routes/todo.py"
TYPES = "output/backend/app/types.py"
CONSTANTS = "output/backend/app/constants.py"


def ask(fake: ScriptedLLMProvider, user_message: str) -> str:
    return fake.complete(system_prompt="sys", user_message=user_message, model="m").text


def test_a_scripted_file_gets_its_scripted_reply():
    reply = {"code": "router = 1\n", "classes": [], "functions": []}
    fake = ScriptedLLMProvider("backend", files={ROUTER: reply})

    text = ask(fake, request_header([ROUTER]) + "## File to Generate\n...")

    assert json.loads(text) == reply


def test_a_batch_gets_one_reply_per_file_in_header_order_each_naming_its_file():
    fake = ScriptedLLMProvider("backend", files={
        TYPES: {"code": "T = 1\n"},
        CONSTANTS: {"code": "C = 1\n"},
    })

    items = json.loads(ask(fake, request_header([CONSTANTS, TYPES]) + "Generate the following 2 files."))

    assert items == [
        {"code": "C = 1\n", "filepath": CONSTANTS},
        {"code": "T = 1\n", "filepath": TYPES},
    ]


def test_an_unscripted_file_raises_naming_the_agent_and_the_file():
    fake = ScriptedLLMProvider("frontend", files={TYPES: {"code": "x"}})

    with pytest.raises(UnscriptedRequestError) as raised:
        ask(fake, request_header([TYPES, "output/frontend/index.html"]) + "...")

    assert "frontend" in str(raised.value)
    assert "output/frontend/index.html" in str(raised.value)


def test_single_call_agents_are_answered_call_by_call_and_refuse_an_extra_call():
    fake = ScriptedLLMProvider("planner", calls=[{"skeleton": []}, "not json at all"])

    assert json.loads(ask(fake, "## Requirements\n...")) == {"skeleton": []}
    assert ask(fake, "## Requirements\n...") == "not json at all"
    with pytest.raises(UnscriptedRequestError) as raised:
        ask(fake, "## Requirements\n...")

    assert "planner" in str(raised.value)
    assert "call 3" in str(raised.value)


def test_every_request_is_recorded_with_the_files_it_named():
    fake = ScriptedLLMProvider("backend", files={ROUTER: {"code": "x"}}, calls=["{}"])

    ask(fake, request_header([ROUTER]) + "body")
    fake.complete(system_prompt="repair system", user_message="fix this json", model="m")

    assert [(r.files, r.system_prompt) for r in fake.requests] == [
        ([ROUTER], "sys"),
        ([], "repair system"),
    ]
    assert fake.requests[0].user_message.endswith("body")


def test_configuration_cannot_select_the_scripted_fake():
    assert not any("scripted" in name for name in LLMFactory.available())
    with pytest.raises(ValueError):
        LLMFactory.get("scripted")
