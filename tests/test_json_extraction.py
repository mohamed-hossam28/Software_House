import os, sys
from dotenv import load_dotenv

workspace_dir = r"c:\Users\bebawy1\Desktop\Software_House"
sys.path.insert(0, workspace_dir)
load_dotenv(os.path.join(workspace_dir, ".env"))

from agents.base import BaseAgent

# Check all methods exist
for m in ["extract_json", "extract_json_list", "extract_json_and_validate",
          "_strip_fences", "_extract_raw_json", "_try_parse", "_repair_json_via_llm"]:
    print(f"  {m}: {hasattr(BaseAgent, m)}")

# Test 1: fenced JSON
t1 = BaseAgent.extract_json('```json\n{"a": 1, "b": [2,3]}\n```\nsome text')
assert t1 == {"a": 1, "b": [2, 3]}, f"Test 1 failed: {t1}"
print("Test 1 (fenced object): OK")

# Test 2: multiple objects — should return the FIRST one
t2 = BaseAgent.extract_json('Here is: {"x": 1} and {"y": 2}')
assert t2 == {"x": 1}, f"Test 2 failed: {t2}"
print("Test 2 (first of two objects): OK")

# Test 3: list extraction
t3 = BaseAgent.extract_json_list('```json\n[{"a":1},{"b":2}]\n```')
assert t3 == [{"a": 1}, {"b": 2}], f"Test 3 failed: {t3}"
print("Test 3 (fenced list): OK")

# Test 4: trailing comma (json_repair should fix)
t4 = BaseAgent.extract_json('{"key": "val",}')
assert t4 == {"key": "val"}, f"Test 4 failed: {t4}"
print("Test 4 (trailing comma repair): OK")

# Test 5: nested objects
t5 = BaseAgent.extract_json('{"outer": {"inner": [1,2,3]}, "x": true}')
assert t5 == {"outer": {"inner": [1, 2, 3]}, "x": True}, f"Test 5 failed: {t5}"
print("Test 5 (nested): OK")

print("\nAll tests passed!")


# ── pytest section: deep unescape (double-encoded LLM output) ─────────────────

def test_deep_unescape_single_encoded_passthrough():
    from agents.base import deep_unescape
    data = {"code": "def f():\n    pass", "name": "plain", "items": [1, 2]}
    assert deep_unescape(data) == data


def test_deep_unescape_double_encoded_code_value():
    import json as _json
    from agents.base import deep_unescape
    payload = '{"code": "\\"from db import X\\n\\nclass A: pass\\""}'
    data = _json.loads(payload)
    assert data["code"] == '"from db import X\n\nclass A: pass"'
    fixed = deep_unescape(data)
    assert fixed["code"] == "from db import X\n\nclass A: pass"


def test_deep_unescape_does_not_touch_plain_strings():
    from agents.base import deep_unescape
    assert deep_unescape('"use strict"') == '"use strict"'
    assert deep_unescape("'single'") == "'single'"
    assert deep_unescape("return 'x'") == "return 'x'"


def test_extract_json_unescapes_double_encoded_code():
    import json as _json
    resp = '{"code": "\\"def add():\\n    return 1\\""}'
    out = BaseAgent.extract_json(resp)
    assert out["code"] == "def add():\n    return 1"


def test_parse_batch_response_unescapes_double_encoded_items():
    from agents.base import BaseAgent as BA

    class _Stub(BA):
        def system_prompt(self) -> str:
            return ""
        def __init__(self, *args, **kwargs):
            object.__init__(self)

    resp = '[{"filepath": "a.py", "code": "\\"x = 1\\""}, {"filepath": "b.py", "code": "y = 2"}]'
    items = _Stub()._parse_batch_response(resp)
    assert items[0]["code"] == "x = 1"
    assert items[1]["code"] == "y = 2"


def test_parse_metadata_handles_bad_manifest_values():
    from agents.utils import parse_metadata

    # Null table_name must not raise (previously crashed DatabaseAgent tasks).
    classes, functions = parse_metadata({"manifest": {"table_name": None, "columns": []}})
    assert classes == [] and functions == []

    # Non-string table_name (e.g. a dict) must be skipped, not crash.
    classes2, _ = parse_metadata({"manifest": {"table_name": {"name": "X"}, "columns": [{"name": "id"}]}})
    assert classes2 == []

    # Non-dict column entries are skipped; valid ones still parsed.
    classes3, _ = parse_metadata({
        "manifest": {"table_name": "users",
                     "columns": ["bad", {"name": "id", "type": "int"}]}})
    assert len(classes3) == 1
    assert classes3[0].name == "users"
    assert [a.name for a in classes3[0].arguments] == ["id"]
