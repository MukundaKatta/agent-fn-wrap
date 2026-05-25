"""Tests for agent-fn-wrap."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from agent_fn_wrap import ToolRegistry, ToolWrapError


# ---------------------------------------------------------------------------
# ToolRegistry basics
# ---------------------------------------------------------------------------

def test_empty_registry():
    reg = ToolRegistry()
    assert len(reg) == 0
    assert reg.names() == []
    assert reg.schemas() == []

def test_register_with_decorator():
    reg = ToolRegistry()

    @reg.tool(description="Search the web.")
    def web_search(q: str, max_results: int = 5) -> str:
        return f"results for {q}"

    assert "web_search" in reg.names()
    assert len(reg) == 1

def test_register_no_args_decorator():
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        return "pong"

    assert reg.has("ping")

def test_register_explicit():
    reg = ToolRegistry()
    def my_fn(x: str) -> str: return x
    reg.register(my_fn, description="Test fn")
    assert reg.has("my_fn")

def test_has():
    reg = ToolRegistry()
    @reg.tool
    def fn(): pass
    assert reg.has("fn") is True
    assert reg.has("missing") is False


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

def test_schema_structure():
    reg = ToolRegistry()
    @reg.tool(description="Do something.")
    def my_tool(x: str) -> str:
        return x
    schemas = reg.schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["name"] == "my_tool"
    assert s["description"] == "Do something."
    assert "input_schema" in s
    assert s["input_schema"]["type"] == "object"

def test_schema_required_param():
    reg = ToolRegistry()
    @reg.tool
    def fn(q: str) -> str: return q
    s = reg.schemas()[0]
    assert "q" in s["input_schema"]["required"]

def test_schema_optional_param():
    reg = ToolRegistry()
    @reg.tool
    def fn(q: str, limit: int = 10) -> str: return q
    s = reg.schemas()[0]
    assert "q" in s["input_schema"]["required"]
    assert "limit" not in s["input_schema"].get("required", [])

def test_schema_string_type():
    reg = ToolRegistry()
    @reg.tool
    def fn(q: str) -> str: return q
    props = reg.schemas()[0]["input_schema"]["properties"]
    assert props["q"]["type"] == "string"

def test_schema_int_type():
    reg = ToolRegistry()
    @reg.tool
    def fn(n: int) -> str: return str(n)
    props = reg.schemas()[0]["input_schema"]["properties"]
    assert props["n"]["type"] == "integer"

def test_schema_float_type():
    reg = ToolRegistry()
    @reg.tool
    def fn(x: float) -> str: return str(x)
    props = reg.schemas()[0]["input_schema"]["properties"]
    assert props["x"]["type"] == "number"

def test_schema_bool_type():
    reg = ToolRegistry()
    @reg.tool
    def fn(flag: bool = False) -> str: return str(flag)
    props = reg.schemas()[0]["input_schema"]["properties"]
    assert props["flag"]["type"] == "boolean"

def test_schema_no_annotation():
    reg = ToolRegistry()
    @reg.tool
    def fn(x) -> str: return str(x)
    props = reg.schemas()[0]["input_schema"]["properties"]
    assert props["x"]["type"] == "string"  # defaults to string

def test_schema_default_value():
    reg = ToolRegistry()
    @reg.tool
    def fn(limit: int = 10) -> str: return str(limit)
    props = reg.schemas()[0]["input_schema"]["properties"]
    assert props["limit"]["default"] == 10

def test_schema_multiple_params():
    reg = ToolRegistry()
    @reg.tool
    def fn(q: str, limit: int = 5, verbose: bool = False) -> str: return q
    s = reg.schemas()[0]
    props = s["input_schema"]["properties"]
    assert "q" in props
    assert "limit" in props
    assert "verbose" in props
    assert s["input_schema"]["required"] == ["q"]

def test_description_from_docstring():
    reg = ToolRegistry()
    @reg.tool
    def fn(q: str) -> str:
        """Search the web for information."""
        return q
    assert reg.schemas()[0]["description"] == "Search the web for information."

def test_explicit_description_overrides_docstring():
    reg = ToolRegistry()
    @reg.tool(description="Custom description.")
    def fn(q: str) -> str:
        """Docstring."""
        return q
    assert reg.schemas()[0]["description"] == "Custom description."

def test_custom_name():
    reg = ToolRegistry()
    @reg.tool(name="search", description="Search.")
    def web_search_impl(q: str) -> str: return q
    assert reg.has("search")
    assert not reg.has("web_search_impl")


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------

def test_call_returns_result():
    reg = ToolRegistry()
    @reg.tool(description="Add two numbers.")
    def add(a: int, b: int) -> int:
        return a + b
    result = reg.call("add", {"a": 3, "b": 4})
    assert result == 7

def test_call_with_default():
    reg = ToolRegistry()
    @reg.tool
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"
    result = reg.call("greet", {"name": "World"})
    assert result == "Hello, World!"

def test_call_missing_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolWrapError, match="no tool registered"):
        reg.call("nonexistent", {})

def test_call_passes_all_args():
    reg = ToolRegistry()
    calls = []
    @reg.tool
    def record(x: str, y: int = 0) -> None:
        calls.append((x, y))
    reg.call("record", {"x": "hi", "y": 42})
    assert calls == [("hi", 42)]


# ---------------------------------------------------------------------------
# original function still callable
# ---------------------------------------------------------------------------

def test_decorated_fn_still_works():
    reg = ToolRegistry()
    @reg.tool(description="Double.")
    def double(n: int) -> int:
        return n * 2
    assert double(5) == 10


# ---------------------------------------------------------------------------
# Multiple tools
# ---------------------------------------------------------------------------

def test_multiple_tools():
    reg = ToolRegistry()
    @reg.tool
    def fn_a(x: str) -> str: return x
    @reg.tool
    def fn_b(y: int) -> int: return y
    assert len(reg) == 2
    assert set(reg.names()) == {"fn_a", "fn_b"}
    assert len(reg.schemas()) == 2


# ---------------------------------------------------------------------------
# schema_override
# ---------------------------------------------------------------------------

def test_schema_override():
    reg = ToolRegistry()
    custom = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    @reg.tool(description="Custom schema.", schema_override=custom)
    def search(q): return q
    s = reg.schemas()[0]
    assert s["input_schema"] == custom
