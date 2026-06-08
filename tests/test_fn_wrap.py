"""Tests for agent-fn-wrap (standard-library ``unittest`` only).

Run from the repository root with::

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest
from typing import Optional, Union

# Make the in-tree package importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_fn_wrap import (  # noqa: E402
    ToolRegistry,
    ToolWrapError,
    WrappedTool,
    call_tool,
    get_tools,
    reset_default_registry,
    tool,
)


class RegistryBasicsTest(unittest.TestCase):
    def test_empty_registry(self):
        reg = ToolRegistry()
        self.assertEqual(len(reg), 0)
        self.assertEqual(reg.names(), [])
        self.assertEqual(reg.schemas(), [])

    def test_register_with_decorator(self):
        reg = ToolRegistry()

        @reg.tool(description="Search the web.")
        def web_search(q: str, max_results: int = 5) -> str:
            return f"results for {q}"

        self.assertIn("web_search", reg.names())
        self.assertEqual(len(reg), 1)

    def test_register_no_args_decorator(self):
        reg = ToolRegistry()

        @reg.tool
        def ping() -> str:
            return "pong"

        self.assertTrue(reg.has("ping"))

    def test_register_explicit(self):
        reg = ToolRegistry()

        def my_fn(x: str) -> str:
            return x

        returned = reg.register(my_fn, description="Test fn")
        self.assertTrue(reg.has("my_fn"))
        # register() returns the registry for chaining.
        self.assertIs(returned, reg)

    def test_has(self):
        reg = ToolRegistry()

        @reg.tool
        def fn():
            pass

        self.assertTrue(reg.has("fn"))
        self.assertFalse(reg.has("missing"))

    def test_repr(self):
        reg = ToolRegistry()

        @reg.tool
        def fn():
            pass

        self.assertIn("fn", repr(reg))


class SchemaGenerationTest(unittest.TestCase):
    def test_schema_structure(self):
        reg = ToolRegistry()

        @reg.tool(description="Do something.")
        def my_tool(x: str) -> str:
            return x

        schemas = reg.schemas()
        self.assertEqual(len(schemas), 1)
        s = schemas[0]
        self.assertEqual(s["name"], "my_tool")
        self.assertEqual(s["description"], "Do something.")
        self.assertIn("input_schema", s)
        self.assertEqual(s["input_schema"]["type"], "object")

    def test_schema_required_param(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(q: str) -> str:
            return q

        s = reg.schemas()[0]
        self.assertIn("q", s["input_schema"]["required"])

    def test_schema_optional_param_not_required(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(q: str, limit: int = 10) -> str:
            return q

        s = reg.schemas()[0]
        self.assertIn("q", s["input_schema"]["required"])
        self.assertNotIn("limit", s["input_schema"].get("required", []))

    def test_scalar_type_mappings(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(a: str, b: int, c: float, d: bool = False):
            return None

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["a"]["type"], "string")
        self.assertEqual(props["b"]["type"], "integer")
        self.assertEqual(props["c"]["type"], "number")
        self.assertEqual(props["d"]["type"], "boolean")

    def test_no_annotation_defaults_to_string(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(x) -> str:
            return str(x)

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["x"]["type"], "string")

    def test_default_value_in_schema(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(limit: int = 10) -> str:
            return str(limit)

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["limit"]["default"], 10)

    def test_no_required_key_when_all_optional(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(a: int = 1, b: int = 2):
            return a + b

        schema = reg.schemas()[0]["input_schema"]
        self.assertNotIn("required", schema)


class TypeMappingEdgeCaseTest(unittest.TestCase):
    """Regression tests for optional/union and parameterized generics."""

    def test_optional_int_maps_to_integer(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(n: Optional[int] = None):
            return n

        props = reg.schemas()[0]["input_schema"]["properties"]
        # Previously these mapped to "string" due to a broken union check.
        self.assertEqual(props["n"]["type"], "integer")

    def test_pep604_optional_maps_to_underlying_type(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(n: "int | None" = None, s: "str | None" = None):
            return n

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["n"]["type"], "integer")
        self.assertEqual(props["s"]["type"], "string")

    def test_union_picks_first_non_none(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(x: Union[bool, int]):
            return x

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["x"]["type"], "boolean")

    def test_list_maps_to_array(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(items: list):
            return items

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["items"]["type"], "array")
        # A bare list has no element type, so no items key.
        self.assertNotIn("items", props["items"])

    def test_typed_list_emits_items(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(tags: list[str], counts: list[int]):
            return tags

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["tags"]["type"], "array")
        self.assertEqual(props["tags"]["items"], {"type": "string"})
        self.assertEqual(props["counts"]["items"], {"type": "integer"})

    def test_dict_maps_to_object(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(meta: dict[str, int]):
            return meta

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["meta"]["type"], "object")

    def test_unknown_type_falls_back_to_string(self):
        reg = ToolRegistry()

        class Custom:
            pass

        @reg.tool
        def fn(x: Custom = None):
            return x

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertEqual(props["x"]["type"], "string")


class VariadicAndMethodTest(unittest.TestCase):
    def test_var_args_and_kwargs_skipped(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(q: str, *args, **kwargs):
            return q

        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertIn("q", props)
        self.assertNotIn("args", props)
        self.assertNotIn("kwargs", props)

    def test_self_is_skipped(self):
        reg = ToolRegistry()

        class Handlers:
            def lookup(self, key: str) -> str:
                return key

        reg.register(Handlers().lookup, name="lookup")
        props = reg.schemas()[0]["input_schema"]["properties"]
        self.assertIn("key", props)
        self.assertNotIn("self", props)


class DescriptionTest(unittest.TestCase):
    def test_description_from_docstring_first_line(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(q: str) -> str:
            """Search the web for information.

            Extra detail that should be ignored.
            """
            return q

        self.assertEqual(
            reg.schemas()[0]["description"], "Search the web for information."
        )

    def test_explicit_description_overrides_docstring(self):
        reg = ToolRegistry()

        @reg.tool(description="Custom description.")
        def fn(q: str) -> str:
            """Docstring."""
            return q

        self.assertEqual(reg.schemas()[0]["description"], "Custom description.")

    def test_missing_docstring_gives_empty_description(self):
        reg = ToolRegistry()

        @reg.tool
        def fn(q: str) -> str:
            return q

        self.assertEqual(reg.schemas()[0]["description"], "")


class NamingTest(unittest.TestCase):
    def test_custom_name(self):
        reg = ToolRegistry()

        @reg.tool(name="search", description="Search.")
        def web_search_impl(q: str) -> str:
            return q

        self.assertTrue(reg.has("search"))
        self.assertFalse(reg.has("web_search_impl"))

    def test_same_name_overwrites(self):
        reg = ToolRegistry()

        @reg.tool(name="dup")
        def a(x: str) -> str:
            return "a"

        @reg.tool(name="dup")
        def b(x: str) -> str:
            return "b"

        self.assertEqual(len(reg), 1)
        self.assertEqual(reg.call("dup", {"x": "hi"}), "b")


class CallTest(unittest.TestCase):
    def test_call_returns_result(self):
        reg = ToolRegistry()

        @reg.tool(description="Add two numbers.")
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(reg.call("add", {"a": 3, "b": 4}), 7)

    def test_call_with_default(self):
        reg = ToolRegistry()

        @reg.tool
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        self.assertEqual(reg.call("greet", {"name": "World"}), "Hello, World!")

    def test_call_missing_tool_raises(self):
        reg = ToolRegistry()
        with self.assertRaises(ToolWrapError):
            reg.call("nonexistent", {})

    def test_call_passes_all_args(self):
        reg = ToolRegistry()
        calls = []

        @reg.tool
        def record(x: str, y: int = 0) -> None:
            calls.append((x, y))

        reg.call("record", {"x": "hi", "y": 42})
        self.assertEqual(calls, [("hi", 42)])

    def test_decorated_fn_still_callable_directly(self):
        reg = ToolRegistry()

        @reg.tool(description="Double.")
        def double(n: int) -> int:
            return n * 2

        self.assertEqual(double(5), 10)


class MultipleToolsTest(unittest.TestCase):
    def test_multiple_tools(self):
        reg = ToolRegistry()

        @reg.tool
        def fn_a(x: str) -> str:
            return x

        @reg.tool
        def fn_b(y: int) -> int:
            return y

        self.assertEqual(len(reg), 2)
        self.assertEqual(set(reg.names()), {"fn_a", "fn_b"})
        self.assertEqual(len(reg.schemas()), 2)


class SchemaOverrideTest(unittest.TestCase):
    def test_schema_override(self):
        reg = ToolRegistry()
        custom = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }

        @reg.tool(description="Custom schema.", schema_override=custom)
        def search(q):
            return q

        s = reg.schemas()[0]
        self.assertEqual(s["input_schema"], custom)


class WrappedToolTest(unittest.TestCase):
    def test_wrapped_tool_schema_shape(self):
        def fn(q: str) -> str:
            return q

        wt = WrappedTool(fn, description="d")
        schema = wt.schema()
        self.assertEqual(
            set(schema), {"name", "description", "input_schema"}
        )
        self.assertEqual(schema["name"], "fn")

    def test_wrapped_tool_call_and_invoke(self):
        def fn(a: int, b: int) -> int:
            return a + b

        wt = WrappedTool(fn)
        self.assertEqual(wt.call({"a": 1, "b": 2}), 3)
        # __call__ forwards positionally to the underlying function.
        self.assertEqual(wt(1, 2), 3)

    def test_repr_contains_name(self):
        wt = WrappedTool(lambda: None, name="ping")
        self.assertIn("ping", repr(wt))


class ModuleLevelApiTest(unittest.TestCase):
    def setUp(self):
        reset_default_registry()

    def tearDown(self):
        reset_default_registry()

    def test_global_tool_and_get_tools(self):
        @tool(description="Echo input.")
        def echo(text: str) -> str:
            return text

        schemas = get_tools()
        names = [s["name"] for s in schemas]
        self.assertIn("echo", names)

    def test_global_call_tool(self):
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(call_tool("add", {"a": 2, "b": 5}), 7)

    def test_global_tool_with_schema_override(self):
        custom = {"type": "object", "properties": {}, "required": []}

        @tool(description="Custom.", schema_override=custom)
        def fn():
            return None

        self.assertEqual(get_tools()[0]["input_schema"], custom)

    def test_reset_clears_registry(self):
        @tool
        def temp(x: str) -> str:
            return x

        self.assertTrue(any(s["name"] == "temp" for s in get_tools()))
        reset_default_registry()
        self.assertEqual(get_tools(), [])


if __name__ == "__main__":
    unittest.main()
