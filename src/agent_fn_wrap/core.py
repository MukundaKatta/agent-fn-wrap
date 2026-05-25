"""Wrap Python functions as Anthropic tool schemas.

Inspects function signatures to generate tool input schemas automatically.
Zero dependencies — uses inspect and typing from the standard library.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints


class ToolWrapError(Exception):
    """Raised on invalid tool configuration."""


# Map Python types to JSON Schema types
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_DEFAULT_REGISTRY: "ToolRegistry | None" = None


def _get_default_registry() -> "ToolRegistry":
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry()
    return _DEFAULT_REGISTRY


def _json_type(annotation: Any) -> str:
    """Convert a Python type annotation to a JSON Schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    # Handle Optional[X] (Union[X, None]) from 3.10+
    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        return "string"
    # Union types — pick the first non-None
    if str(origin) in ("<class 'types.UnionType'>", "typing.Union"):
        args = [a for a in getattr(annotation, "__args__", []) if a is not type(None)]
        return _json_type(args[0]) if args else "string"
    return _TYPE_MAP.get(annotation, "string")


def _build_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build an Anthropic-style input_schema from a function signature."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, inspect.Parameter.empty)
        json_type = _json_type(annotation)
        prop: dict[str, Any] = {"type": json_type}

        # Use param's default as description hint if it's a string sentinel
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

        # Pull description from docstring param section if present
        properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class WrappedTool:
    """A function wrapped as an Anthropic tool."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str = "",
        schema_override: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        _doc_lines = (fn.__doc__ or "").strip().splitlines()
        self.description = description or (_doc_lines[0].strip() if _doc_lines else "")
        self._schema = schema_override or _build_schema(fn)

    def schema(self) -> dict[str, Any]:
        """Return the Anthropic-style tool dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._schema,
        }

    def call(self, input_data: dict[str, Any]) -> Any:
        """Call the wrapped function with the given input dict."""
        return self.fn(**input_data)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"WrappedTool(name={self.name!r})"


class ToolRegistry:
    """Registry of wrapped tools for an agent.

    Example::

        registry = ToolRegistry()

        @registry.tool(description="Search the web.")
        def web_search(q: str, max_results: int = 5) -> str:
            return search(q, max_results)

        schemas = registry.schemas()   # list of Anthropic tool dicts
        result = registry.call("web_search", {"q": "Paris"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, WrappedTool] = {}

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str = "",
        schema_override: dict[str, Any] | None = None,
    ) -> Any:
        """Register a function as a tool (decorator).

        Can be used with or without arguments::

            @registry.tool
            def ping() -> str: ...

            @registry.tool(description="Search the web.")
            def search(q: str) -> str: ...
        """
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            wrapped = WrappedTool(
                f,
                name=name,
                description=description,
                schema_override=schema_override,
            )
            self._tools[wrapped.name] = wrapped
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def register(self, fn: Callable[..., Any], **kwargs: Any) -> "ToolRegistry":
        """Explicitly register a function without using it as a decorator."""
        wrapped = WrappedTool(fn, **kwargs)
        self._tools[wrapped.name] = wrapped
        return self

    def schemas(self) -> list[dict[str, Any]]:
        """Return Anthropic-style tool schema list."""
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, input_data: dict[str, Any]) -> Any:
        """Call a registered tool by name.

        Raises:
            ToolWrapError: if no tool with that name is registered.
        """
        if name not in self._tools:
            raise ToolWrapError(f"no tool registered with name {name!r}")
        return self._tools[name].call(input_data)

    def names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools)

    def has(self, name: str) -> bool:
        """Return True if a tool with that name is registered."""
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.names()})"


# ---------------------------------------------------------------------------
# Module-level convenience API using a default global registry
# ---------------------------------------------------------------------------

def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
) -> Any:
    """Register a function as a tool in the default global registry."""
    return _get_default_registry().tool(fn, name=name, description=description)


def get_tools() -> list[dict[str, Any]]:
    """Return all tool schemas from the default global registry."""
    return _get_default_registry().schemas()


def call_tool(name: str, input_data: dict[str, Any]) -> Any:
    """Call a tool by name using the default global registry."""
    return _get_default_registry().call(name, input_data)
