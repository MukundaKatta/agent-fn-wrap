"""Wrap Python functions as Anthropic tool schemas.

Inspects function signatures to generate tool input schemas automatically.
Zero dependencies — uses inspect and typing from the standard library.
"""

from __future__ import annotations

import inspect
import types
import typing
from typing import Any, Callable, get_args, get_origin, get_type_hints


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


def _is_union(origin: Any) -> bool:
    """Return True if ``origin`` is a union origin (typing.Union or X | Y).

    Works across Python versions where ``X | Y`` may report its origin as
    ``types.UnionType`` (3.10) or ``typing.Union`` (3.14+).
    """
    if origin is typing.Union:
        return True
    union_type = getattr(types, "UnionType", None)
    return union_type is not None and origin is union_type


def _json_type(annotation: Any) -> str:
    """Convert a Python type annotation to a JSON Schema type string.

    Handles bare types (``str`` -> ``"string"``), parameterized generics
    (``list[int]`` -> ``"array"``), and optionals/unions
    (``int | None`` / ``Optional[int]`` -> ``"integer"``), picking the first
    non-``None`` member of a union. Unknown annotations fall back to
    ``"string"``.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return "string"

    origin = get_origin(annotation)

    # Union / Optional — pick the first non-None member.
    if _is_union(origin):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if args else "string"

    # Parameterized generics such as list[str] or dict[str, int]: map the
    # container origin (list -> "array", dict -> "object").
    if origin is not None:
        return _TYPE_MAP.get(origin, "string")

    return _TYPE_MAP.get(annotation, "string")


def _array_item_type(annotation: Any) -> str | None:
    """Return the JSON Schema type of a parameterized list's element type.

    For ``list[int]`` this returns ``"integer"``; for a bare ``list`` (no
    element type) it returns ``None`` so no ``items`` key is emitted.
    """
    if get_origin(annotation) is list:
        args = get_args(annotation)
        if args:
            return _json_type(args[0])
    return None


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
        # *args / **kwargs cannot be expressed as named JSON Schema
        # properties, so they are skipped rather than emitted as scalars.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, inspect.Parameter.empty)
        json_type = _json_type(annotation)
        prop: dict[str, Any] = {"type": json_type}

        # For typed arrays (e.g. list[str]), include the element type so the
        # model knows what the items should look like.
        if json_type == "array":
            item_type = _array_item_type(annotation)
            if item_type is not None:
                prop["items"] = {"type": item_type}

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

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
    schema_override: dict[str, Any] | None = None,
) -> Any:
    """Register a function as a tool in the default global registry.

    Mirrors :meth:`ToolRegistry.tool` but operates on a process-wide default
    registry, which is convenient for small scripts. Use a dedicated
    :class:`ToolRegistry` instance when you need isolation (e.g. in tests or
    when serving multiple agents).
    """
    return _get_default_registry().tool(
        fn,
        name=name,
        description=description,
        schema_override=schema_override,
    )


def get_tools() -> list[dict[str, Any]]:
    """Return all tool schemas from the default global registry."""
    return _get_default_registry().schemas()


def call_tool(name: str, input_data: dict[str, Any]) -> Any:
    """Call a tool by name using the default global registry."""
    return _get_default_registry().call(name, input_data)


def reset_default_registry() -> None:
    """Clear the process-wide default registry.

    Primarily useful in tests to isolate state between cases that exercise the
    module-level :func:`tool`, :func:`get_tools`, and :func:`call_tool` API.
    """
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None
