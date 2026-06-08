# agent-fn-wrap

Wrap Python functions as Anthropic tool schemas — auto-generates `input_schema` from type hints.

Zero dependencies. Python 3.10+. MIT.

## Install

```bash
pip install agent-fn-wrap
```

## Usage

```python
from agent_fn_wrap import ToolRegistry

registry = ToolRegistry()

@registry.tool(description="Search the web for current information.")
def web_search(q: str, max_results: int = 5) -> str:
    return search_api.query(q, n=max_results)

@registry.tool(description="Read a local file.")
def read_file(path: str) -> str:
    return open(path).read()

# Get Anthropic-style tool schemas
schemas = registry.schemas()
# [{"name": "web_search", "description": "...", "input_schema": {...}}, ...]

# Pass to the API
response = client.messages.create(
    model="claude-sonnet-4-5",
    tools=schemas,
    messages=messages,
)

# Call a tool by name from the LLM's tool_use block
result = registry.call("web_search", {"q": "Paris", "max_results": 3})
```

## Auto-generated schema

Type hints map to JSON Schema types:

| Python annotation        | JSON Schema type      |
| ------------------------ | --------------------- |
| `str`                    | `"string"`            |
| `int`                    | `"integer"`           |
| `float`                  | `"number"`            |
| `bool`                   | `"boolean"`           |
| `list` / `list[T]`       | `"array"`             |
| `dict` / `dict[K, V]`    | `"object"`            |
| `Optional[T]` / `T \| None` | type of `T`        |
| _no annotation / unknown_ | `"string"` (fallback) |

- Parameters **without** defaults are marked `required`.
- Parameters **with** defaults get the `default` value in the schema and are not required.
- A typed list such as `list[str]` also gets an `"items"` entry describing the element type.
- `Optional[T]` / `T | None` unwraps to the type of `T`; for a multi-type union the first non-`None` member wins.
- `*args` and `**kwargs` cannot be represented as named JSON Schema properties and are skipped.

```python
@registry.tool(description="Filter records.")
def filter_records(tags: list[str], limit: int = 10) -> list[dict]:
    ...

# input_schema:
# {
#   "type": "object",
#   "properties": {
#     "tags": {"type": "array", "items": {"type": "string"}},
#     "limit": {"type": "integer", "default": 10},
#   },
#   "required": ["tags"],
# }
```

## Description from docstring

```python
@registry.tool
def web_search(q: str) -> str:
    """Search the web for current information."""
    return search(q)
# description = "Search the web for current information."
```

## Schema override

```python
@registry.tool(description="Search.", schema_override={
    "type": "object",
    "properties": {"q": {"type": "string", "description": "Search query"}},
    "required": ["q"],
})
def search(q): ...
```

## Module-level convenience API

For small scripts you can skip creating a registry and use a process-wide
default one:

```python
from agent_fn_wrap import tool, get_tools, call_tool, reset_default_registry

@tool(description="Echo input.")
def echo(text: str) -> str:
    return text

schemas = get_tools()                  # all default-registry schemas
result = call_tool("echo", {"text": "hi"})
reset_default_registry()               # clear global state (handy in tests)
```

Prefer a dedicated `ToolRegistry` instance when you need isolation, such as in
tests or when serving multiple agents from the same process.

## API

| Symbol                            | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| `ToolRegistry()`                  | Container for wrapped tools.                                             |
| `ToolRegistry.tool(...)`          | Decorator that registers a function (usable with or without arguments). |
| `ToolRegistry.register(fn, ...)`  | Register a function without decorator syntax; returns `self`.           |
| `ToolRegistry.schemas()`          | List of Anthropic-style tool dicts.                                     |
| `ToolRegistry.call(name, input)`  | Invoke a registered tool by name. Raises `ToolWrapError` if unknown.    |
| `ToolRegistry.names()` / `has()`  | List tool names / check membership.                                     |
| `WrappedTool`                     | A single function wrapped as a tool (`.schema()`, `.call(input)`).      |
| `tool` / `get_tools` / `call_tool`| Module-level helpers backed by the default registry.                    |
| `reset_default_registry()`        | Clear the default registry.                                             |
| `ToolWrapError`                   | Raised on invalid configuration or unknown tool names.                  |

## Development

The package has no runtime dependencies, and the test suite uses only the
standard library (`unittest`). From the repository root:

```bash
python -m unittest discover -s tests
```

## License

MIT
