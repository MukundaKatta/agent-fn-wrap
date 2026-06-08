"""agent-fn-wrap: wrap Python functions with auto-generated Anthropic tool schemas."""

from .core import (
    ToolRegistry,
    ToolWrapError,
    WrappedTool,
    call_tool,
    get_tools,
    reset_default_registry,
    tool,
)

__all__ = [
    "ToolRegistry",
    "ToolWrapError",
    "WrappedTool",
    "call_tool",
    "get_tools",
    "reset_default_registry",
    "tool",
]

__version__ = "0.1.0"
