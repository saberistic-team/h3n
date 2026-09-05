"""Built-in tools and their workspace/permission boundary."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """An error safe to return to the model as a tool observation."""


def _check_argument_type(name: str, value: Any, schema: dict[str, Any]) -> None:
    expected = schema.get("type")
    valid = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
    }
    if expected in valid and not valid[expected](value):
        raise ToolError(f"invalid argument {name}: expected {expected}")
    minimum = schema.get("minimum")
    if minimum is not None and isinstance(value, (int, float)) and value < minimum:
        raise ToolError(f"invalid argument {name}: must be at least {minimum}")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    privileged: bool = False

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }}


class ToolRegistry:
    def __init__(self, workspace: str | Path, *, approve: Callable[[str, dict[str, Any]], bool] | None = None,
                 yes: bool = False, output_limit: int = 20_000, shell_timeout: float = 30.0):
        self.workspace = Path(workspace).resolve()
        self.approve = approve
        self.yes = yes
        self.output_limit = output_limit
        self.shell_timeout = shell_timeout
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ToolError(f"invalid tool arguments JSON: {exc}") from exc
            if not isinstance(arguments, dict):
                raise ToolError("tool arguments must be a JSON object")
            self._validate_arguments(tool, arguments)
            if tool.privileged and not self.yes:
                if self.approve is None or not self.approve(name, arguments):
                    raise ToolError(f"action denied: {name}")
            result = tool.handler(**arguments)
            return {"ok": True, "result": result}
        except (ToolError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"{name} failed: {exc}"}

    def _validate_arguments(self, tool: Tool, arguments: dict[str, Any]) -> None:
        schema = tool.parameters or {}
        properties = schema.get("properties") or {}
        additional_allowed = schema.get("additionalProperties", True)
        for key, value in arguments.items():
            if key not in properties:
                if not additional_allowed:
                    raise ToolError(f"unexpected argument: {key}")
                continue
            _check_argument_type(key, value, properties[key])
        for required in schema.get("required", []):
            if required not in arguments:
                raise ToolError(f"missing required argument: {required}")

    def resolve(self, value: str = ".", *, must_exist: bool = True) -> Path:
        candidate = (self.workspace / value).resolve(strict=False)
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError(f"workspace path violation: {value}") from exc
        if must_exist and not candidate.exists():
            raise ToolError(f"path not found: {value}")
        # resolve() follows existing symlinks, so the relative check also rejects escapes.
        return candidate


def default_registry(workspace: str | Path, **kwargs: Any) -> ToolRegistry:
    registry = ToolRegistry(workspace, **kwargs)

    def read(path: str, start: int = 1, end: int | None = None) -> str:
        target = registry.resolve(path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        lines = target.read_text(encoding="utf-8").splitlines()
        if start < 1 or (end is not None and end < start):
            raise ToolError("invalid line range")
        stop = len(lines) if end is None else min(end, len(lines))
        return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, stop + 1))

    def list_files(path: str = ".", recursive: bool = False) -> list[str]:
        target = registry.resolve(path)
        if target.is_file():
            return [str(target.relative_to(registry.workspace))]
        iterator = target.rglob("*") if recursive else target.iterdir()
        found = []
        for item in iterator:
            # Reject/skip entries resolving outside, including directory symlinks.
            try:
                resolved = item.resolve().relative_to(registry.workspace)
            except (ValueError, OSError):
                continue
            label = str(item.relative_to(registry.workspace)) + ("/" if item.is_dir() else "")
            found.append(label)
        return sorted(found)

    def search(pattern: str, path: str = ".", max_results: int = 200) -> list[str]:
        target = registry.resolve(path)
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        if max_results < 1:
            raise ToolError("max_results must be positive")
        candidates = [target] if target.is_file() else target.rglob("*")
        matches: list[str] = []
        for item in candidates:
            if len(matches) >= max_results:
                break
            try:
                resolved = item.resolve()
                resolved.relative_to(registry.workspace)
                if not resolved.is_file():
                    continue
                for number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
                    if expression.search(line):
                        matches.append(f"{item.relative_to(registry.workspace)}:{number}:{line}")
                        if len(matches) >= max_results:
                            break
            except (UnicodeDecodeError, OSError, ValueError):
                continue
        return matches

    def write(path: str, content: str) -> str:
        target = registry.resolve(path, must_exist=False)
        # Check the nearest existing parent in case a parent component is a symlink.
        parent = target.parent.resolve()
        try:
            parent.relative_to(registry.workspace)
        except ValueError as exc:
            raise ToolError(f"workspace path violation: {path}") from exc
        parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {path}"

    def edit(path: str, old: str, new: str) -> str:
        target = registry.resolve(path)
        content = target.read_text(encoding="utf-8")
        count = content.count(old)
        if count != 1:
            raise ToolError(f"expected exactly one match, found {count}")
        target.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"edited {path}"

    def shell(command: str, timeout: float | None = None) -> dict[str, Any]:
        duration = registry.shell_timeout if timeout is None else min(max(timeout, 0.1), registry.shell_timeout)
        try:
            completed = subprocess.run(command, cwd=registry.workspace, shell=True, text=True,
                                       capture_output=True, timeout=duration, executable=os.environ.get("SHELL"))
        except subprocess.TimeoutExpired as exc:
            output = ((exc.stdout or "") + (exc.stderr or ""))
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            raise ToolError(f"command timed out after {duration:g}s; output: {truncate(output, registry.output_limit)}") from exc
        return {"exit_status": completed.returncode,
                "stdout": truncate(completed.stdout, registry.output_limit),
                "stderr": truncate(completed.stderr, registry.output_limit)}

    obj = {"type": "object", "additionalProperties": False}
    registry.register(Tool("read", "Read a UTF-8 file with line numbers.", obj | {"properties": {
        "path": {"type": "string"}, "start": {"type": "integer", "minimum": 1},
        "end": {"type": "integer", "minimum": 1}}, "required": ["path"]}, read))
    registry.register(Tool("list", "List workspace files and directories.", obj | {"properties": {
        "path": {"type": "string", "default": "."}, "recursive": {"type": "boolean", "default": False}}}, list_files))
    registry.register(Tool("search", "Search UTF-8 workspace files with a regular expression.", obj | {"properties": {
        "pattern": {"type": "string"}, "path": {"type": "string", "default": "."},
        "max_results": {"type": "integer", "default": 200}}, "required": ["pattern"]}, search))
    registry.register(Tool("write", "Create or replace a UTF-8 file.", obj | {"properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, write, True))
    registry.register(Tool("edit", "Replace exactly one occurrence of text in a UTF-8 file.", obj | {"properties": {
        "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
        "required": ["path", "old", "new"]}, edit, True))
    registry.register(Tool("shell", "Run a shell command from the workspace.", obj | {"properties": {
        "command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"]}, shell, True))
    return registry


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + f"\n...[truncated {omitted} characters]"
