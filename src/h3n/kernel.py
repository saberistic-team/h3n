"""Ollama transport and h3n's action/observation loop."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .tools import ToolRegistry


DEFAULT_SYSTEM = """You are h3n, a coding agent operating in a local workspace.
Inspect before editing. Make focused changes. Verify completed work. Use tools whenever
claims depend on workspace state. Never claim success without supporting tool output.
When calling tools, put one short sentence in content explaining your next decision.
Finish with a concise summary and verification result. Your final response is printed
directly in a terminal: use clean plain text, not Markdown, and never backslash-escape
formatting characters."""


class OllamaError(Exception):
    pass


class StepLimitError(Exception):
    pass


@dataclass
class OllamaClient:
    host: str = "http://localhost:11434"
    timeout: float = 120.0

    def _request(self, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        url = self.host.rstrip("/") + "/api/chat"
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
            with response:
                for raw in response:
                    if not raw.strip():
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise OllamaError("Ollama returned malformed JSON") from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 404 and "model" in detail.lower():
                raise OllamaError(f"model not found: {detail}") from exc
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}; is it running? ({exc.reason})") from exc
        except TimeoutError as exc:
            raise OllamaError(f"Ollama request timed out at {self.host}") from exc

    def chat(self, *, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools is not None:
            payload["tools"] = tools
        chunks = list(self._request(payload))
        if not chunks or "message" not in chunks[-1]:
            raise OllamaError("Ollama returned no chat message")
        return chunks[-1]["message"]

    def stream_chat(self, *, model: str, messages: list[dict[str, Any]]) -> Iterable[str]:
        for chunk in self._request({"model": model, "messages": messages, "stream": True}):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content


class AgentKernel:
    def __init__(self, client: OllamaClient, registry: ToolRegistry, *, model: str,
                 system: str = DEFAULT_SYSTEM, max_steps: int = 20,
                 on_event: Callable[[str], None] | None = None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.client, self.registry, self.model = client, registry, model
        self.max_steps = max_steps
        self.on_event = on_event
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)

    @staticmethod
    def describe_tool(name: str, arguments: Any) -> str:
        if not isinstance(arguments, dict):
            return f"Calling {name or 'an unnamed tool'}"
        path = arguments.get("path", ".")
        if name == "read":
            return f"Reading {path}"
        if name == "list":
            return f"Inspecting files in {path}"
        if name == "search":
            return f"Searching {path} for {arguments.get('pattern', '')!r}"
        if name == "write":
            return f"Preparing to write {path}"
        if name == "edit":
            return f"Preparing to edit {path}"
        if name == "shell":
            command = str(arguments.get("command", ""))
            if len(command) > 120:
                command = command[:117] + "..."
            return f"Preparing to run: {command}"
        return f"Calling tool: {name or '<missing name>'}"

    def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        for step in range(1, self.max_steps + 1):
            self.emit(f"Waiting for {self.model} (step {step}/{self.max_steps})...")
            message = self.client.chat(model=self.model, messages=self.messages, tools=self.registry.schemas)
            self.messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                return message.get("content", "")
            decision = message.get("content", "").strip()
            if decision:
                self.emit(decision)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                arguments = function.get("arguments", {})
                # Ollama may return tool arguments as either an object or a JSON string.
                display_arguments = arguments
                if isinstance(arguments, str):
                    try:
                        display_arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                self.emit(self.describe_tool(name, display_arguments))
                result = self.registry.execute(name, arguments)
                if not result.get("ok"):
                    self.emit(f"Tool failed: {result.get('error', 'unknown error')}")
                observation = {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}
                if call.get("id"):
                    observation["tool_call_id"] = call["id"]
                self.messages.append(observation)
        raise StepLimitError(f"maximum step count reached ({self.max_steps})")
