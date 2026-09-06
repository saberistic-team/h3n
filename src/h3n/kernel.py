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
Reason only until the next concrete action is clear, then call the tool immediately.
Do not analyze the entire task or narrate implementation details before taking action.
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
            try:
                detail = exc.read().decode(errors="replace")
            finally:
                exc.close()
            if exc.code == 404 and "model" in detail.lower():
                raise OllamaError(f"model not found: {detail}") from exc
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}; is it running? ({exc.reason})") from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama request timed out after {self.timeout:g}s at {self.host}; "
                "retry with --timeout SECONDS or pre-load the model with ollama run"
            ) from exc

    def chat(self, *, model: str, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None, stream: bool = True,
             on_chunk: Callable[[dict[str, Any]], None] | None = None,
             options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if tools is not None:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        received = False
        tool_calls: list[dict[str, Any]] = []
        thinking: list[str] = []
        done_reason: str | None = None
        for chunk in self._request(payload):
            part = chunk.get("message")
            if not isinstance(part, dict):
                continue
            received = True
            if on_chunk is not None:
                on_chunk(part)
            message["content"] += part.get("content", "")
            if part.get("thinking"):
                thinking.append(part["thinking"])
            if part.get("tool_calls"):
                tool_calls.extend(part["tool_calls"])
            if part.get("role"):
                message["role"] = part["role"]
            if chunk.get("done_reason"):
                done_reason = chunk["done_reason"]
        if not received:
            raise OllamaError("Ollama returned no chat message")
        if thinking:
            message["thinking"] = "".join(thinking)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if done_reason:
            # Internal transport metadata; AgentKernel removes it before history.
            message["_done_reason"] = done_reason
        return message

    def stream_chat(self, *, model: str, messages: list[dict[str, Any]], stream: bool = True,
                    on_thinking: Callable[[str], None] | None = None) -> Iterable[str]:
        if not stream:
            message = self.chat(model=model, messages=messages, stream=False)
            if message.get("thinking") and on_thinking is not None:
                on_thinking(message["thinking"])
            if message.get("content"):
                yield message["content"]
            return
        received = False
        for chunk in self._request({"model": model, "messages": messages, "stream": True}):
            message = chunk.get("message", {})
            if message:
                received = True
            if message.get("thinking") and on_thinking is not None:
                on_thinking(message["thinking"])
            if message.get("content"):
                yield message["content"]
        if not received:
            raise OllamaError("Ollama returned no chat message")


# Tools that modify the workspace and therefore leave it unverified until a
# successful verification command runs afterwards.
CHANGE_TOOLS = frozenset({"write", "edit"})


class CompletionController:
    """Verification-aware completion and no-progress detection.

    The controller only produces observations for the model. It never
    terminates a run and never imposes a step limit. It tracks whether
    workspace-changing tools have run since the most recent successful
    verification, recognizing a successful verification by a shell command's
    exit status rather than by anything the model claims.
    """

    def __init__(self, repeat_limit: int = 3) -> None:
        self.repeat_limit = repeat_limit
        self.changes_unverified = False
        self.verified = False
        self._signature: str | None = None
        self._repetition = 0
        self._verified_this_batch = False

    def begin_batch(self) -> None:
        """Reset per-batch bookkeeping before executing a batch of calls."""
        self._verified_this_batch = False

    def record(self, name: str, arguments: Any, result: dict) -> str | None:
        """Update state for one tool result; return a repetition observation or None."""
        if result.get("ok") and name in CHANGE_TOOLS:
            self.changes_unverified = True
        if result.get("ok") and name == "shell":
            payload = result.get("result")
            if isinstance(payload, dict) and payload.get("exit_status") == 0:
                self.verified = True
                self.changes_unverified = False
                self._verified_this_batch = True
        return self._record_repetition(name, arguments, result)

    def verification_observation(self) -> str | None:
        """Return a concise completion note after a successful verification."""
        if not self._verified_this_batch:
            return None
        state = ("no unverified changes remain"
                 if not self.changes_unverified else "changes remain unverified")
        return ("[kernel] Verification succeeded; " + state +
                ". If every requirement is complete, return your final answer now; "
                "otherwise make the remaining changes and verify them again.")

    def _record_repetition(self, name: str, arguments: Any, result: dict) -> str | None:
        signature = self._signature_for(name, arguments, result)
        if signature == self._signature:
            self._repetition += 1
        else:
            self._signature = signature
            self._repetition = 1
        if self._repetition >= self.repeat_limit and self._repetition % self.repeat_limit == 0:
            return ("[kernel] The same tool call returned the same result "
                    f"{self.repeat_limit} times in a row with no progress. "
                    "Choose a different action, change the input, or return your final answer.")
        return None

    def _signature_for(self, name: str, arguments: Any, result: dict) -> str:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            args_key = json.dumps(parsed, sort_keys=True, default=str)
        except (ValueError, TypeError):
            args_key = str(arguments)
        try:
            result_key = json.dumps(result, sort_keys=True, default=str)
        except (ValueError, TypeError):
            result_key = str(result)
        return f"{name}|{args_key}|{result_key}"


class AgentKernel:
    def __init__(self, client: OllamaClient, registry: ToolRegistry, *, model: str,
                 system: str = DEFAULT_SYSTEM, max_steps: int = 0,
                 stream: bool = True, show_reasoning: bool = False,
                 max_tools_per_step: int = 3, action_tokens: int = 0,
                 observation_limit: int = 8_000, context_limit: int = 50_000,
                 on_event: Callable[[str], None] | None = None,
                 on_reasoning: Callable[[str], None] | None = None):
        if max_steps < 0:
            raise ValueError("max_steps must be zero (unlimited) or positive")
        self.client, self.registry, self.model = client, registry, model
        self.max_steps = max_steps
        self.completion = CompletionController()
        self.stream = stream
        self.show_reasoning = show_reasoning
        self.max_tools_per_step = max_tools_per_step
        self.action_tokens = action_tokens
        self.observation_limit = observation_limit
        self.context_limit = context_limit
        self.on_event = on_event
        self.on_reasoning = on_reasoning
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def compact_context(self) -> None:
        """Mechanically compact old tool output while preserving recent evidence."""
        size = sum(len(json.dumps(item, ensure_ascii=False)) for item in self.messages)
        if size <= self.context_limit:
            return
        tool_indexes = [i for i, item in enumerate(self.messages) if item.get("role") == "tool"]
        for index in tool_indexes[:-4]:
            content = self.messages[index].get("content", "")
            if content.startswith("[compacted "):
                continue
            self.messages[index]["content"] = f"[compacted tool observation: {len(content):,} chars]"
            size = sum(len(json.dumps(item, ensure_ascii=False)) for item in self.messages)
            if size <= self.context_limit:
                break

    def observation(self, result: dict[str, Any]) -> str:
        serialized = json.dumps(result, ensure_ascii=False)
        if len(serialized) <= self.observation_limit:
            return serialized
        preview_limit = max(0, self.observation_limit - 160)
        return json.dumps({
            "ok": result.get("ok", False),
            "truncated": True,
            "original_chars": len(serialized),
            "preview": serialized[:preview_limit],
        }, ensure_ascii=False)

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
        self.emit(f"Objective: {task}")
        self.completion = CompletionController()
        pending_calls: list[dict[str, Any]] = []
        token_budget = self.action_tokens or None
        step = 1

        def execute_calls(calls: list[dict[str, Any]]) -> None:
            self.completion.begin_batch()
            notes: list[str] = []
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                arguments = function.get("arguments", {})
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
                else:
                    self.emit(f"Completed: {name}")
                observation = {
                     "role": "tool",
                     "content": self.observation(result),
                     "tool_name": name,
                 }
                if call.get("id"):
                    observation["tool_call_id"] = call["id"]
                self.messages.append(observation)
                note = self.completion.record(name, arguments, result)
                if note:
                    notes.append(note)
            note = self.completion.verification_observation()
            if note:
                notes.append(note)
            if notes:
                  # A kernel observation is appended as a user message so that
                  # tool observations keep a valid call/response protocol.
                self.messages.append({"role": "user", "content": "\n".join(notes)})

        while True:
            if self.max_steps and step > self.max_steps:
                raise StepLimitError(
                    f"Reached the {self.max_steps}-step limit; request a higher limit "
                     "or continue the conversation.")
            step_label = f"step {step}/{self.max_steps or '\u221e'}"
            if pending_calls:
                calls = pending_calls[:self.max_tools_per_step]
                pending_calls = pending_calls[self.max_tools_per_step:]
                self.emit(
                    f"Running {len(calls)} deferred tool call{'s' if len(calls) != 1 else ''} "
                     f"({step_label})")
                  # Record a matching assistant call message so native Ollama tool
                  # observations remain protocol-valid without another model request.
                self.messages.append({"role": "assistant", "content": "", "tool_calls": calls})
                try:
                    execute_calls(calls)
                except KeyboardInterrupt:
                    self.emit("Interrupted by user")
                    raise
                if pending_calls:
                    self.emit(
                        f"Deferred {len(pending_calls)} tool call"
                         f"{'s' if len(pending_calls) != 1 else ''} to keep this step focused")
                step += 1
                continue
            self.compact_context()
            context_chars = sum(len(json.dumps(item, ensure_ascii=False)) for item in self.messages)
            self.emit(
                f"Waiting for {self.model} ({step_label}; "
                 f"{len(self.messages)} messages, {context_chars:,} context chars)...")
            saw_chunk = False

            def receive(part: dict[str, Any]) -> None:
                nonlocal saw_chunk
                if not saw_chunk:
                    saw_chunk = True
                    self.emit("Streaming response from model...")
                thinking = part.get("thinking", "")
                if thinking and self.show_reasoning and self.on_reasoning is not None:
                    self.on_reasoning(thinking)

            try:
                message = self.client.chat(model=self.model, messages=self.messages,
                                          tools=self.registry.schemas, stream=self.stream,
                                          on_chunk=receive,
                                          options=({"num_predict": token_budget}
                                                   if token_budget is not None else None))
            except KeyboardInterrupt:
                self.emit("Interrupted by user")
                raise
            done_reason = message.pop("_done_reason", None)
            all_calls = message.get("tool_calls") or []
            calls = all_calls[:self.max_tools_per_step]
            pending_calls = all_calls[self.max_tools_per_step:]
            deferred = len(pending_calls)
            if not calls and done_reason in {"length", "max_tokens"}:
                if token_budget is None:
                    raise StepLimitError(
                          "model stopped because of its generation limit without producing "
                           "an action or final response")
                self.emit(
                    f"Generation budget reached ({token_budget} tokens); "
                      "retrying once without an action-token cap")
                token_budget = None
                  # Do not put a partial reasoning turn into history. Retrying the
                  # same prompt with a larger budget avoids repetitive continuations.
                continue
            if calls:
                message = dict(message)
                message["tool_calls"] = calls
            self.messages.append(message)
            if not calls:
                return message.get("content", "")
            token_budget = self.action_tokens or None
            decision = message.get("content", "").strip()
            if decision:
                self.emit(decision)
            if deferred:
                self.emit(f"Deferred {deferred} tool call{'s' if deferred != 1 else ''} to keep this step focused")
            try:
                execute_calls(calls)
            except KeyboardInterrupt:
                self.emit("Interrupted by user")
                raise
            step += 1
