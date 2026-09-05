"""Command-line interface for h3n."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TextIO

from . import __version__
from .kernel import DEFAULT_SYSTEM, AgentKernel, OllamaClient, OllamaError, StepLimitError
from .tools import default_registry


def _quote_path(value: str) -> str:
    """Return a path rendered unambiguously, even when it contains spaces."""
    return '"' + value + '"'


def environment_context(workspace, python_executable=None) -> str:
    """Build the trusted runtime-environment block appended to the system prompt.

    It exposes only the resolved absolute workspace path and the active Python
    executable, never unrelated environment variables, credentials, tokens, or
    secrets.
    """
    resolved = Path(workspace).resolve()
    executable = python_executable if python_executable is not None else sys.executable
    return "\n".join([
        "## Runtime environment",
        "These values are resolved by h3n and trusted:",
        f"Workspace path (absolute): {_quote_path(str(resolved))}",
        f"Python executable (absolute): {_quote_path(str(executable))}",
        "",
        "All file tools and shell commands already run from the workspace, so pass "
        "workspace-relative paths to file tools. Do not guess absolute paths such as "
        "/workspace, and do not prefix shell commands with `cd` into the workspace.",
        "Use the Python executable above when running tests or Python commands.",
    ])


def compose_system_prompt(base_system: str, workspace, python_executable=None) -> str:
    """Append the trusted runtime-environment context to a base system prompt.

    A custom prompt supplied through --system is preserved verbatim; the
    environment context is only appended, never replacing it.
    """
    return f"{base_system}\n\n{environment_context(workspace, python_executable)}"

DEFAULT_MODEL = "qwen3.8:27b-mlx"
DEFAULT_KERNEL = "h3n"
DEFAULT_TIMEOUT = 300.0
_MARKDOWN_ESCAPE = re.compile(r"\\([`*_{}\[\]()#+.!|>~-])")


class ProgressReporter:
    """TTY-aware, dependency-free progress display for agent activity."""

    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    def __init__(self, stream: TextIO = sys.stderr, environ: dict[str, str] | None = None):
        env = os.environ if environ is None else environ
        self.stream = stream
        self.tty = bool(getattr(stream, "isatty", lambda: False)())
        self.color = self.tty and "NO_COLOR" not in env and env.get("TERM") != "dumb"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._waiting = ""
        self._phase = ""
        self._reasoning_open = False
        self._reasoning_ended_line = True

    def _paint(self, text: str, color: str) -> str:
        return f"{color}{text}{self.RESET}" if self.color else text

    def __call__(self, event: str) -> None:
        event = terminal_text(event)
        if event.startswith("Waiting for "):
            self.start_waiting(event)
            return
        if event.startswith("Streaming response"):
            self._phase = "receiving the model's response"
            if not self.tty:
                print("h3n 📡 Receiving streamed response", file=self.stream, flush=True)
            return
        self.stop_waiting()
        self.stop_reasoning()
        icon, color = self._style(event)
        print(f"{self._paint('h3n', self.CYAN)} {icon} {self._paint(event, color)}",
              file=self.stream, flush=True)

    def _style(self, event: str) -> tuple[str, str]:
        if event.startswith("Objective:"):
            return "🎯", self.MAGENTA
        if event.startswith("Completed:"):
            return "✅", self.GREEN
        if event.startswith("Deferred "):
            return "⏭️ ", self.YELLOW
        if event.startswith("Generation budget reached"):
            return "🔄", self.YELLOW
        if event.startswith("Reading "):
            return "📖", self.BLUE
        if event.startswith("Inspecting "):
            return "🗂️ ", self.BLUE
        if event.startswith("Searching "):
            return "🔎", self.BLUE
        if event.startswith("Preparing to run"):
            return "⚙️ ", self.YELLOW
        if event.startswith(("Preparing to write", "Preparing to edit")):
            return "✏️ ", self.YELLOW
        if event.startswith("Tool failed"):
            return "⚠️ ", self.RED
        return "💭", self.GREEN

    def start_waiting(self, event: str) -> None:
        self.stop_waiting()
        # Turn "Waiting for model (step N/M)..." into useful context.
        detail = event.removeprefix("Waiting for ").removesuffix("...")
        self._waiting = detail
        self._phase = ("loading model and awaiting its first response"
                       if "(step 1/" in detail else
                       "evaluating tool results and planning the next step")
        self._stop.clear()
        if not self.tty:
            print(f"h3n 🧠 Model is reviewing observations · {detail}", file=self.stream, flush=True)
            return
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        started = time.monotonic()
        tick = 0
        last_milestone = -1
        milestones = (
            (0, "request submitted to Ollama"),
            (5, "model is evaluating the conversation"),
            (15, "still evaluating context; no streamed chunk yet"),
            (30, "Ollama is busy; waiting for the first streamed chunk"),
            (60, "long prompt evaluation; model has not produced a chunk"),
            (120, "extended evaluation; check ollama ps if resources look constrained"),
        )
        while not self._stop.wait(0.12):
            elapsed = int(time.monotonic() - started)
            thought = self._phase
            if self._phase != "receiving the model's response":
                milestone = max(i for i, (seconds, _) in enumerate(milestones) if elapsed >= seconds)
                thought = milestones[milestone][1]
                if milestone != last_milestone and last_milestone >= 0:
                    update = (f"{self._paint('h3n', self.CYAN)} ⏱️  "
                              f"{self._paint(thought, self.YELLOW)} "
                              f"{self._paint(f'· {elapsed}s elapsed', self.DIM)}")
                    print("\r\033[2K" + update, file=self.stream, flush=True)
                last_milestone = milestone
            line = (f"{self._paint('h3n', self.CYAN)} "
                    f"{self._paint(self.FRAMES[tick % len(self.FRAMES)], self.GREEN)} 🧠 "
                    f"{thought} {self._paint(f'· {self._waiting} · {elapsed}s', self.DIM)}")
            print("\r\033[2K" + line, end="", file=self.stream, flush=True)
            tick += 1

    def stop_waiting(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=0.3)
            self._thread = None
            print("\r\033[2K", end="", file=self.stream, flush=True)

    def reasoning(self, text: str) -> None:
        self.stop_waiting()
        if not self._reasoning_open:
            print(f"{self._paint('h3n', self.CYAN)} 🧠 {self._paint('Reasoning', self.MAGENTA)}",
                  file=self.stream, flush=True)
            self._reasoning_open = True
        print(self._paint(text, self.DIM), end="", file=self.stream, flush=True)
        self._reasoning_ended_line = text.endswith("\n")

    def stop_reasoning(self) -> None:
        if self._reasoning_open:
            if not self._reasoning_ended_line:
                print(file=self.stream, flush=True)
            self._reasoning_open = False
            self._reasoning_ended_line = True

    def close(self) -> None:
        self.stop_waiting()
        self.stop_reasoning()


def parser(environ: dict[str, str] | None = None) -> argparse.ArgumentParser:
    env = os.environ if environ is None else environ
    result = argparse.ArgumentParser(prog="h3n", description="Standalone coding agent for Ollama")
    result.add_argument("task", nargs="?", help="task or chat prompt; omit for interactive mode")
    result.add_argument("--version", action="version", version=f"h3n {__version__}",
                        help="show program version and exit")
    result.add_argument("-m", "--model", default=env.get("H3N_MODEL", DEFAULT_MODEL))
    result.add_argument("--host", default=env.get("OLLAMA_HOST", "http://localhost:11434"))
    result.add_argument("--timeout", type=positive_float,
                        default=env.get("H3N_TIMEOUT", DEFAULT_TIMEOUT),
                        help="Ollama request timeout in seconds (default: 300)")
    result.add_argument("-s", "--system", default=DEFAULT_SYSTEM)
    result.add_argument("--kernel", choices=("h3n", "direct"), default=env.get("H3N_KERNEL", DEFAULT_KERNEL))
    result.add_argument("--no-stream", dest="stream", action="store_false",
                        help="disable response streaming")
    reasoning = result.add_mutually_exclusive_group()
    reasoning.add_argument("--show-reasoning", dest="show_reasoning", action="store_true",
                           help="show model thinking on stderr (default)")
    reasoning.add_argument("--hide-reasoning", dest="show_reasoning", action="store_false",
                           help="hide model thinking")
    result.set_defaults(show_reasoning=True)
    result.add_argument("-y", "--yes", action="store_true", help="approve privileged tools without prompting")
    result.add_argument("--max-steps", type=positive_int, default=20)
    result.add_argument("--max-tools-per-step", type=positive_int, default=3,
                        help="maximum tool calls executed per model round (default: 3)")
    result.add_argument("--action-tokens", type=nonnegative_int, default=0,
                        help="optional generation cap per round; 0 is unlimited (default)")
    result.add_argument("--observation-limit", type=positive_int, default=8000,
                        help="maximum characters retained per tool result (default: 8000)")
    result.add_argument("--context-limit", type=positive_int, default=50000,
                        help="compact old tool results above this size (default: 50000)")
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def confirm(name: str, arguments: dict, *, input_stream: TextIO = sys.stdin, error_stream: TextIO = sys.stderr) -> bool:
    summary = arguments.get("path") or arguments.get("command") or ""
    print(f"Approve {name}: {summary!s}? [y/N] ", end="", file=error_stream, flush=True)
    answer = input_stream.readline()
    return answer.strip().lower() in {"y", "yes"}


def terminal_text(value: str) -> str:
    r"""Undo Markdown presentation escapes in output intended for a terminal.

    Only backslashes before Markdown punctuation are removed. Escapes used by
    regular expressions (such as ``\d``) and path separators remain untouched.
    """
    return _MARKDOWN_ESCAPE.sub(r"\1", value)


def run_direct(client: OllamaClient, model: str, system: str, task: str, history: list[dict] | None = None,
               output: TextIO = sys.stdout, *, stream: bool = True,
               on_thinking: Callable[[str], None] | None = None,
               on_content_start: Callable[[], None] | None = None) -> str:
    messages = history if history is not None else [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": task})
    parts = []
    thinking_parts = []
    content_started = False

    def receive_thinking(text: str) -> None:
        thinking_parts.append(text)
        if on_thinking is not None:
            on_thinking(text)

    for part in client.stream_chat(model=model, messages=messages, stream=stream,
                                   on_thinking=receive_thinking):
        if not content_started:
            content_started = True
            if on_content_start is not None:
                on_content_start()
        print(part, end="", file=output, flush=True)
        parts.append(part)
    print(file=output)
    content = "".join(parts)
    assistant = {"role": "assistant", "content": content}
    if thinking_parts:
        assistant["thinking"] = "".join(thinking_parts)
    messages.append(assistant)
    return content


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    client = OllamaClient(args.host, timeout=args.timeout)
    workspace = Path.cwd().resolve()
    system_prompt = compose_system_prompt(args.system, workspace, sys.executable)
    try:
        if args.kernel == "direct":
            reporter = ProgressReporter()
            history = [{"role": "system", "content": system_prompt}]
            reasoning = reporter.reasoning if args.show_reasoning else None
            try:
                if args.task is not None:
                    run_direct(client, args.model, system_prompt, args.task, history,
                               stream=args.stream, on_thinking=reasoning,
                               on_content_start=reporter.stop_reasoning)
                else:
                    interactive(lambda text: run_direct(
                        client, args.model, system_prompt, text, history,
                        stream=args.stream, on_thinking=reasoning,
                        on_content_start=reporter.stop_reasoning))
            finally:
                reporter.close()
        else:
            reporter = ProgressReporter()
            registry = default_registry(workspace, yes=args.yes,
                approve=lambda name, values: confirm(name, values))
            kernel = AgentKernel(client, registry, model=args.model, system=system_prompt,
                                 max_steps=args.max_steps,
                                 stream=args.stream, show_reasoning=args.show_reasoning,
                                 max_tools_per_step=args.max_tools_per_step,
                                 action_tokens=args.action_tokens,
                                 observation_limit=args.observation_limit,
                                 context_limit=args.context_limit,
                                 on_event=reporter, on_reasoning=reporter.reasoning)
            def run_agent(text: str) -> None:
                response = kernel.run(text)
                reporter.close()
                print(terminal_text(response))
            try:
                if args.task is not None:
                    run_agent(args.task)
                else:
                    interactive(run_agent)
            finally:
                reporter.close()
        return 0
    except OllamaError as exc:
        print(f"h3n: {exc}", file=sys.stderr)
        return 2
    except StepLimitError as exc:
        print(f"h3n: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def interactive(handler) -> None:
    while True:
        try:
            text = input("h3n> ")
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return
        if text.strip() in {"/exit", "/quit"}:
            return
        if text.strip():
            handler(text)
