import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

from h3n.cli import (DEFAULT_KERNEL, DEFAULT_MODEL, DEFAULT_TIMEOUT, ProgressReporter,
                    compose_system_prompt, environment_context, parser, run_direct, terminal_text)
from h3n.kernel import AgentKernel, OllamaClient, OllamaError, StepLimitError
from h3n.tools import Tool, ToolRegistry, default_registry


class FakeClient:
    def __init__(self, responses): self.responses = iter(responses); self.seen = []
    def chat(self, **kwargs): self.seen.append(kwargs); return next(self.responses)


class KernelTests(unittest.TestCase):
    def test_multistep_object_and_string_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.txt").write_text("hello")
            client = FakeClient([
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "read", "arguments": {"path": "a.txt"}}}]},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "list", "arguments": '{"path":"."}'}}]},
                {"role": "assistant", "content": "done"},
            ])
            kernel = AgentKernel(client, default_registry(tmp), model="m")
            self.assertEqual(kernel.run("inspect"), "done")
            observations = [m for m in kernel.messages if m["role"] == "tool"]
            self.assertEqual(len(observations), 2)
            self.assertIn("hello", observations[0]["content"])
            self.assertEqual(observations[0]["tool_name"], "read")
            self.assertEqual(observations[1]["tool_name"], "list")

    def test_max_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = {"role": "assistant", "tool_calls": [{"function": {"name": "list", "arguments": {}}}]}
            with self.assertRaises(StepLimitError):
                AgentKernel(FakeClient([response]), default_registry(tmp), model="m", max_steps=1).run("x")

    def test_progress_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            client = FakeClient([
                {"role": "assistant", "tool_calls": [{"function": {"name": "list", "arguments": {}}}]},
                {"role": "assistant", "content": "done"},
            ])
            kernel = AgentKernel(client, default_registry(tmp), model="m", on_event=events.append)
            self.assertEqual(kernel.run("inspect"), "done")
            self.assertEqual(events[0], "Objective: inspect")
            self.assertEqual(events[2], "Inspecting files in .")
            self.assertEqual(events[3], "Completed: list")
            self.assertRegex(events[1], r"Waiting for m \(step 1/∞; 2 messages, [\d,]+ context chars\)\.\.\.")
            self.assertRegex(events[4], r"Waiting for m \(step 2/∞; 4 messages, [\d,]+ context chars\)\.\.\.")

    def test_micro_step_caps_tools_and_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [{"function": {"name": "list", "arguments": {}}} for _ in range(4)]
            client = FakeClient([
                {"role": "assistant", "content": "inspect", "tool_calls": calls},
                {"role": "assistant", "content": "done"},
            ])
            events = []
            kernel = AgentKernel(client, default_registry(tmp), model="m",
                                 max_tools_per_step=2, action_tokens=123,
                                 on_event=events.append)
            self.assertEqual(kernel.run("task"), "done")
            self.assertEqual(len(kernel.messages[2]["tool_calls"]), 2)
            self.assertEqual(client.seen[0]["options"], {"num_predict": 123})
            self.assertIn("Deferred 2 tool calls to keep this step focused", events)
            observations = [m for m in kernel.messages if m["role"] == "tool"]
            self.assertEqual(len(observations), 4)

    def test_deferred_calls_are_preserved_and_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [{"function": {"name": "list", "arguments": {}}} for _ in range(3)]
             # Three calls at once but a per-step budget of two, so the third is
             # carried over and must still execute rather than be dropped.
            client = FakeClient([
                {"role": "assistant", "content": "inspect", "tool_calls": calls},
                {"role": "assistant", "content": "done"},
            ])
            events = []
            kernel = AgentKernel(client, default_registry(tmp), model="m",
                                 max_tools_per_step=2, on_event=events.append)
            self.assertEqual(kernel.run("task"), "done")
            observations = [m for m in kernel.messages if m["role"] == "tool"]
            self.assertEqual(len(observations), 3)
            self.assertIn("Deferred 1 tool call to keep this step focused", events)

    def test_large_deferred_queue_stays_within_batch_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [{"function": {"name": "list", "arguments": {}}} for _ in range(7)]
            client = FakeClient([
                {"role": "assistant", "content": "inspect", "tool_calls": calls},
                {"role": "assistant", "content": "done"},
            ])
            events = []
            kernel = AgentKernel(client, default_registry(tmp), model="m",
                                 max_tools_per_step=2, on_event=events.append)
            self.assertEqual(kernel.run("task"), "done")
            self.assertEqual(len(client.seen), 2)
            batches = [message["tool_calls"] for message in kernel.messages
                       if message.get("role") == "assistant" and message.get("tool_calls")]
            self.assertEqual([len(batch) for batch in batches], [2, 2, 2, 1])
            self.assertEqual(len([m for m in kernel.messages if m["role"] == "tool"]), 7)

    def test_observation_limit_and_context_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(FakeClient([]), default_registry(tmp), model="m",
                                 observation_limit=200, context_limit=300)
            limited = json.loads(kernel.observation({"ok": True, "result": "x" * 1000}))
            self.assertTrue(limited["truncated"])
            kernel.messages.extend({"role": "tool", "content": "x" * 200} for _ in range(6))
            kernel.compact_context()
            self.assertTrue(kernel.messages[1]["content"].startswith("[compacted "))

    def test_length_limited_turn_retries_once_without_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient([
                {"role": "assistant", "content": "partial", "thinking": "unfinished",
                 "_done_reason": "length"},
                {"role": "assistant", "content": "complete", "_done_reason": "stop"},
            ])
            events = []
            kernel = AgentKernel(client, default_registry(tmp), model="m",
                                 action_tokens=100, on_event=events.append)
            self.assertEqual(kernel.run("task"), "complete")
            self.assertNotIn("_done_reason", kernel.messages[2])
            self.assertNotIn("partial", [item.get("content") for item in kernel.messages])
            self.assertEqual(client.seen[0]["options"], {"num_predict": 100})
            self.assertIsNone(client.seen[1]["options"])
            self.assertIn("Generation budget reached (100 tokens); retrying once without an action-token cap", events)

    def test_model_decision_and_natural_tool_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            client = FakeClient([
                {"role": "assistant", "content": "I'll read the overview first.", "tool_calls": [
                    {"function": {"name": "read", "arguments": '{"path":"README.md"}'}}]},
                {"role": "assistant", "content": "done"},
            ])
            AgentKernel(client, default_registry(tmp), model="m", on_event=events.append).run("inspect")
            self.assertIn("I'll read the overview first.", events)
            self.assertIn("Reading README.md", events)

    def test_unknown_and_handler_failure_are_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(tmp)
            registry.register(Tool("bad", "bad", {"type": "object"}, lambda: 1 / 0))
            self.assertFalse(registry.execute("missing", {})["ok"])
            self.assertIn("failed", registry.execute("bad", {})["error"])
            self.assertFalse(registry.execute("bad", "not json")["ok"])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / "file.txt").write_text("alpha\nbeta\n")
    def tearDown(self): self.temp.cleanup()

    def test_read_list_search(self):
        registry = default_registry(self.root)
        self.assertEqual(registry.execute("read", {"path": "file.txt", "start": 2})["result"], "2: beta")
        self.assertIn("file.txt", registry.execute("list", {})["result"])
        self.assertIn("file.txt:1:alpha", registry.execute("search", {"pattern": "alp"})["result"])

    def test_approval_denial_and_yes(self):
        denied = default_registry(self.root, approve=lambda *_: False)
        self.assertIn("denied", denied.execute("write", {"path": "x", "content": "x"})["error"])
        allowed = default_registry(self.root, approve=lambda *_: True)
        self.assertTrue(allowed.execute("write", {"path": "x", "content": "x"})["ok"])
        auto = default_registry(self.root, yes=True, approve=lambda *_: self.fail("prompted"))
        self.assertTrue(auto.execute("write", {"path": "y", "content": "y"})["ok"])

    def test_invalid_privileged_arguments_do_not_prompt(self):
        registry = default_registry(self.root, approve=lambda *_: self.fail("approval requested"))
        missing = registry.execute("write", {"content": "x"})
        self.assertFalse(missing["ok"])
        self.assertIn("missing required argument: path", missing["error"])
        unexpected = registry.execute("shell", {"command": "true", "surprise": True})
        self.assertFalse(unexpected["ok"])
        self.assertIn("unexpected argument: surprise", unexpected["error"])
        wrong_type = registry.execute("shell", {"command": 123})
        self.assertFalse(wrong_type["ok"])
        self.assertIn("expected string", wrong_type["error"])

    def test_traversal_symlink_and_search_containment(self):
        outside = Path(self.temp.name).parent / (Path(self.temp.name).name + "-outside")
        outside.mkdir(exist_ok=True); (outside / "secret").write_text("secret")
        try:
            (self.root / "link").symlink_to(outside, target_is_directory=True)
            registry = default_registry(self.root)
            self.assertFalse(registry.execute("read", {"path": "../x"})["ok"])
            self.assertFalse(registry.execute("read", {"path": "link/secret"})["ok"])
            self.assertEqual(registry.execute("search", {"pattern": "secret"})["result"], [])
        finally:
            (outside / "secret").unlink(); outside.rmdir()

    def test_edit_exact_match(self):
        registry = default_registry(self.root, yes=True)
        self.assertTrue(registry.execute("edit", {"path": "file.txt", "old": "alpha", "new": "gamma"})["ok"])
        self.assertFalse(registry.execute("edit", {"path": "file.txt", "old": "missing", "new": "x"})["ok"])
        (self.root / "file.txt").write_text("x x")
        self.assertFalse(registry.execute("edit", {"path": "file.txt", "old": "x", "new": "y"})["ok"])

    def test_shell_status_capture_truncation_timeout(self):
        registry = default_registry(self.root, yes=True, output_limit=5, shell_timeout=.1)
        result = registry.execute("shell", {"command": "printf 123456; printf err >&2; exit 4"})["result"]
        self.assertEqual(result["exit_status"], 4); self.assertIn("truncated", result["stdout"]); self.assertEqual(result["stderr"], "err")
        timed = registry.execute("shell", {"command": "sleep 1"})
        self.assertFalse(timed["ok"]); self.assertIn("timed out", timed["error"])


class CliTests(unittest.TestCase):
    def test_defaults_and_environment(self):
        args = parser({}).parse_args([])
        self.assertEqual((args.model, args.kernel), (DEFAULT_MODEL, DEFAULT_KERNEL))
        self.assertEqual(args.timeout, DEFAULT_TIMEOUT)
        args = parser({"H3N_MODEL": "x", "H3N_KERNEL": "direct", "OLLAMA_HOST": "http://x",
                       "H3N_TIMEOUT": "45"}).parse_args([])
        self.assertEqual((args.model, args.kernel, args.host, args.timeout), ("x", "direct", "http://x", 45.0))
        self.assertTrue(args.stream); self.assertTrue(args.show_reasoning)
        self.assertEqual((args.max_tools_per_step, args.action_tokens,
                          args.observation_limit, args.context_limit), (3, 0, 8000, 50000))
        args = parser({}).parse_args(["--no-stream", "--hide-reasoning"])
        self.assertFalse(args.stream); self.assertFalse(args.show_reasoning)

    def test_version_option(self):
        from h3n import __version__
        output = io.StringIO()
        with mock.patch("sys.stdout", new=output):
            with self.assertRaises(SystemExit) as ctx:
                parser().parse_args(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"h3n {__version__}")

    def test_direct_stream_parsing(self):
        client = mock.Mock(); client.stream_chat.return_value = iter(["hel", "lo"]); output = io.StringIO()
        history = [{"role": "system", "content": "s"}]
        self.assertEqual(run_direct(client, "m", "s", "q", history, output), "hello")
        self.assertEqual(output.getvalue(), "hello\n"); self.assertEqual(history[-1]["content"], "hello")

    def test_direct_preserves_thinking_and_content_transition(self):
        client = mock.Mock()
        def stream_chat(**kwargs):
            kwargs["on_thinking"]("thought ")
            yield "answer"
        client.stream_chat.side_effect = stream_chat
        history = [{"role": "system", "content": "s"}]
        thoughts, transitions = [], []
        run_direct(client, "m", "s", "q", history, io.StringIO(),
                   on_thinking=thoughts.append, on_content_start=lambda: transitions.append(True))
        self.assertEqual(thoughts, ["thought "])
        self.assertEqual(transitions, [True])
        self.assertEqual(history[-1]["thinking"], "thought ")

    def test_terminal_text_removes_systematic_markdown_escapes(self):
        escaped = r"\- **No:** \`src/h3n/\_\_init\_\_.py\`"
        self.assertEqual(terminal_text(escaped), "- **No:** `src/h3n/__init__.py`")
        self.assertEqual(terminal_text(r"\`python\` isn't available"), "`python` isn't available")
        self.assertEqual(terminal_text(r"regex \d+ and \w+"), r"regex \d+ and \w+")
        self.assertEqual(terminal_text(r"C:\temp\file"), r"C:\temp\file")

    def test_progress_reporter_is_plain_when_redirected(self):
        output = io.StringIO()
        reporter = ProgressReporter(output, {})
        reporter("Waiting for model (step 1/20)...")
        reporter("Reading README.md")
        self.assertEqual(output.getvalue(),
                         "h3n 🧠 Model is reviewing observations · model (step 1/20)\n"
                         "h3n 📖 Reading README.md\n")
        self.assertNotIn("\033[", output.getvalue())

    @mock.patch("h3n.cli.AgentKernel")
    @mock.patch("h3n.cli.default_registry")
    def test_cli_routes_to_kernel(self, registry, kernel):
        from h3n.cli import main
        kernel.return_value.run.return_value = "ok"
        with mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(main(["-y", "task"]), 0)
        kernel.return_value.run.assert_called_once_with("task")


class TransportTests(unittest.TestCase):
    def test_streamed_chat_accumulates_content_thinking_and_tools(self):
        body = b''.join([
            b'{"message":{"role":"assistant","thinking":"plan "}}\n',
            b'{"message":{"content":"I will inspect.","tool_calls":[{"function":{"name":"list","arguments":{}}}]}}\n',
            b'{"message":{"thinking":"more","content":" Next."},"done":true}\n',
        ])
        seen = []
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body)) as opened:
            message = OllamaClient().chat(model="m", messages=[], tools=[], on_chunk=seen.append)
        payload = json.loads(opened.call_args.args[0].data)
        self.assertTrue(payload["stream"])
        self.assertEqual(message["content"], "I will inspect. Next.")
        self.assertEqual(message["thinking"], "plan more")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "list")
        self.assertIsNone(message.get("_done_reason"))
        self.assertEqual(len(seen), 3)

    def test_non_streaming_fallback(self):
        body = b'{"message":{"role":"assistant","content":"complete"},"done":true,"done_reason":"stop"}'
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body)) as opened:
            message = OllamaClient().chat(model="m", messages=[], stream=False)
        self.assertFalse(json.loads(opened.call_args.args[0].data)["stream"])
        self.assertEqual(message["content"], "complete")
        self.assertEqual(message["_done_reason"], "stop")

    def test_connection_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=URLError("refused")):
            with self.assertRaisesRegex(OllamaError, "cannot reach"):
                OllamaClient().chat(model="m", messages=[])

    def test_http_and_missing_model_errors(self):
        for code, body, phrase in [(500, b"broken", "HTTP 500"), (404, b"model missing", "model not found")]:
            error = HTTPError("u", code, "x", {}, io.BytesIO(body))
            with mock.patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(OllamaError, phrase):
                    OllamaClient().chat(model="m", messages=[])


class EnvironmentContextTests(unittest.TestCase):
    def test_workspace_appears_in_model_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            system = compose_system_prompt("base system", workspace, "/usr/bin/python3")
            kernel = AgentKernel(FakeClient([{"role": "assistant", "content": "done"}]),
                                 default_registry(workspace), model="m", system=system)
            kernel.run("inspect")
            model_system = kernel.messages[0]["content"]
            self.assertEqual(kernel.messages[0]["role"], "system")
            self.assertIn(str(workspace), model_system)
            self.assertIn("Workspace path (absolute):", model_system)

    def test_active_python_appears_in_model_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            system = compose_system_prompt("base", workspace, "/usr/bin/python3")
            kernel = AgentKernel(FakeClient([{"role": "assistant", "content": "done"}]),
                                 default_registry(workspace), model="m", system=system)
            kernel.run("inspect")
            model_system = kernel.messages[0]["content"]
            self.assertIn("/usr/bin/python3", model_system)
            self.assertIn("Python executable (absolute):", model_system)

    def test_active_python_defaults_to_current_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = environment_context(Path(tmp))
            self.assertIn(str(sys.executable), text)

    def test_custom_system_text_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            text = compose_system_prompt("MY CUSTOM INSTRUCTIONS", workspace, "/usr/bin/python3")
            self.assertIn("MY CUSTOM INSTRUCTIONS", text)
            self.assertIn("## Runtime environment", text)
            self.assertLess(text.index("MY CUSTOM INSTRUCTIONS"), text.index("## Runtime environment"))

    def test_environment_context_exposes_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            text = environment_context(workspace, "/usr/bin/python3")
            for sensitive in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"):
                self.assertNotIn(sensitive, text)

    def test_context_is_not_duplicated_across_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            system = compose_system_prompt("base", workspace, "/usr/bin/python3")
            kernel = AgentKernel(FakeClient([
                 {"role": "assistant", "content": "one"},
                 {"role": "assistant", "content": "two"},
             ]), default_registry(workspace), model="m", system=system)
            kernel.run("first turn")
            kernel.run("second turn")
            systems = [m for m in kernel.messages if m["role"] == "system"]
            self.assertEqual(len(systems), 1)
            self.assertEqual(systems[0]["content"].count("## Runtime environment"), 1)

    def test_path_with_spaces_is_unambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            spaced = Path(tmp, "a b").resolve()
            spaced.mkdir()
            text = environment_context(spaced, "/usr/bin/python3")
            quoted = '"' + str(spaced) + '"'
            self.assertIn(quoted, text)
            self.assertNotIn(str(spaced) + "\n", text)


class UnlimitedAndCompletionTests(unittest.TestCase):
    def test_default_is_unlimited(self):
        self.assertEqual(parser({}).parse_args([]).max_steps, 0)

    def test_zero_is_unlimited(self):
        responses = [
             {"role": "assistant", "tool_calls": [
                   {"function": {"name": "list", "arguments": {"path": "."}}}] }
             for _ in range(25)
         ] + [{"role": "assistant", "content": "done"}]
        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(FakeClient(responses), default_registry(Path(tmp)),
                                 model="m", max_steps=0)
            self.assertEqual(kernel.run("go"), "done")
            self.assertEqual(sum(1 for m in kernel.messages if m.get("role") == "tool"), 25)

    def test_positive_limit_is_enforced(self):
        responses = [
             {"role": "assistant", "tool_calls": [
                   {"function": {"name": "list", "arguments": {"path": "."}}}] }
             for _ in range(6)
         ] + [{"role": "assistant", "content": "done"}]
        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(FakeClient(responses), default_registry(Path(tmp)),
                                 model="m", max_steps=5)
            with self.assertRaises(StepLimitError):
                kernel.run("go")

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(SystemExit):
            parser({}).parse_args(["--max-steps", "-1"])

    def test_progress_shows_infinity(self):
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(FakeClient([
                     {"role": "assistant", "tool_calls": [
                          {"function": {"name": "list", "arguments": {}}}
                      ]},
                     {"role": "assistant", "content": "done"},
                 ]), default_registry(Path(tmp)), model="m", on_event=events.append)
            kernel.run("inspect")
        self.assertTrue(any("step 1/∞" in event for event in events))

    def test_ctrl_c_stops_unlimited_run(self):
        class InterruptingClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    raise KeyboardInterrupt
                return {"role": "assistant", "tool_calls": [
                         {"function": {"name": "list", "arguments": {}}}]}

        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(InterruptingClient(), default_registry(Path(tmp)),
                                 model="m", max_steps=0)
            with self.assertRaises(KeyboardInterrupt):
                kernel.run("go")

    def test_unlimited_deferred_calls_drain(self):
        calls = [{"function": {"name": "list", "arguments": {}}} for _ in range(30)]
        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(FakeClient([
                     {"role": "assistant", "tool_calls": calls},
                     {"role": "assistant", "content": "done"},
                 ]), default_registry(Path(tmp)), model="m", max_tools_per_step=2,
                                  max_steps=0)
            self.assertEqual(kernel.run("go"), "done")
            self.assertEqual(sum(1 for m in kernel.messages if m.get("role") == "tool"), 30)

    def test_verification_after_edit_succeeds(self):
        write_call = {"function": {"name": "write",
                                    "arguments": {"path": "a.txt", "content": "hello"}}}
        verify_call = {"function": {"name": "shell", "arguments": {"command": "true"}}}
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            kernel = AgentKernel(FakeClient([
                     {"role": "assistant", "tool_calls": [write_call, verify_call]},
                     {"role": "assistant", "content": "complete"},
                 ]), default_registry(workspace, yes=True), model="m", max_steps=0)
            kernel.run("make and verify")
            notes = [m for m in kernel.messages
                     if m.get("role") == "user" and "Verification succeeded" in m.get("content", "")]
            self.assertEqual(len(notes), 1)
            self.assertIn("no unverified changes remain", notes[0]["content"])
            self.assertFalse(kernel.completion.changes_unverified)
            self.assertTrue(kernel.completion.verified)

    def test_failed_verification_leaves_changes_unverified(self):
        write_call = {"function": {"name": "write",
                                    "arguments": {"path": "a.txt", "content": "hello"}}}
        failed_call = {"function": {"name": "shell", "arguments": {"command": "false"}}}
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            kernel = AgentKernel(FakeClient([
                     {"role": "assistant", "tool_calls": [write_call, failed_call]},
                     {"role": "assistant", "content": "still fixing"},
                 ]), default_registry(workspace, yes=True), model="m", max_steps=0)
            kernel.run("make and verify")
            self.assertTrue(kernel.completion.changes_unverified)
            self.assertFalse(kernel.completion.verified)
            self.assertFalse(any("Verification succeeded" in m.get("content", "")
                                 for m in kernel.messages if m.get("role") == "user"))

    def test_repeated_no_progress_reports_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "a.txt").write_text("same\n", encoding="utf-8")
            read_call = {"role": "assistant", "tool_calls": [
                     {"function": {"name": "read", "arguments": {"path": "a.txt"}}}]}
            kernel = AgentKernel(FakeClient([read_call, read_call, read_call,
                     {"role": "assistant", "content": "done"}]),
                 default_registry(workspace), model="m", max_steps=0)
            kernel.run("stuck")
            self.assertTrue(any("no progress" in m.get("content", "")
                                 for m in kernel.messages if m.get("role") == "user"))

    def test_no_false_success_from_model_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            kernel = AgentKernel(FakeClient([
                     {"role": "assistant",
                      "content": "All requirements are verified and complete!"}]),
                 default_registry(Path(tmp)), model="m", max_steps=0)
            self.assertEqual(kernel.run("done"),
                              "All requirements are verified and complete!")
            self.assertFalse(kernel.completion.verified)
            self.assertFalse(any("Verification succeeded" in m.get("content", "")
                                 for m in kernel.messages if m.get("role") == "user"))


if __name__ == "__main__": unittest.main()
