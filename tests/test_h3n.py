import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

from h3n.cli import DEFAULT_KERNEL, DEFAULT_MODEL, DEFAULT_TIMEOUT, ProgressReporter, parser, run_direct, terminal_text
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
            self.assertEqual(events, ["Waiting for m (step 1/20)...", "Inspecting files in .",
                                      "Waiting for m (step 2/20)..."])

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

    def test_direct_stream_parsing(self):
        client = mock.Mock(); client.stream_chat.return_value = iter(["hel", "lo"]); output = io.StringIO()
        history = [{"role": "system", "content": "s"}]
        self.assertEqual(run_direct(client, "m", "s", "q", history, output), "hello")
        self.assertEqual(output.getvalue(), "hello\n"); self.assertEqual(history[-1]["content"], "hello")

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


if __name__ == "__main__": unittest.main()
