# h3n

`h3n` is a lightweight, standalone coding-agent microkernel that runs local models
through Ollama. It owns its conversation loop, tools, approvals, workspace boundary,
and CLI; it does not install, import, wrap, or invoke Pi, OpenHands, Goose, or OpenCode.

## Install and run

Python 3.10+ and a running Ollama installation are required.

```sh
python -m pip install -e .
h3n "Inspect this repository"
```

Without installation, point Python at the source tree:

```sh
PYTHONPATH=src python -m h3n -m qwen3.8:27b "Inspect this repository"
```

The default tag is `qwen3.8:27b-mlx`; your installed Ollama tag may differ, so pass
it explicitly with `-m`. Omitting the task starts an interactive session whose
history is preserved. Exit using `/exit`, `/quit`, Ctrl-D, or Ctrl-C.

Agent progress is written to stderr while the final, terminal-friendly response is
written to stdout. A repository inspection may take several model/tool rounds; lines
such as `Waiting for ...` and `Running tool: ...` show that the agent is still active.
Ollama responses stream by default. A model's `thinking` stream is shown on stderr
when provided; use `--hide-reasoning` for privacy or quieter output, or `--no-stream`
for compatibility and debugging. Reasoning can be verbose and may contain sensitive
workspace context.

By default, `write`, `edit`, and `shell` actions require approval. Pass `--yes` (or
`-y`) to approve them automatically. Reads, listings, and searches never prompt.
All file operations remain confined to the current workspace, including through
symlinks. Shell commands run in that workspace with bounded output and runtime.

For tool-free streamed conversation:

```sh
h3n --kernel direct -m llama3.2 "Write a haiku"
```

To inspect a model's streamed reasoning while dogfooding:

```sh
h3n --show-reasoning --timeout 600 \
  "Add a --version option, update tests and README, then run all tests."
```

## Options and environment

```text
--version               Show program version and exit
-m, --model MODEL       Ollama model tag
--host URL              Ollama server (default http://localhost:11434)
--timeout SECONDS       Ollama request timeout (default 300)
-s, --system TEXT       Replace the system prompt
--kernel h3n|direct     Agent loop or tool-free chat
--no-stream             Disable response streaming
--show-reasoning        Show model thinking on stderr (default)
--hide-reasoning        Hide model thinking
-y, --yes               Skip privileged-action prompts
--max-steps N           Bound agent iterations; 0 is unlimited (default)
--max-tools-per-step N  Bound each tool batch (default 3)
--action-tokens N       Optional generation cap; 0 is unlimited (default)
--observation-limit N   Characters retained per tool result (default 8000)
--context-limit N       Compact old tool output above this size (default 50000)
```

The bounded tool and observation defaults encourage small action/observation cycles.
Generation is unlimited by default because reasoning models may consume a token cap
before producing a tool call. `h3n` shows the current objective, completed tools, and
deferred calls. When a batch exceeds the per-step limit, the deferred calls are kept and
executed in a later step so no call is silently lost. Old tool results are compacted
mechanically when the conversation grows, while the four most recent observations remain
intact.

`H3N_MODEL`, `H3N_KERNEL`, `H3N_TIMEOUT`, and `OLLAMA_HOST` provide defaults; command-line options
override them. The agent reports Ollama connectivity, model, HTTP, tool, timeout,
permission, containment, and step-limit failures concisely.

## Unlimited steps and verification-aware completion

The default `--max-steps` is `0`, which means an **unlimited** run: the kernel keeps
working until the model returns a final answer with no tool calls and no deferred calls
remain. Progress is shown as `step N/∞`. Pass a positive integer (for example
`--max-steps 5`) to impose an optional finite limit; the run then stops with a clear
step-limit message once the limit is reached. Negative values are rejected at the CLI
with `--max-steps must be 0 (unlimited) or positive`. There is no hidden default cap:
the only stopping conditions are a successful final answer, a positive `--max-steps`
limit, or user interruption.

Unlimited runs are bounded in practice by a lightweight **completion controller** that
only advises the model and never terminates a run or imposes a step limit:

- It tracks whether workspace-changing tools (`write`, `edit`) have run since the last
  successful verification.
- It recognizes a successful verification by a `shell` command's exit status `0`, not by
  anything the model claims, so a model cannot assert completion in plain text.
- After a successful verification it appends a concise kernel observation stating that
  verification succeeded, whether unverified changes remain, and that the model should
  return its final answer immediately if every requirement is complete. It never claims
  success from a zero exit status alone, and it never forces completion while writes or
  edits remain unverified.
- It detects three consecutive identical tool calls with identical arguments and results
  and then appends an actionable observation asking the model to choose a different
  action; this does not end the run or count toward a step limit.

Kernel observations are appended as ordinary user messages so the tool call/response
history stays valid.

### Risks

Because a default run is unbounded, an unclear task or a model that keeps issuing the
same tool call can consume model time and tokens without finishing. The repetition
observation and the per-step verification notes reduce but do not eliminate this. Use a
positive `--max-steps` for bounded work, and interrupt an unresponsive run with
`Ctrl-C`, which stops an unlimited run cleanly.

## Runtime environment context

Before each session, `h3n` appends a short, trusted runtime-environment block to the
system prompt. It states the resolved absolute workspace path and the active Python
executable, and tells the model that file tools and shell commands already run from the
workspace, so it should not guess absolute paths such as `/workspace` or prefix shell
commands with `cd`, and should use the reported Python executable for tests and Python
commands. The block is added once per session, so interactive turns do not duplicate it.
A prompt supplied through `-s`/`--system` is preserved; the environment context is only
appended. Paths containing spaces are rendered in quotes so they cannot be misread.
Unrelated environment variables, credentials, tokens, and secrets are never exposed.

## Tests

```sh
python -m unittest discover -s tests -v
```
