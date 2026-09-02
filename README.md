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
-m, --model MODEL       Ollama model tag
--host URL              Ollama server (default http://localhost:11434)
--timeout SECONDS       Ollama request timeout (default 300)
-s, --system TEXT       Replace the system prompt
--kernel h3n|direct     Agent loop or tool-free chat
--no-stream             Disable response streaming
--show-reasoning        Show model thinking on stderr (default)
--hide-reasoning        Hide model thinking
-y, --yes               Skip privileged-action prompts
--max-steps N           Bound agent iterations (default 20)
```

`H3N_MODEL`, `H3N_KERNEL`, `H3N_TIMEOUT`, and `OLLAMA_HOST` provide defaults; command-line options
override them. The agent reports Ollama connectivity, model, HTTP, tool, timeout,
permission, containment, and step-limit failures concisely.

## Tests

```sh
python -m unittest discover -s tests -v
```
