# RLM IDE

A local-first desktop IDE for Linux and Windows with a coding agent that thinks in code. The agent
writes Python in a persistent REPL, calls project tools from that code, and streams every step
(thinking, cells, tool calls, diffs, approvals) into the agent panel.

- **Built-in models, no account needed.** FrogNano 4B (ONNX, WebGPU) runs the agent and bge-m3 handles memory embeddings. Both run on your machine.
- **Or any OpenAI-compatible API.** OpenAI, OpenRouter, Ollama, LM Studio, vLLM or a custom URL, chosen per role (agent, summarizer, embeddings). Keys are encrypted with the OS keychain.
- **Long sessions.** Context budgets, automatic compaction with checkpoints, a pinned goal card, and long-term memory across chats.
- **Safe by default.** Edits are reviewable diffs you can keep or revert, and side effects need approval. The Python kernel is sandboxed (bubblewrap on Linux) and there is no telemetry.
- **Team hub.** Host groups and projects over LAN or SSH, with protected branches, pull requests, bug reports routed to the commit author, and notifications.
- **Models hub.** Download models from Hugging Face and deploy them locally (vLLM, llama.cpp, ONNX, sentence-transformers) behind an API key.

## Install

Download the AppImage, the `.deb` or the Windows installer from
[Releases](https://github.com/adamsmith988/rlm-ide/releases).

```bash
chmod +x RLM-IDE-0.1.0.AppImage && ./RLM-IDE-0.1.0.AppImage
# or
sudo apt install ./rlm-ide_0.1.0_amd64.deb
```

## Screenshots

| Section | Dark | Light |
| --- | --- | --- |
| Explorer + agent chat | ![](docs/screenshots/dark-explorer.png) | ![](docs/screenshots/light-explorer.png) |
| Search | ![](docs/screenshots/dark-search.png) | ![](docs/screenshots/light-search.png) |
| Source control | ![](docs/screenshots/dark-source-control.png) | ![](docs/screenshots/light-source-control.png) |
| Chats | ![](docs/screenshots/dark-chats.png) | ![](docs/screenshots/light-chats.png) |
| Team project | ![](docs/screenshots/dark-team-project.png) | ![](docs/screenshots/light-team-project.png) |
| Bug reports | ![](docs/screenshots/dark-bug-reports.png) | ![](docs/screenshots/light-bug-reports.png) |
| Models hub | ![](docs/screenshots/dark-models-hub.png) | ![](docs/screenshots/light-models-hub.png) |
| Memory | ![](docs/screenshots/dark-memory.png) | ![](docs/screenshots/light-memory.png) |
| Settings | ![](docs/screenshots/dark-settings.png) | ![](docs/screenshots/light-settings.png) |

The screenshots are produced by `node evals/sections-run.mjs`, which also checks every section in both themes.

## Layout

```
apps/desktop        Electron app: main, preload, orchestrator + inference utility processes, React renderer
packages/contracts  zod schemas shared by every process (settings, IPC, turn events)
packages/core       domain, use cases and ports (hexagonal core, no I/O)
packages/adapters   SQLite/sqlite-vec, Python kernel, LLM providers, tools
packages/inference  local model runtime and Model Manager
python/rlm_worker   the REPL kernel
prompts/            system prompts and templates
evals/              dogfood evals that drive the built app
docs/               architecture, decisions, specs, legacy reference
```

See [docs/architecture.md](docs/architecture.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

## Develop

Requirements: Node 22+, pnpm 10, Python 3.10+ (or run `pnpm fetch:python` to fetch the bundled runtime), and git.

```bash
pnpm install
pnpm fetch:models -- --only builtin:bge-m3   # embedder (~600 MB). FrogNano installs from Settings → Models.
pnpm dev                                     # Ubuntu 23.10+: pnpm dev:nosandbox
```

On Ubuntu, AppArmor blocks the Chromium sandbox for unpackaged Electron. Use `pnpm dev:nosandbox`
in development. Packaged AppImage and deb builds keep the sandbox.

To use an API model instead of the built-in one, open Settings → Models → Add provider, pick a preset,
paste the key, press Detect, and set the model as the Agent role.

## Test

```bash
pnpm typecheck
pnpm test          # unit + integration (scripted model, real kernel and tools)
pnpm test:py       # kernel tests
pnpm build && pnpm eval -- --scenarios quotes-app,fix-bugs
```

The evals launch the built app with a throwaway profile, type prompts into the agent panel, and score
the result. Scoring covers the files, tests, a headless render with click probes, and an LLM rubric
judge. Results go to `evals/results/<timestamp>/summary.md`. The defaults target a local
OpenAI-compatible server; set `EVAL_API_URL`, `EVAL_API_MODEL`, `EVAL_API_KEY` (and `EVAL_JUDGE_*`)
to change that. Pass `--model local` to evaluate the built-in model. See
[docs/quality-report.md](docs/quality-report.md) for the latest results.

## Package

```bash
pnpm fetch:python
pnpm fetch:models -- --only builtin:bge-m3
pnpm dist:linux    # AppImage + deb
pnpm dist:win      # NSIS installer (run on Windows)
```

`scripts/make-icons.cjs` regenerates `resources/icons`. CI (`.github/workflows/release.yml`) runs
the tests, then packages on Linux and Windows and publishes on `v*` tags.
