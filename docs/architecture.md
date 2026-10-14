# RLM IDE architecture

RLM IDE is an Electron desktop IDE (Linux and Windows) whose agent runs the RLM engine: the model
thinks in a Python REPL, calls tools from code, and the host streams every step to the UI.

The code is hexagonal. The domain and use cases live in `packages/core` and only talk to the
outside world through ports. Adapters implement those ports. Electron processes are the
composition roots that wire adapters into the core.

```
                    ┌──────────────────────── renderer (React) ───────────────────────┐
                    │  editor · explorer · terminal · agent panel · settings · memory  │
                    └───────────────▲──────────────────────────┬──────────────────────┘
                         turn events │ (push)        typed IPC  │ (contracts/ipc.ts, zod)
                    ┌───────────────┴──────────────────────────▼──────────────────────┐
                    │ main process: windows, fs (confined), pty, git, settings,       │
                    │ safeStorage secrets, inference host, orchestrator host          │
                    └──────┬───────────────────────┬──────────────────────────┬───────┘
                MessagePort│              utilityProcess                hidden window
          ┌────────────────▼─────────┐  ┌───────────────────┐  ┌──────────────────────────┐
          │ orchestrator (Node)      │  │ inference-cpu     │  │ inference window (WebGPU)│
          │  core use cases          │  │ onnxruntime-node  │  │ transformers.js + ORT web│
          │  SQLite + sqlite-vec     │  │ FrogNano fallback │  │ FrogNano q4f16           │
          │  bge-m3 embedder         │  └───────────────────┘  └──────────────────────────┘
          │  Python kernel (JSONL)   │
          └──────────────────────────┘
```

## Packages

| Package | Role |
| --- | --- |
| `packages/contracts` | Shared zod schemas: settings, model/provider config, IPC channels, turn events. No logic. |
| `packages/core` | Domain (`domain/`: budgets, tokens, chunking, checkpoints, tool ids), use cases (`application/`: orchestrator, turn runner, session compactor, memory service, tool executor, tool surface) and ports (`ports/index.ts`). No Electron, no I/O. |
| `packages/adapters` | Port implementations: SQLite repos and vector store, workspace confinement, Python kernel, OpenAI-compatible client, local bridge, provider router, built-in tools, shell. |
| `packages/inference` | Local model runtime: FrogNano generation, bge-m3 embeddings, Model Manager (download, import, verify). |
| `apps/desktop` | Electron app. `main/`, `preload/`, `orchestrator/` (utilityProcess), `inference-cpu/` (utilityProcess), `renderer/` (UI and the hidden inference page). |
| `python/rlm_worker` | The REPL kernel the agent's code runs in. Tool groups are importable modules backed by host calls. |
| `prompts/` | System prompts and templates, loaded at runtime through `PromptsPort`. |
| `evals/` | Dogfood harness that drives the built app over CDP and scores what the agent builds. |

## Ports

The core depends only on these interfaces (`packages/core/src/ports/index.ts`):

- **Models:** `ChatModelPort`, `EmbeddingPort`, `ModelCatalogPort` (context and output limits), `ModelRouterPort` (role to model).
- **Storage:** `ConversationRepo`, `MemoryVectorStore`, `PinnedFactsRepo`, `TranscriptSnapshotStore`, `EditJournal`.
- **Execution:** `KernelFactory` / `KernelSession`, `ToolRegistryPort`.
- **Environment:** `PromptsPort`, `ClockPort`, `SettingsPort`, `WorkspacePort`, `LoggerPort`, `StatusPort`.

## Model providers

`ProviderRouter` resolves a model per role: `agent`, `summarizer` (falls back to the agent model) and `embed`.
Every model has a provider of one of two kinds:

- **`local_onnx` (built-in).** Nothing leaves the machine.
  - **FrogNano 4B** (`RASMUS/FrogNano-4B-2609-ONNX`, q4f16) runs through transformers.js on WebGPU in a hidden renderer window. The CPU fallback uses `onnxruntime-node` in a utilityProcess, which works but is slow. The model is installed by the Model Manager on first use.
  - **bge-m3** (q8) runs in-process in the orchestrator and ships inside the app.
- **`openai_compat`.** Any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, LM Studio, vLLM, custom). API keys are encrypted with Electron `safeStorage` and never reach the renderer. `Detect` reads the context length from `/models`.

Budgets come from `ModelCatalogPort`. The history share, compaction trigger, output reserve and memory limits are settings with defaults, never constants in logic.

## A turn

1. The renderer sends `chat:send` with the prompt, the editor context and any @-mentions.
2. The orchestrator builds the context: system prompt, tool surface, pinned facts, recalled memory, compacted history.
3. `TurnRunner` loops. The model streams text and thinking. When it writes a fenced `repl` cell, the kernel runs the cell, tool calls go back to the host through JSONL RPC, and the host checks permissions. A tool that needs approval emits `approval_request` and waits for the user. File edits are journaled as pending edits, which can be reviewed, kept or reverted.
4. Every step is a `TurnEvent` with a sequence number, stored and pushed to the renderer. The UI rebuilds the view with a pure reducer (`chat/model.ts`), so reattaching to a running turn after a reload only replays events.
5. After the turn, memory distills facts and the session compactor checkpoints history once it passes the trigger ratio.

## Security

- `contextIsolation` and `sandbox` are on for every window. The preload exposes only typed channels.
- All file access is confined to the open workspace (`fs/confine.ts`).
- The Python kernel runs in bubblewrap on Linux (no network by default) and in a Job Object with a memory cap on Windows.
- There is no telemetry. Network calls go only to providers you configure and, when you ask, to the Hugging Face model download.
