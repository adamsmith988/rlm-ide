# MASTER PROMPT: Build "RLM IDE", an Electron desktop coding IDE powered only by the RLM engine

> Paste this whole file as the first message to your coding agent (Claude Code, Cursor, etc.) in an empty repo.
> Work through the phases in section 14 in order. Do not skip ahead. Finish and verify each phase before starting the next.

---

## 0. Your role

You are a senior full-stack engineer building a production-quality desktop AI coding IDE, comparable in feel to Cursor and Claude Code. It ships as an Electron app with installers for **Windows** and **Linux**.

The AI brain of this IDE is **one engine only: RLM (Recursive Language Model)**, together with its **memory** (short-term, long-term, pinned facts) and its **compactor** (session compactor + RLM transcript compaction + in-turn window guard). The design is ported from an existing server-side system (the "AIv2 orchestrator"). This prompt tells you exactly what to keep, what to drop, and how to adapt it to a desktop app.

---

## 1. Scope

### In scope
1. Electron desktop app (Windows + Linux installers, auto-update).
2. IDE shell: file explorer, tabbed code editor, integrated terminal, AI chat/agent panel, command palette, settings, diff review.
3. **RLM engine**: persistent Python kernel the model writes code into, tools exposed as kernel callables, sub-agents, context budget, transcript compaction.
4. **Session compactor** (host-level checkpoint compaction of the conversation).
5. **Memory**: pinned facts, short-term (per chat) vector memory, long-term (per user) distilled facts, recall into the prompt.
6. Side-effect safety for tools (approval + per-turn de-duplication).
7. Turns that survive UI reloads (background turn runner with buffered events).
8. The UI design system described in section 10.

### Explicitly OUT of scope (do not build, do not stub)
- Other engines: `native_tools`, `SRLM`, `RRSI`. No engine registry, no manifests scanning, no engine switching, no peer engines.
- Gateway, llm-proxy, storage service, tool-runner microservice, Qdrant server, vLLM, web frontend pages, `active_plugin` page scoping, plugins, RAG corpus / smart search, multi-user ACLs, the "long input" parallel-worker digest.
- Anything cloud-hosted. This is a local-first desktop app. The only network calls are to the LLM/embedding endpoint the user configures (and the auto-updater).

If you are tempted to add something not listed under "In scope", stop and ask.

---

## 2. Architecture (desktop adaptation)

The original server design is a host (owns conversation, compaction, memory, tools) plus engines (decide how to answer one turn). Here there is a host plus exactly one engine, all running locally.

```
┌──────────────────────── Electron ─────────────────────────┐
│ Renderer (React + TS)                                      │
│   IDE shell UI  ⇄  typed IPC (contextBridge only)          │
├────────────────────────────────────────────────────────────┤
│ Main process (Node/TS)                                     │
│   WindowManager, FS service, PTY service (node-pty),       │
│   Settings + SecretStore (safeStorage), AutoUpdater        │
├────────────────────────────────────────────────────────────┤
│ Orchestrator host (Node worker / utilityProcess)           │
│   TurnRunner  → preflight → RLM engine → persist           │
│   SessionCompactor · Memory (pinned/short/long) · Tools    │
│   ModelCatalog · LLM client (OpenAI-compatible)            │
│   SQLite (better-sqlite3 + sqlite-vec)                     │
├────────────────────────────────────────────────────────────┤
│ RLM kernel sidecar (bundled Python, IPython worker)        │
│   JSONL RPC over stdin/stdout ⇄ host `host_request`        │
└────────────────────────────────────────────────────────────┘
```

Mapping from the original system to this app:

| Original | Desktop replacement |
|---|---|
| Gateway → orchestrator over HTTP | Renderer → main → orchestrator worker over typed IPC / MessagePort |
| llm-proxy model catalog + bindings | Local `ModelCatalog` driven by Settings (see 6.1) |
| vLLM chat / embed | Any OpenAI-compatible chat + embeddings endpoint the user configures (OpenAI, OpenRouter, Ollama, LM Studio, vLLM, etc.) |
| Storage service (chat sessions) | Local SQLite database in the user data dir |
| Qdrant | `sqlite-vec` (or LanceDB if sqlite-vec proves unreliable) inside the same SQLite file |
| tool-runner (`tool.json` tools) | In-process tool registry; each tool keeps a `tool.json`-style descriptor |
| Frontend page block | **Editor context block** (open file, selection, diagnostics, cwd, git status) |
| `AIv2_*` env settings | Settings UI + `settings.json`, env-overridable for tests |

### Key principles to preserve
- **The host owns the conversation.** The RLM transcript is derived state and can always be rebuilt from the stored conversation.
- **Numbers come from configuration, prompts are `.md` files, tool behaviour comes from tool descriptors.** No magic constants in logic code.
- **Turns outlive the UI connection.** A turn is a background task that buffers its stream events. Closing the panel, reloading the renderer, or switching tabs only ends the *subscription*. Only an explicit **Stop** cancels the turn.
- **Soft failure.** If the embedding endpoint is down, memory recall and save fail softly: the turn still runs, without recall. Never block a turn on memory.

---

## 3. Tech stack (fixed)

- **Electron** (latest stable), **electron-vite**, **TypeScript** everywhere, **React 18**, **Zustand** for state.
- Editor: **Monaco**. Terminal: **xterm.js** + **node-pty**. Diff: Monaco diff editor.
- Styling: CSS variables (design tokens) + **Tailwind**; no component library that fights the design system.
- Persistence: **better-sqlite3**, migrations via plain `.sql` files; vectors via **sqlite-vec**.
- Python sidecar: bundled **python-build-standalone** with `ipykernel`/IPython and the RLM worker script. Do not depend on a system Python.
- Packaging: **electron-builder** (NSIS `.exe` for Windows; **AppImage + .deb** for Linux, `.rpm` optional), **electron-updater**.
- Tests: Vitest (unit), Playwright for Electron (e2e), pytest for the Python worker.

---

## 4. Repository layout

```
rlm-ide/
├─ apps/desktop/
│  ├─ src/main/            # Electron main: windows, fs, pty, settings, secrets, updater
│  ├─ src/preload/         # contextBridge API (typed, allow-listed channels only)
│  ├─ src/renderer/        # React UI (see section 10)
│  └─ electron-builder.yml
├─ packages/orchestrator/  # runs in a utilityProcess; no Electron imports
│  ├─ src/host/            # turn_runner, preflight, host_context, persist
│  ├─ src/engine/rlm/      # session, kernel client, handlers, context_budget, compaction, compact_resume
│  ├─ src/context/         # session_compactor, planCut, transcript_search
│  ├─ src/memory/          # client, facts, store, pinned_facts
│  ├─ src/tools/           # registry, executor, side_effects, tool_surface, builtin/*
│  ├─ src/llm/             # openai-compatible client, model_catalog, token_estimator
│  ├─ src/db/              # sqlite, migrations/
│  └─ prompts/             # *.md only (see section 12)
├─ python/rlm_worker/      # IPython worker (JSONL RPC), sandbox shims
├─ resources/              # icons, illustrations (SVG), bundled python (per platform, build-time)
├─ scripts/                # fetch-python, build, sign, release
└─ .github/workflows/      # matrix build: windows-latest, ubuntu-latest
```

---

## 5. The RLM engine (core of the product)

The model does not call tools through a JSON `tool_calls` channel. It **writes Python into a persistent kernel**, and tools are **callables inside that kernel**. State (variables, imports, loaded data) persists across cells and across turns of the same chat while the kernel is warm.

### 5.1 Kernel (`python/rlm_worker` + `engine/rlm/kernel`)
- A subprocess IPython worker. Host and worker speak **JSONL RPC over stdin/stdout** (one JSON object per line, request ids, streaming stdout/stderr chunks, cell results, errors).
- Operations: `start`, `execute(cell)`, `interrupt`, `reset`, `shutdown`, plus the reverse channel `host_request` (the kernel asks the host to do something, e.g. call a tool) with a response routed back by id.
- Per-cell timeout, output truncation with a clear "[truncated N chars]" marker, and a hard kill path if interrupt fails.
- **Sandbox**: on Linux use `bwrap` when available, else Landlock, else run unsandboxed with a visible warning in the UI. On Windows there is no bwrap/Landlock: run the worker with a restricted working directory, a Job Object (kill-on-close, memory cap), and rely on the tool approval flow (section 8) as the primary guard. State this limitation honestly in Settings → Security. Never claim sandbox guarantees you do not have.
- The worker's cwd is the open workspace folder. File writes outside the workspace are denied by the tools (not just by the sandbox).

### 5.2 Session (`engine/rlm/session`)
One RLM session per chat. It owns: the kernel, the transcript (code cells + REPL output + assistant text), child sub-agents, an inbox for sub-agent results, and the turn loop:

1. Build the request: system prompt + transcript + host blocks (memory, pinned facts, editor context, clock) + the new user message.
2. Hard-cap with `context_budget`.
3. Stream the model response; extract code cells; execute in the kernel; append output to the transcript.
4. Repeat until the model produces a final answer without another cell, or `max_steps` is reached, or the turn is stopped.
5. Emit stream events for every step (thinking text, code cell, cell output, tool call, final text) so the UI can render them live and replay them after a reload.

### 5.3 Tools as kernel callables
- Tools are exposed with `tools_binding.mode = kernel_callable`.
- `tool_surface` builds, from each tool's JSON Schema descriptor: the Python alias, the namespace it lives under (e.g. `fs.read_file`, `fs.edit_file`, `shell.run`, `git.diff`, `search.grep`), and the **prompt section** documenting every callable (signature, args, return shape, side-effect flag).
- A tool can only be called if its id is in the session's `tool_ids`. The `host_request` handler enforces this on the host side; never trust the kernel.
- Tool errors return as structured results the model can read and recover from, not as crashes.

### 5.4 Sub-agents
- RLM can spawn child sessions from inside the kernel (e.g. `spawn_agent(task, tool_ids=[...])`). Track **depth** and enforce `max_subagent_depth` and a max concurrent children (both settings).
- Each child gets its own kernel and transcript, and only the tools listed in `subagent_tool_ids`. Children return a result to the parent's inbox. Show children as a nested tree in the UI.

### 5.5 Context budget (`context_budget`)
- Hard-cap every request so that `estimated_input_tokens + max_tokens < context_length` of the configured model.
- When over budget: first truncate oldest REPL outputs, then trigger transcript compaction (5.7), and only then fail with a readable error. Use a safe token estimator (tiktoken-compatible or provider-reported usage with a conservative fallback).

### 5.6 RLM memory ladder (simplified: one engine)
Transcript restore order: **warm session → disk snapshot → reseed from the stored conversation**.
- Warm: session and kernel are alive in the orchestrator process.
- Disk snapshot: transcript (and a note of which stored turn it reaches) saved per chat, so an app restart can resume cheaply. **Kernel variables are not restored after a restart.** Tell the model, in a short system note, that variables from earlier cells may need to be re-created.
- Reseed: if the stored conversation no longer reaches the transcript's last turn (e.g. session compaction folded it), rebuild the transcript from **checkpoint + verbatim recent tail**.
- (The original also appended turns answered by other engines. You have no other engines, so skip that.)

### 5.7 RLM transcript compaction (`engine/rlm/compaction`, `compact_resume`)
- Fires when the private transcript reaches **90% of `context_length`** (default; setting `rlm.compactTriggerRatio`).
- Keeps a **verbatim recent tail** plus a **structured checkpoint**. **Never cut a tool call from its result.**
- Writes the **same checkpoint format** as the session compactor (section 6.2) so both layers stay consistent.
- `compact_resume`: after compaction, the session continues seamlessly with the checkpoint as "earlier work".

---

## 6. Compactor

There are two host-visible layers plus a guard, all writing **one checkpoint format**.

### 6.1 Model catalog (feeds every budget)
`ModelCatalog` replaces llm-proxy. Per configured model it provides `context_length` and `max_output_tokens`.
- Auto-detect where the provider exposes it (e.g. Ollama `/api/show`, OpenRouter model metadata, vLLM `/v1/models` `max_model_len`).
- Otherwise the user enters them in Settings → Models. Block sending until `context_length` is known. Never guess silently.
- Stamp both values onto the per-turn engine config before each turn (`prepare_orchestrator_config`).

### 6.2 Session compactor (`context/session_compactor`)
Runs in preflight on every turn, before the engine.

Budget (all ratios are settings, not constants):
- `window` = model `context_length`
- `output_reserve` = `max_output_tokens`, capped at `max_output_ratio` × window
- `history` = window share allowed for conversation (`history_share`)
- `trigger` = `history × trigger_ratio`
- `keep` = `history × keep_recent_ratio`

> The source document does not give numeric defaults for these ratios. Start with `history_share 0.6`, `trigger_ratio 0.8`, `keep_recent_ratio 0.35`, `max_output_ratio 0.25`, `max_rows_ratio` tuned for your row counting, then tune with the tests in section 14. Treat these as starting values, not requirements.

What is measured: the conversation the engine will receive, i.e. the carried checkpoint + stored turns since it + the new user message. A **second trigger** fires on row count (`max_rows_ratio`).

How it compacts:
- The **oldest whole turns** fold into a structured checkpoint.
- The **newest turns stay verbatim** (`planCut`). A very long newest turn is still kept when it fits `newest_limit`, so `kept_turns` never collapses to 0.
- The checkpoint is **updated from the previous checkpoint**, never restarted, so facts survive many compactions.

Checkpoint sections (exact):
1. **Goal** (the overall objective, not only the latest request)
2. **Constraints & Preferences**
3. **Facts & Values** (verbatim, task-relevant only)
4. **Tool Results** (tool, arguments, values)
5. **Requests answered**
6. **Open requests**
7. **Dead ends**

Rules for the summarizer prompt (`prompts/session_compact/summarizer.md`):
- Copy facts verbatim. Facts come **only from the user or tool results**, never from assistant guesses.
- Carry earlier facts over unless corrected; keep only the corrected value.
- Drop small talk and asides in every section.
- Never record the checkpoint instruction itself.

For an IDE, "Tool Results" should preserve file paths, symbol names, error messages, test results and command outputs that matter. Keep that verbatim.

Storage: the checkpoint is saved in the turn metadata (`self_compact.summary`, `compacted_by="session"`) and is what the engine receives as "earlier conversation".

### 6.3 In-turn guard (SelfCompact, minimal)
Keep only what an RLM turn needs to stay safe during many tool rounds:
- Pre-send window guard and **provider-overflow recovery** (if the provider rejects for length, compact and retry once).
- **Ownership**: when the session compactor is active, in-turn compaction only guards *within* the turn, does not re-summarize history the host already sized, and writes the **same checkpoint format** (section 6.2) using the same summarizer prompt.
- The rubric-judged "COMPRESS verdict" step from the original is optional. Skip it unless the simple guard proves insufficient.

---

## 7. Memory

### 7.1 Layers

| Layer | Lifetime | Storage | Purpose |
|---|---|---|---|
| Stored conversation | the chat | SQLite | source of truth |
| Session checkpoint | the chat | turn metadata | folded older turns |
| Pinned facts | the chat | SQLite | deterministic goal, constraints, decisions, re-injected every turn |
| **Short-term memory** | the chat | sqlite-vec, `memory_type=short_term` | semantic recall of anything said in this chat |
| **Long-term memory** | across all chats | sqlite-vec, `memory_type=long_term` | durable facts and preferences about the user |
| RLM transcript | the chat | disk | code + REPL state (section 5.6) |

### 7.2 Short-term memory
- Every turn is **chunked** (`memory.chunkChars`, `memory.chunkOverlap`, `memory.maxChunksPerTurn`) and embedded, so a fact inside a long pasted file or log stays retrievable.
- Recall filter: `user_id` + `session_id` + `memory_type=short_term`.

### 7.3 Long-term memory
After each turn, a **background task** (`learnFacts`) distils durable facts:
- Input: the turn plus related known facts (`memory.factKnownLimit`).
- Output: **add / update / delete** operations.
- Facts must be **grounded in what the user said** (`memory.factRequireUserGrounding`).
- Near-duplicates (`memory.factDedupScore`) **update** the existing fact instead of adding one.
- Keep prior versions (`memory.historyKeep`).
- A **per-user lock** serializes distillation so concurrent chats do not race.
- Recall filter: `user_id` + `memory_type=long_term`.

> IDE adaptation (my suggestion, not in the source): add an optional `workspace_id` payload field so coding conventions learned in one repo (e.g. "uses pnpm, strict TS") can be scoped to that project, with a Settings toggle for user-wide vs project-scoped long-term memory. Default: user-wide, exactly as the original.

### 7.4 Recall into the prompt
- Runs in preflight with cosine floor `memory.minScore` and limits `memory.sessionLimit`, `memory.userLimit`.
- Rendered into a **memory prompt block**, capped by `memory.promptMaxChars`.
- **Recalled memory never authorizes an action on its own** (see section 8).
- Memory, pinned facts, editor context and clock are **host blocks**. The RLM session swaps these sections **in place** each turn so they never pile up in the persistent transcript.

### 7.5 Vector store isolation
- One vector table per `{prefix}__{embed_model}__{dim}`, prefix `orchestrator_memory`. Changing the embedding model never mixes vector spaces (re-embed lazily or offer a "re-index memory" action).
- Every memory row carries `user_id`, `session_id`, `memory_type`, `kind`, `category` (and optional `workspace_id`), all indexed, and **every** search, update and delete is filtered by them. `user_id` is a local profile id.
- Memory operations touch only memory tables. Nothing else in the database is ever searched by them.
- Memory is a separate, clearly labelled prompt block, so the model does not confuse remembered user statements with file contents.

### 7.6 Dependencies and failure
- Embeddings: the configured OpenAI-compatible embedding endpoint (default model suggestion: `bge-m3` via Ollama; any model works as long as the dim is stored). Fact distillation uses the chat model.
- If embeddings are unavailable: recall and save are skipped, a small status indicator shows "memory offline", the turn still runs.
- Provide Settings → Memory: view, search, edit, delete individual long-term facts, "forget everything", export/import JSON. Users must be able to see and remove what is remembered.

---

## 8. Tools and side-effect safety

### 8.1 Built-in tools (descriptor per tool: id, JSON Schema, timeout, permissions, `side_effects`)
- Read-only: `fs.read_file` (with line ranges), `fs.list_dir`, `fs.glob`, `search.grep`, `git.status`, `git.diff`, `git.log`, `editor.get_context`, `diagnostics.get`, `session.read` (search the stored chat history by `scope=session`, so the engine can look up compacted details without re-reading every turn).
- Side-effecting (`"side_effects": true`): `fs.write_file`, `fs.edit_file` (search/replace or patch), `fs.delete`, `fs.move`, `shell.run`, `git.commit`, `git.checkout`, `net.fetch` (if you add it).
- Session file helper: fill a missing `file_id`/line range for "continue reading"-style requests before validation.

### 8.2 Side-effect handling (`tools/side_effects`)
1. **Approval gate (default):** every side-effecting call shows an approval card in the chat (tool, arguments, a diff preview for edits). Modes: ask every time / allow for this session / allow for this workspace (per-tool allowlist, stored in settings). Shell commands match a user-editable allow/deny pattern list.
2. **Action check (optional, off by default):** a short model call using `prompts/action_check.md` asks whether the user requested *this* action, *now*, with *these* values. A model must not act on guesses, recalled memory or a request the user put on hold.
3. **Per-turn de-duplication:** inside one turn (`begin_tool_turn`), identical side-effecting calls run once and share the result.
4. File edits go through a **review layer**: changes are written to a pending-changes store and shown as diffs in the editor (accept / reject per hunk) before they touch disk, unless the user enabled auto-apply. Provide checkpoints/undo for each turn's edits.

### 8.3 Executor
All execution goes through one `ToolExecutor.execute(toolId, args, ctx)` that validates args against the schema, enforces `tool_ids` and workspace path confinement (resolve symlinks, reject `..` escapes), applies timeouts, and records every call to the turn log.

---

## 9. Turn lifecycle (host)

For each user message:

1. **Model limits**: stamp `context_length`, `max_output_tokens` on the config.
2. **Session workspace**: load stored turns, attached files/selections, run the **session compactor** (6.2).
3. **Preflight blocks**: memory recall (short + long term), pinned facts, **editor context block**, clock, persona/system rules.
4. **Tool selection**: core coding tools always available; the user can toggle tool groups per chat. (No page scoping.)
5. **RLM engine runs** and streams events.
6. **Persist**: store the turn; save short-term memory chunks; start long-term fact distillation in the background.

`TurnRunner`: each turn is a background task with a bounded event buffer and sequence numbers. The renderer subscribes with a `lastSeq` cursor so it can resume the stream after a reload. Explicit **Stop** interrupts the kernel, aborts the model request, and marks the turn cancelled (partial output is kept).

---

## 10. UI and design system

### 10.1 Source of the look
The reference images are two monochrome mobile-app mockups (a motivational quotes app and a music app). Their shared style is: strict **black / white / light-gray**, **hand-drawn flat line-art illustrations of people**, **rounded cards and pill buttons**, **bold italic serif headings** paired with a clean sans-serif, **soft shadows**, and **decorative thin circle outlines** in the background. Translate that into a serious, dense, desktop IDE without making it a toy. Calm, editorial, high-contrast.

### 10.2 Tokens (CSS variables; implement both themes)
- **Dark (default)**: `--bg #0B0B0C`, `--surface #141415`, `--surface-2 #1C1C1E`, `--border #2A2A2D`, `--text #F4F4F2`, `--text-dim #9A9A9F`, `--accent #FFFFFF` (inverted buttons: white on black).
- **Light**: `--bg #F2F2F0`, `--surface #FFFFFF`, `--surface-2 #E9E9E7`, `--border #D9D9D6`, `--text #111111`, `--text-dim #6B6B70`, `--accent #111111`.
- Status colors are the **only** non-mono exception and must be muted: success, warning, error, plus standard diff red/green (needed for usability). Syntax highlighting uses a custom mono-leaning Monaco theme per mode, with minimal hue.
- Radius: `--r-sm 8px`, `--r-md 14px`, `--r-lg 20px`. Shadows soft and large-blur in light mode, border-driven in dark mode.
- Fonts: **Heading** an italic serif (e.g. Playfair Display Italic or Instrument Serif Italic); **UI** Inter; **Code** JetBrains Mono. Bundle the fonts locally (no CDN at runtime).

### 10.3 Layout
- Left **activity bar** (icons): Explorer, Search, Source control, Chats, Memory, Settings.
- **Explorer / sidebar** panel with file tree.
- Center: **tabbed Monaco editor** with split view and a diff view for pending AI edits.
- Right: **Agent panel** (chat). Resizable. Can dock bottom or float.
- Bottom: **terminal** (xterm), problems, output.
- **Status bar**: workspace, branch, model, **context meter**, memory status, kernel status.
- **Command palette** (Ctrl+Shift+P) and quick-open (Ctrl+P). Full keyboard navigation.

### 10.4 Screen patterns borrowed from the references
- **Welcome / splash** (like "Get Motivated!"): black full-bleed screen, large italic serif title, small caps subtitle, a large line-art illustration, thin circle outlines behind it. Buttons: "Open folder", "Clone repo", recent projects as rounded cards.
- **Category tiles** (like the Categories screen): filled black tiles = selected, outlined white tiles = unselected. Use this pattern for chat modes/tool groups and settings sections.
- **Profile-style card**: the Memory screen shows a user card with counts (facts stored, chats) and two stacked buttons: outlined **"View memories"** and filled **"Add memory"**.
- **Quote card** (Home screen): used for pinned facts, shown as an italic-serif card at the top of the chat ("Goal: …").
- **Thumbs up / down** round buttons on each assistant message (feedback stored locally, used for nothing else unless asked).
- **Player-style progress bar** (from the music app): use it for the **context meter** (tokens used vs `history` budget, with a marker at the compaction trigger) and for long-running turn progress.
- **Empty states**: line-art illustrations (create inline SVGs yourself, original, in a single stroke weight, no stock or copyrighted art).
- Decorative circle outlines appear as subtle background elements on welcome, empty states and settings headers only. Never behind code.

### 10.5 Agent panel details
- Message types rendered distinctly: user, assistant text, **code cell** (collapsible, shows output), **tool call card** (name, args, result, duration), **approval card**, **sub-agent tree**, **compaction notice** ("Earlier conversation folded into a checkpoint", expandable to read the checkpoint), **memory recalled** chip list (click to see what was injected).
- Composer: multiline, `@file` and `@symbol` mentions, attach selection, model picker, Stop button, send on Enter.
- Everything streams. Nothing blocks the UI thread.

### 10.6 Quality bar
Accessible contrast (WCAG AA), reduced-motion support, window resizing down to 1100×700, HiDPI sharp, custom titlebar that still behaves natively on both OSes.

---

## 11. Security and privacy (Electron)

- `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` for the renderer; strict **CSP**; no remote content; `webSecurity` on.
- Preload exposes a **typed, allow-listed** API only. Validate every IPC payload (zod). Never expose `fs` or `child_process` to the renderer.
- API keys stored with Electron **`safeStorage`**, never in plain `settings.json`, never logged.
- All file access confined to opened workspace roots plus the app data dir.
- Telemetry: **none** by default.
- The model cannot reach the network from the kernel unless a tool explicitly does it; on Linux with bwrap, disable network inside the jail by default.

---

## 12. Prompts (all `.md` files in `packages/orchestrator/prompts/`)

Write these as separate files, not strings in code:
- `rlm/system.md`: explains the kernel, persistence of variables, how to call tools, how to write a final answer, how to spawn sub-agents, how to be concise. Includes the **tool surface section** generated by `tool_surface`.
- `rlm/compact_resume.md`: framing for "earlier work" after compaction.
- `session_compact/summarizer.md`: the checkpoint prompt from section 6.2.
- `memory/fact_distill.md`: add/update/delete operations, user-grounding rule.
- `action_check.md`: the optional action check.
- `editor_context.md`: template for the editor context block.

Prompts must reference tools only through the generated tool surface, not hard-coded names, so adding a tool never requires editing a prompt (the original lists hard-coded tool names in prompts as a known weakness; avoid repeating it).

---

## 13. Packaging and installers

- `electron-builder.yml` targets: **Windows NSIS** (`.exe`, per-user and per-machine options, desktop + start menu shortcuts, uninstaller), **Linux AppImage + `.deb`** (`.rpm` optional).
- **Bundled Python**: a build script (`scripts/fetch-python`) downloads python-build-standalone for the target platform/arch, installs `ipykernel` and the worker dependencies, and places it in `extraResources`. Verify the app runs on a machine with **no Python installed**.
- Native modules (`better-sqlite3`, `node-pty`, `sqlite-vec`) rebuilt for Electron's ABI per platform (`electron-rebuild` / prebuilds).
- **CI** (`.github/workflows/release.yml`): matrix `windows-latest` and `ubuntu-latest`, build, run e2e smoke test, upload artifacts to a GitHub Release.
- **Auto-update** via `electron-updater`, with a Settings toggle and channel (stable/beta).
- Code signing: wire env-based hooks for a Windows code-signing certificate; document that unsigned Windows builds trigger SmartScreen warnings. Do not invent certificates.
- Note: building the Windows installer is most reliable on a Windows runner. Do not promise cross-compiling from Linux unless you have tested it.

---

## 14. Phases and acceptance criteria

Complete each phase, run its checks, and summarize what passed before moving on.

**Phase 1: Shell.** Electron + React + Monaco + xterm + file tree, both themes, design tokens, welcome screen, command palette. *Accept:* open a folder, edit and save files, run a terminal, switch themes, on Windows and Linux.

**Phase 2: Orchestrator host + LLM.** utilityProcess, SQLite, ModelCatalog, OpenAI-compatible streaming client, TurnRunner with buffered events and resubscribe, plain chat (no RLM yet). *Accept:* reload the renderer mid-stream and the turn continues; Stop cancels.

**Phase 3: RLM kernel.** Bundled Python worker, JSONL RPC, execute/interrupt/reset, sandbox (Linux) and Job Object (Windows), cell streaming to UI. *Accept:* variables persist across cells and across two turns; a runaway loop is interrupted by Stop.

**Phase 4: Tools and safety.** Registry, `tool_surface`, kernel callables, `host_request` enforcement of `tool_ids`, approval cards, diff review, de-duplication. *Accept:* the model can read, grep, edit and run tests; edits appear as reviewable diffs; a tool not in `tool_ids` is refused host-side.

**Phase 5: Compactor.** Budget math from the model catalog, session compactor, checkpoint format, RLM transcript compaction at 90%, in-turn overflow recovery, compaction notice in UI, context meter. *Accept (scripted tests, small context window model):* (a) a code/identifier given in turn 2 is still answered correctly after several compactions; (b) small talk is dropped; (c) a user correction replaces the old value; (d) a tool call is never separated from its result; (e) `kept_turns` never reaches 0; (f) an app restart resumes via snapshot or reseed.

**Phase 6: Memory.** sqlite-vec store, chunking, short-term recall, long-term distillation with add/update/delete, dedup, history, per-user lock, pinned facts, memory UI. *Accept:* a fact buried in a long pasted log is retrievable later; a long-term fact from chat A is recalled in chat B; user A-style isolation holds across `session_id`; with the embedding endpoint off, turns still work; deleting a fact in the UI removes it from recall.

**Phase 7: Sub-agents.** Spawn, depth limits, inbox, tree UI. *Accept:* a parent delegates two parallel searches and merges results; depth limit enforced.

**Phase 8: Packaging.** Installers, bundled Python, CI matrix, auto-update. *Accept:* fresh VM installs on Windows 10/11 and Ubuntu 22.04+, app launches with no Python/Node installed, RLM turn works end to end, uninstall is clean.

---

## 15. Working rules for you (the coding agent)

1. Read this whole file first. Then propose the exact repo scaffold and wait for no one: proceed, but list your assumptions at the start of each phase.
2. Stay inside scope (section 1). When something is ambiguous, pick the simplest option consistent with this document and note it in `DECISIONS.md`.
3. Write tests alongside code; the acceptance criteria above are your definition of done.
4. Keep modules small and typed. No `any` in the orchestrator package. No magic numbers: every threshold is a setting with a default and a comment explaining it.
5. Never fabricate results. If a check cannot run in your environment (for example a Windows build on a Linux machine), say so plainly and write the verification steps for the user instead of claiming success.
6. At the end of each phase, output: what was built, what was verified (with evidence), what is known to be incomplete.
7. Do not add telemetry, analytics, or network calls beyond those listed.

---

## 16. Open decisions to confirm with the product owner before Phase 3

1. **Default LLM provider/models** to prefill (the app works with any OpenAI-compatible endpoint, but a good default list improves first-run).
2. **Embedding model** default and whether to offer a bundled local embedder for fully offline use.
3. **Windows sandbox strategy** beyond Job Objects (for example running the kernel inside WSL2 as an optional mode).
4. **Long-term memory scope** default: user-wide (as in the source design) or per-project.
5. **Auto-apply edits** default: off (review first) is recommended.
