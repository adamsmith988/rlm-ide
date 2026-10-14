# Decisions

Short records of choices that shape the code. Newest last.

## Hexagonal core, Electron as composition root
The agent logic (turn loop, compaction, memory, tool permissions) has to be testable without
Electron and reusable from other hosts. `packages/core` holds only domain code, use cases and ports.
The orchestrator utilityProcess and the Electron main process wire adapters into it.

## Orchestrator in a utilityProcess
Turns run for minutes and do CPU work (embeddings, SQLite, kernel I/O). Keeping them out of main
keeps the window responsive. A crashed orchestrator is restarted without losing the UI.
Interrupted turns are marked `cancelled` at startup (`recoverInterrupted`).

## Two model providers behind one router
`local_onnx` keeps everything on the machine. `openai_compat` covers OpenAI, OpenRouter, Ollama,
LM Studio, vLLM and custom servers with one adapter. Roles (agent, summarizer, embed) pick a model each,
so you can pair a remote agent with local embeddings.

## FrogNano on WebGPU in a hidden window
FrogNano is published only as q4f16 ONNX. ORT web with WebGPU runs it at a usable speed. CPU
(`onnxruntime-node`) runs it too, but too slowly for agent work, so CPU is only a fallback.
- NVIDIA on Linux: Dawn hides `shader-f16` on Vulkan unless the switch `enable-dawn-features=vulkan_enable_f16_on_nvidia` is set.
- Adapter choice: the probe tries both power preferences and prefers an adapter with f16.
- GPU out-of-memory (for example when another local server holds the VRAM) is reported as a clear error and resets the runtime.

## bge-m3 in-process, q8
Embedding is on the hot path of every turn (memory recall, chunking). Running it in the orchestrator
avoids an IPC hop. `q8` maps to `model_quantized.onnx`.

## Event-sourced turns
Each step of a turn is a `TurnEvent` with a sequence number. The renderer rebuilds messages with a
pure reducer, so reload, reattach and multiple views all come down to replaying events.

## Tools as importable modules in the kernel
The model writes ordinary Python (`fs.read_file(...)`, and `import fs` works too). Each tool group is a real module in
`sys.modules` whose functions call the host. The host enforces tool ids and approvals. File writes
must use `fs.write_file`/`fs.edit_file`, never shell redirection, because only fs edits are
journaled for review and revert.

## Settings carry every threshold
Budgets, ratios, limits and timeouts are zod settings with defaults and a comment explaining each.
Logic never hard-codes them. Evals override them through `RLM_SETTINGS_JSON`.

## Dev on Ubuntu runs with `--no-sandbox`
Ubuntu's AppArmor blocks unprivileged user namespaces, and a dev checkout's `chrome-sandbox` is
not SUID. `pnpm dev:nosandbox` is for development only. Packaged AppImage and deb builds keep the sandbox.

## Only immediate repeats of side-effecting calls are de-duplicated
De-duplication exists so a model that repeats itself does not write twice or ask for approval twice.
It used to cache results for the whole turn, so re-running the tests after a fix returned the old
failure. Now a call is reused only when the previous side-effecting call was identical.

## The IDE checks the agent's work, not just the model
Evals showed a 4B agent claiming success on files that did not parse or pages that threw on load.
Three host-side checks close that gap without relying on the model to remember them:
- After `fs.write_file`/`fs.edit_file`, JS (`node --check`) and JSON are syntax-checked. Problems are
  sent as tool notices, which are appended to the cell output even when the model ignores the return value.
- `web.check` opens a workspace page in a hidden window with the sandbox on, network off and an in-memory
  session. It reports console errors, missing files, unreadable text, and the effect of clicks.
- After a turn that edited files, the engine asks the model once to check its work against the request
  and the remembered preferences (`rlm.selfReview`).

## Cut-off cells are retried, not treated as answers
A reply that hits the output limit inside a ```` ```repl ```` fence used to end the turn silently with
nothing written. The engine now detects the unclosed fence, keeps only its head in the transcript, and
asks the model to continue in smaller cells.

## Dogfood evals drive the real app
`evals/` launches the built app with an isolated profile and workspace, types prompts into the
composer over CDP, auto-handles approvals, then scores the result with deterministic checks
(files, tests, a headless render with interaction probes) plus an LLM rubric judge.

## Chat previews run on their own protocol
`srcdoc`, `blob:` and `data:` frames inherit the renderer's CSP, which forbids inline scripts. Fenced
`html`, `svg` and `canvas` blocks (and plain JavaScript that only draws on a 2D canvas) are stored in
main and served on `rlm-preview://doc/<id>` with their own CSP: scripts allowed, every network request
blocked. The frame is `sandbox="allow-scripts"` without same-origin, so it cannot reach the IDE.
Mermaid renders in the renderer with `securityLevel: "strict"` and a monochrome theme.

## The built-in browser is a native view the agent can drive
The browser is a `WebContentsView` on its own persistent partition, laid over a box the renderer
reserves. DevTools dock in a second view. Element picking, page snapshots and click targets run in an
isolated world, so page scripts cannot see or tamper with them. Agent clicks and typing are real input
events. Screenshots and picked elements become attachments: images on `rlm-attach://` plus markdown in
the message text, so the model reads the selector and HTML and the transcript shows the image.
`browser.click`, `browser.type` and `browser.press` are marked `repeatable`: three identical clicks are
three clicks, not one de-duplicated call.

## Team hubs are hosted by an IDE, not a separate server
`@rlm/team` is its own hexagonal package (domain, service, ports; SQLite, git CLI, HTTP, SSH and UDP
adapters). Any IDE can host a hub from the Team screen; everyone else, including the host's own IDE,
is a client of its HTTP API. One process serves:
- the JSON API, live events (server-sent events) and git smart HTTP through `git http-backend`;
- an SSH server (`ssh2`) for git over SSH with the member's key, and port forwarding that can only
  reach the hub's own HTTP port. Newcomers log in as `join` with an invite code; key-less members as
  `token`;
- a UDP beacon so IDEs on the LAN list the hub.
Remote teammates reach it through the SSH port or through any OpenSSH server that can see the hub (a
jump host); the IDE opens the tunnel and rewrites the remote with `url.<tunnel>.insteadOf`. SSH host
keys are pinned on first connect. Tokens and jump passwords live in safeStorage; `SecretStore.all()`
only hands provider keys to the orchestrator.

Protected branches are a `pre-receive` hook driven by an environment variable the transports set per
pusher, so the same check covers HTTP and SSH. Pull requests merge on the hub without a working tree
(`git merge-tree --write-tree`, `commit-tree`, compare-and-swap `update-ref`; git 2.38 or newer).

## Bug reports go to whoever introduced the code
A bug names a file and line (pre-filled from the editor). The hub blames that line on the reported
branch (falling back to the default branch), then routes the report to the commit author if they are a
member, else to whoever pushed that commit (pushes are recorded by diffing refs around every push),
else to the author of the pull request that merged it, else to the group's maintainers. The report is
one markdown file: the code around the line, the error, the commit, pull request and pusher, steps,
environment and a fix checklist. The assignee answers with a status (acknowledged, in progress, fixed
with a commit, won't fix) and a comment, which notifies the reporter. Clones and shared folders commit
with the hub identity (repo-local `user.email`, only if unset) so blame finds the right person.
