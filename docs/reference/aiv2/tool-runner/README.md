# Tool Runner — docs index

## Normative contracts

- **[LLM Routing Contract](../../modules/LLM_ROUTING_CONTRACT.md)** — how tools and plugins must call models (`/internal/llm/*` only; binding → `logical_id`; llm-proxy load-balances backends). Read this before adding any new LLM-backed tool or plugin.
- **[Output Persistence Contract](../../modules/OUTPUT_PERSISTENCE_CONTRACT.md)** — declare `output_intents`; tool-runner alone stamps identity and writes Storage. Never `POST /upload` from tools/plugins.
- **Session durability (opt-in)** — long-running plugins can keep incomplete runs across tool-runner restarts (see below).

## Session durability (workspace + mid-run checkpoint)

**Opt-in only** — short tools pay nothing. Declare on `plugin.json` (or tool meta):

| Flag | Meaning |
|------|---------|
| `supports_workspace: true` | Use durable disk under `TOOL_RUNNER_WORKSPACE_ROOT` |
| `supports_session_checkpoint: true` | Allowed to upsert mid-run `tool_sessions` rows |

### Workspace layout

```
{TOOL_RUNNER_WORKSPACE_ROOT}/{plugin_id}/{user_id}/{record_id}/{run_id}/
```

- Env: `TOOL_RUNNER_WORKSPACE_ROOT` (cluster: `/data/tool_workspaces` on PVC `aiv2-tool-workspaces`)
- Helper: `modules/tools/run_workspace.py` → `workspace_dir(...)`, `plugin_scratch_root(...)`
- Fallback without env: `{tempdir}/aiv2_tool_workspaces` (dev only; not durable)

### Mid-run checkpoint

- In-process: `modules/tools/session_checkpoint.checkpoint_tool_session(...)`
- HTTP (service-to-service): `POST /internal/sessions/checkpoint` with internal key
- Body: standard `plugin_session` envelope (`plugin_id`, `record_id`, `run_id`, `status`, `output_preview`, slim `payload`/`trace`)
- Throttle: default ~20s (`TOOL_SESSION_CHECKPOINT_MIN_INTERVAL_SEC`); always writes on terminal status or `force=true`
- End-of-execute path unchanged: `result["plugin_session"]` still upserts via registry wrapper

`agent_flow` is the reference consumer (`supports_workspace` + `supports_session_checkpoint`). Other long plugins: set the flags, write/read under `workspace_dir`, and call the checkpoint helper on milestones.

### Case-visible sessions (optional soft-link)

Unrelated to workspace durability. Any plugin/tool may opt in so its **output files** appear on a Case page without binding blobs to that case:

- Set `plugin_session.case_id` (from the plugin's own input) and `expose_to_case: true`
- Keep deliverable `output_intents` with `bind_case: false`
- Helper: `modules/tools/plugin_session.build_plugin_session`
- List API: `GET /cases/{case_id}/tool-sessions` (Case RBAC); downloads use artifact owner/group RBAC

**Not mandatory.** Omit the flag → never listed. tool-runner does not stamp visibility from ambient `X-Case-Id`. Normative details: [modules/OUTPUT_PERSISTENCE_CONTRACT.md](../../modules/OUTPUT_PERSISTENCE_CONTRACT.md#case-visible-sessions-optional-soft-link).

## Implementation

- Gateway routes: `src/routes/internal_llm.py`
- Session checkpoint route: `src/routes/internal.py` → `POST /internal/sessions/checkpoint`
- Client helpers (imported by tools/plugins): `modules/tools/tool_runner_llm_gateway.py`
- Durable persistence: `modules/tools/output_persistence.py`, wired from `src/utils/registry_wrapper.py`
- Workspace + checkpoint: `modules/tools/run_workspace.py`, `modules/tools/session_checkpoint.py`
