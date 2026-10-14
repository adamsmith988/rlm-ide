"""System prompt appendix and sole OpenAI tool schema for RLM mode."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.orchestrators._shared.tool_surface import (
    DEFAULT_NAMESPACE,
    SURFACE_HEADING,
    call_hint,
)
from src.orchestrators._shared.tool_surface import format_tool_surface as _format_shared_surface
from src.orchestrators._shared.tool_surface import replace_tool_surface as _replace_shared_surface
IPYTHON_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute Python or `%%bash` in a persistent IPython kernel. "
            "Variables survive across calls. "
            "Pool files are already under `pool/files/` and `pool/inputs/`. "
            "Delegate with `await rlm(...)` (handle only). Top-level await; no asyncio.run()."
        ),
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python or a `%%bash` cell. If you use `%%bash`, it must be the first line."
                    ),
                }
            },
            "additionalProperties": False,
        },
    },
}

RLM_SYSTEM_APPENDIX = """\
You are a general purpose agent that uses code to solve tasks.
You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and iterating one step at a time.
When you are done, stop calling tools and state your final answer.

You are running in RLM (Recursive Language Model) mode.

IPython is the agent's long-lived notebook: a persistent control environment for reasoning,
context management, state, tool orchestration, and recursive subcalls. Use it to keep
intermediate variables, inspect and transform outputs, write small helper functions, and
preserve useful state across turns or compaction.

You have exactly one OpenAI tool: `ipython`. Emit a real tool call when you need to run code.
Tool/ipython errors return a traceback as the tool result — recover on the next turn
(inspect keys with `.get()` / `list(d.keys())`, fix the call, then continue). Do not stop
with an empty answer after a recoverable error.

When running shell commands, use `%%bash` cells (`%%bash` must be the first line of the cell).
Each `%%bash` cell is a throw-away subshell — keep dependent shell steps in one cell, or use
`%cd` / `os.environ` for kernel-level state. Python state persists across cells.

Use Python for reading, searching, and editing files — assign results to named variables so you
can slice and reuse them later. Prefer keeping large text in variables rather than pasting it
into chat. Large files are never refused: chunk them.

Large-file recipe (do this instead of `Path.read_text()` / printing whole files):
1. `st = skills.file_stats(path)` — if `size_bytes` is large or `binary`, do not dump the body.
2. Prefer `skills.grep_file(path, pattern)` for targeted search.
3. Otherwise page with `skills.read_lines(path, start=1, count=200)` (or 500), assign to a
   variable, then advance `start` using the `next start=N` marker until EOF.
4. `skills.read_text` returns at most ~20k chars and marks `…[truncated…]` — that is not a refusal;
   continue with step 2–3. Keep chunks in variables; do not re-paste into chat.
Large tool dumps may be spilled under `offload/` — open by path; do not re-paste them into chat.
`%%bash` output is capped (~50KB / 2000 lines); when truncated the full dump is written under
`offload/` and the result includes `[Full output: offload/…]` — open that path.

Attached Data Pool files appear under `pool/files/` and `pool/inputs/` when provisioned
(see `pool/INDEX.md`).

Kernel objects (not OpenAI tools): `rlm`, `agent_message`, `agent_observe`, `pools`, `tools`,
`skills`, `refine`, `goal`, `rlm.harness`. Inspect with `dir(...)` / `help(...)`. Top-level
`await` only — never `asyncio.run()`.

Continue from prior turns and kernel state — do not restate the whole plan each turn.
Pick up from the latest tool results / child messages and take the next concrete step.
"""

IPYTHON_HOWTO = """\
IPython mechanics: Python names, imports, and helpers persist across cells. Tool calls are
ordinary `await` expressions you can bind to variables. Always assign read/search results to
named variables so you can revisit them later. If you use `%%bash`, it must be the first line
of that cell so the whole cell runs as one shell context.
"""

REFINE_GUIDANCE = (
    "Treat continual harness refinement as a small, evidence-backed update only after a "
    "repeated failure or a tactic that will help FUTURE runs of this node. Use "
    "`await refine.run()` for durable cross-run lessons only — never for one-shot facts, "
    "file specs, IDs, or task-only details from this request. It returns immediately and "
    "runs when the current turn ends, so continue working normally after calling it."
)

MALFORMED_TOOL_ARGS_NUDGE = (
    "[rlm] The previous `ipython` tool call had truncated or invalid JSON in "
    "`function.arguments` (provider rejected it with a JSON parse error such as "
    "Unterminated string). That call was not executed. "
    "Continue the task: emit a complete `ipython` tool call whose arguments are "
    "valid JSON, e.g. {\"code\": \"...\"}, or write the final answer if done. "
    "Do not repeat a truncated tool call."
)

TRUNCATED_TOOL_ARGS_RESULT = (
    "[rlm] ipython tool call arguments were truncated or invalid JSON "
    "(Unterminated string / incomplete stream). Code was not executed. "
    "Re-send a complete ipython tool call with valid JSON arguments."
)

EMPTY_IPYTHON_REPEAT_NUDGE = (
    "[rlm] You ran the exact same `ipython` cell twice and both times the tool "
    "result was `(empty)` (no stdout, stderr, return value, or error). "
    "That usually means the cell was silent (assignment / pass / side-effect only) "
    "or did not do what you intended. Do NOT repeat the same code. "
    "Change approach: `print(...)` / `repr(...)` key variables, inspect paths with "
    "`list(...)` / `Path(...).exists()`, fix the call, or take a different next step."
)

# How many consecutive identical empty ipython cells before we nudge.
EMPTY_IPYTHON_REPEAT_THRESHOLD = 2

# Post-compact: identical ipython/look repeats before we force a path change.
POST_COMPACT_LOOP_THRESHOLD = 2
POST_COMPACT_GUARD_DISTINCT_OK = 3
POST_COMPACT_GUARD_MAX_TURNS = 8

POST_COMPACT_LOOP_NUDGE = (
    "[rlm] After context compaction you repeated the same `ipython` cell "
    "(or the same look path). Stop re-enumerating / re-looking. Jump to "
    "Next Steps item 1 from the compaction resume notice, skip Dead Ends, "
    "and take a different concrete action."
)

# How many times we tell the model about truncated tool-call JSON before failing.
MAX_MALFORMED_TOOL_ARGS_NUDGES = 2
MAX_CHILDREN_SOFT_WAIT_ROUNDS = 24
# Soft extensions when children still look healthy (no hard children_wait_timeout).
MAX_CHILDREN_WAIT_EXTENSIONS = 8
CHILD_HEALTHY_IDLE_SEC = 120.0
CHILD_WAIT_EXTENSION_SEC = 600.0
CHILD_TASK_PREFIX = "[task from parent]"

FANIN_CONTINUATION_PROMPT = (
    "Child agents have settled. Continue from their `agent_message` notices and/or "
    "files they wrote in the shared workspace — do not restate the whole plan. "
    "Merge only what is new, then produce the synthesized final answer for the user. "
    "If a child failed or completed without a reply, use the runtime notice details "
    "and their workspace outputs before concluding."
)

COMPLETION_DOCTRINE_CHAT = (
    "## Completion (chat mode)\n"
    "When the user task is done, stop calling tools and give a clear user-facing answer. "
    "The host may run a short completion review before finishing the node — answer "
    "honestly if asked; do not invent work you did not do."
)

COMPLETION_DOCTRINE_ARTIFACT = (
    "## Completion (artifact mode)\n"
    "Saying the task is finished in chat is not enough. Host completion requires a "
    "real deliverable file in the workspace (or a configured quality gate that passes). "
    "Write the output file with ipython, then briefly state the path.\n"
    "Before the host seals the run you MUST emit a final worker claim JSON "
    "(done/artifacts/checklist_status/why_complete). Checklist ids come from the "
    "host acceptance contract when present. Do not mark items satisfied that are "
    "still stubs. The host runs probes and may ask a completion review; "
    "do not claim done without the file and an honest claim."
)


def build_completion_doctrine(*, artifact_required: bool) -> str:
    return (
        COMPLETION_DOCTRINE_ARTIFACT
        if artifact_required
        else COMPLETION_DOCTRINE_CHAT
    )


def build_fanin_continuation_message() -> Dict[str, Any]:
    """Host-injected user turn after children settle — forces parent synthesis."""
    return {
        "role": "user",
        "content": FANIN_CONTINUATION_PROMPT,
        "_rlm_continuation": "fanin",
        "_rlm_continuation_reason": "children_fanin",
    }


def build_rlm_session_meta(
    *,
    cwd: str = "",
    depth: int = 0,
    messages_path: str = "",
    max_depth: int = 0,
) -> str:
    """Session facts the model can see (cwd, transcript path, recursion depth)."""
    lines: List[str] = []
    root = str(cwd or "").strip()
    if root:
        lines.append(f"Working directory: {root}")
        if int(depth or 0) > 0:
            lines.append(
                "This cwd is shared with the parent — relative paths the parent "
                "prepared (e.g. work/batches/*.json) are already here."
            )
    log = str(messages_path or "").strip()
    if log:
        lines.append(f"Conversation log: {log}")
    lines.append(f"Recursive agent depth: {int(depth or 0)}")
    _ = max_depth  # reserved for callers; depth alone matches Prime's meta line
    return "\n".join(lines)


def build_recursion_capabilities() -> str:
    """How rlm / messaging / observe work — capabilities, not bans."""
    return "\n".join(
        [
            "A callable `rlm` is already in your global namespace. "
            "`await rlm('sub-task')` spawns a child and returns immediately after task "
            "admission with a handle (`name`, `session_dir`, `model`, …); it never waits "
            "for or returns the child's answer.",
            "Choose a stable child name with `await rlm('sub-task', name='api-reviewer')`. "
            "If omitted, the host generates a unique name.",
            "A child inherits your model. For a different model, use "
            "`await rlm.find_models(...)` and an exact returned selector.",
            "Children reply with `await agent_message.send(message, receiver_role='parent')` "
            "when an answer is needed. Replies arrive as ordinary agent messages; not every "
            "task requires a reply.",
            "Use `await agent_message.list_agents()` to discover family and "
            "`await rlm.list_subagents()` to recover direct child handles. Follow up with "
            "`agent_message.send(..., receiver_role='child', receiver_name=...)`.",
            "Use `agent_observe.list` / `agent_observe.get` / `agent_observe.recent` to "
            "inspect a child's rollout. Messaging and observation reach your parent, "
            "siblings, and direct children.",
            "Delete a direct child with `await rlm.delete_subagent(child)` when it is no "
            "longer needed.",
        ]
    )


def build_subagent_guidance() -> str:
    """Delegation protocol aligned with Prime ``buildSubagentGuidance``."""
    return "\n".join(
        [
            "# Delegating to sub-agents",
            "",
            "Spawn independent, self-contained work with "
            "`handle = await rlm('task', name='worker')`. This returns at admission, "
            "not completion; keep the handle to stop or inspect the child later.",
            "End your turn after admission; do not busy-wait inside the cell. "
            "The host waits and delivers child notices when they settle.",
            "Delegate parallel context-heavy research or independent implementation; "
            "do a single known lookup, edit, or command inline.",
            "Ask for an explicit reply when needed. A child replies with "
            "`await agent_message.send(message, receiver_role='parent')`; parent "
            "follow-ups use `receiver_role='child'` plus the child's name or id. "
            "Not every message needs a reply.",
            "Have children write files and read those files for fan-in.",
            "When spawning, pass the same relative paths you just wrote "
            "(e.g. `work/batches/4.json`) — children share your workspace cwd and "
            "`pool/files/`. Prefer workspace-relative paths over absolute host paths.",
            "Use `await rlm.list_subagents()` after kernel restart or compaction.",
            "Use `agent_observe` for bounded transcript inspection.",
            "Persist genuinely reusable delegation patterns with `await refine.run()`.",
        ]
    )


def format_child_task_prompt(prompt: str) -> str:
    """Label a parent→child task the way Prime marks delegated work."""
    text = str(prompt or "").strip()
    if not text:
        return text
    if text.startswith(CHILD_TASK_PREFIX):
        return text
    return f"{CHILD_TASK_PREFIX}\n{text}"


def build_child_agent_doctrine(
    *,
    depth: int = 0,
    parent_name: str = "",
    child_name: str = "",
) -> Optional[str]:
    """Prime-style child doctrine. Only when ``depth > 0`` (root gets None)."""
    if int(depth or 0) <= 0:
        return None
    parent = str(parent_name or "").strip() or "your parent agent"
    lines = [
        f"You are a child agent spawned by {parent}.",
        f"Recursive agent depth: {int(depth)}.",
        "Task prompts from the parent are labeled `[task from parent]`.",
        "You share the parent's workspace cwd: relative paths the parent wrote "
        "(e.g. `work/batches/4.json`, `report.md`) open with `Path`/`open` as-is. "
        "Attached Data Pool files are under `pool/files/` and `pool/inputs/` in that "
        "same cwd — open them with `Path`/`open`/`ls pool`. Parent path names like "
        "`conversations/thread_2.txt` usually mean the basename under `pool/files/` "
        "(e.g. `pool/files/thread_2.txt`); use `Path('pool').rglob('*')` or "
        "`await rlm.look('thread_2.txt')` if unsure. Do not ask the parent for the "
        "path until you have searched the workspace and `pool/`.",
        "Write your own outputs into this same shared workspace (e.g. "
        "`work/results/<your_name>.json` or a path the parent named) so the parent "
        "can read them for fan-in.",
        "When a task calls for an answer, you are not done until you reply explicitly with "
        '`await agent_message.send(message, receiver_role="parent")` — printing alone does '
        "not complete the task.",
        "Your last printed text / last_content is NOT forwarded to the parent automatically.",
        "Not every message needs a reply; continue cleanup after sending and go idle normally.",
        "Agent messaging is restricted to your parent, siblings, and direct children.",
    ]
    cname = str(child_name or "").strip()
    if cname:
        lines.insert(1, f"Your subagent name is `{cname}`.")
    return "\n".join(lines)


def _pool_label(entry: Mapping[str, Any]) -> str:
    return str(entry.get("name") or "").strip()


def _pool_blurb(entry: Mapping[str, Any]) -> str:
    cfg = entry.get("cfg") if isinstance(entry.get("cfg"), Mapping) else {}
    desc = str(cfg.get("tool_description") or "").strip()
    if desc:
        return desc
    scope = str(cfg.get("pool_scope") or "inventory").strip() or "inventory"
    return f"gathered files/texts/cases (scope={scope})"


def format_attached_surface(
    pool_defs: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_ids: Optional[Sequence[str]] = None,
    *,
    cwd: str = "",
    input_modalities: Optional[Sequence[str]] = None,
    tool_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_surface_options: Optional[Mapping[str, Any]] = None,
) -> str:
    """Prime-style inventory: cwd + names that exist (minimal API noise)."""
    from src.orchestrators.rlm.runtime.native_media import (
        look_turn_limit_hint,
        native_look_kinds,
        normalize_input_modalities,
    )

    sections: List[str] = []
    native = native_look_kinds(normalize_input_modalities(input_modalities))
    if cwd:
        look_hint = ""
        if native:
            look_hint = (
                " Native look: "
                + ", ".join(sorted(native))
                + ' via `await rlm.look("filename")` '
                + f"({look_turn_limit_hint()}; look at a few, note findings, then look at more)."
            )
        # Working directory lives in build_rlm_session_meta; here only the provisioned surface.
        sections.append(
            "Pool workspace: `pool/files/` + `pool/inputs/` (see `pool/INDEX.md`)."
            + look_hint
        )
    pools = [p for p in (pool_defs or []) if isinstance(p, Mapping) and _pool_label(p)]
    if pools:
        lines = [
            "## Attached data pools",
            "On disk under `pool/`. Bound: "
            + ", ".join(f"`pools.{_pool_label(p)}`" for p in pools)
            + ".",
        ]
        for entry in pools:
            name = _pool_label(entry)
            roster = entry.get("files") if isinstance(entry.get("files"), list) else []
            names: List[str] = []
            for row in roster:
                if not isinstance(row, Mapping):
                    continue
                fn = str(row.get("filename") or "").strip()
                if fn:
                    names.append(fn)
                if len(names) >= 16:
                    break
            bit = f"- `pools.{name}` — {_pool_blurb(entry)}"
            if names:
                extra = len(roster) - len(names)
                bit += ": " + ", ".join(names)
                if extra > 0:
                    bit += f", … +{extra} more"
            lines.append(bit)
        sections.append("\n".join(lines))
    tools_section = format_tool_surface(tool_specs, tool_ids, **(tool_surface_options or {}))
    if tools_section:
        sections.append(tools_section)
    return "\n\n".join(sections)


TOOL_SURFACE_HEADING = SURFACE_HEADING
_LEGACY_TOOL_HEADING = "## Allowed tools"
CELL_TOOL_NAME = str(IPYTHON_TOOL_SCHEMA["function"]["name"])


def format_tool_surface(
    tool_specs: Optional[Sequence[Mapping[str, Any]]],
    tool_ids: Optional[Sequence[str]],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    description_chars: int = 0,
    max_params: int = 0,
    usage_hint: str = "",
) -> str:
    """Project tools section for the RLM kernel (shared renderer, RLM cell tool)."""
    return _format_shared_surface(
        tool_specs,
        tool_ids,
        namespace=namespace,
        cell_tool=CELL_TOOL_NAME,
        description_chars=description_chars,
        max_params=max_params,
        usage_hint=usage_hint,
    )


def replace_tool_surface(system_text: str, section: str) -> str:
    """Swap the tools section of an existing system prompt (append when absent)."""
    return _replace_shared_surface(
        system_text, section, headings=(TOOL_SURFACE_HEADING, _LEGACY_TOOL_HEADING)
    )


def reject_non_ipython_tool(
    name: str,
    *,
    pool_defs: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_ids: Optional[Sequence[str]] = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> str:
    """Tool-result hint when the model emits a native tool other than ipython."""
    label = str(name or "").strip() or "unknown"
    parts = [
        f"error: only the `{CELL_TOOL_NAME}` tool is allowed in RLM mode (got {label!r}). "
        f"Do the work inside an {CELL_TOOL_NAME} cell (Path/open/grep under `pool/`, or "
        f"`await {namespace}.<id>(...)` / `await rlm(...)`)."
    ]
    pools = [p for p in (pool_defs or []) if isinstance(p, Mapping) and _pool_label(p)]
    if pools:
        names = ", ".join(f"pools.{_pool_label(p)}" for p in pools)
        parts.append(f"Attached pools: {names}.")
    ids = [str(t).strip() for t in (tool_ids or []) if str(t).strip()]
    if ids:
        examples = ", ".join(call_hint(t, namespace) for t in ids[:8])
        parts.append(f"Allowed tools: {examples}.")
    return " ".join(parts)


SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a precise conversation summarizer. Output only the structured "
    "checkpoint summary — no preamble, no tool calls."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work seamlessly — as if the break never happened.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Keep the objective verbatim when short.]

## Constraints & Preferences
- [Constraints, preferences, or requirements mentioned by the user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes with exact paths when relevant]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale — preserve the *why*]

## Dead Ends
- [Approaches tried and abandoned + why — so they are not retried]

## Artifacts & Paths
- [Exact file paths, deliverables, batch/notes JSON, artifact names — character-for-character]
- [Or "(none)"]

## Kernel / REPL
- [Python names, imports, and helpers that should be reused — kernel state survives compaction]
- [Or "(none)"]

## Subagents & Parallel Work
- [Active/finished child agents, their roles, and results worth keeping]
- [Or "(none)"]

## Media / Looks Already Reviewed
- [Image/video paths already inspected — do NOT re-look the same paths]
- [Or "(none)"]

## Next Steps
1. [Ordered concrete next actions — when work remains, item 1 MUST be a concrete `ipython` action (open/grep/look/write), not narration]

## Critical Context
- [Error messages, identifiers, examples, or facts needed to continue — verbatim]
- [Or "(none)" if not applicable]

Rules:
- Preserve exact file paths, function names, IDs, and error messages.
- Prefer actionable state over narrative chat.
- Do not invent progress that did not happen.
- Dead Ends must list approaches that failed so they are not retried after compact."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary unless contradicted
- ADD new progress, decisions, artifacts, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, error messages, and already-reviewed media paths
- If something is no longer relevant, you may remove it
- Keep Dead Ends so abandoned approaches are not retried

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Dead Ends
- [Preserve prior dead ends; add new ones]

## Artifacts & Paths
- [Preserve and extend exact paths]

## Kernel / REPL
- [Preserve and extend reusable names]

## Subagents & Parallel Work
- [Update child-agent status/results]

## Media / Looks Already Reviewed
- [Accumulate paths already inspected — do NOT suggest re-looking them]

## Next Steps
1. [Update based on current state — first item a concrete `ipython` action when work remains]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

KERNEL_PERSIST_SUMMARY_NOTE = (
    "Note: the IPython kernel keeps running after this summary — every Python "
    "variable, import, and helper you defined stays available. The cells that "
    "defined them won't appear above, so record in the summary any names worth "
    "remembering so you reuse them instead of redefining them."
)

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [What was already done before the kept suffix?]

## Key Facts
- [Facts the suffix still needs]

Keep it short. Preserve exact names and paths."""


def build_summarization_prompt(
    *,
    custom_instructions: str = "",
    previous_summary: Optional[str] = None,
) -> str:
    base = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base += (
            "\n\n<user-instructions>\nThe user provided these instructions for "
            "this summary. Follow them with high priority while keeping the "
            "section format above: emphasize what they ask to focus on, and "
            "preserve verbatim anything they ask to remember.\n"
            f"{custom_instructions}\n</user-instructions>"
        )
    return f"{base}\n\n{KERNEL_PERSIST_SUMMARY_NOTE}"
