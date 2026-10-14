"""Subprocess IPython worker: JSONL protocol over stdin/stdout."""

from __future__ import annotations

# This module is executed as: python -m src.orchestrators.rlm.runtime.worker
# Protocol (one JSON object per line):
#   Host → Worker: {"op":"bootstrap","pools":[...],"tools":[...],"id":"..."}
#   Host → Worker: {"op":"exec","code":"...","id":"..."}
#   Host → Worker: {"op":"host_response","id":"...","ok":true,"result":{...}}
#   Host → Worker: {"op":"shutdown"}
#   Worker → Host: {"op":"host_request","id":"...","type":"...","payload":{...}}
#   Worker → Host: {"op":"exec_result","id":"...","ok":true,"stdout":"...","error":null}
#   Worker → Host: {"op":"ready"}

import asyncio
import json
import os
import sys
import traceback
from io import StringIO
from typing import Any, Dict, List, Optional

from src.orchestrators._shared.tool_surface import surface_namespace, tool_alias


# Soft-cap per-field payloads in JSONL frames. Host StreamReader limit is much
# higher (512MiB+); this keeps one print(huge_dump) from flooding the next LLM
# turn. Override with AIV2_RLM_CELL_OUTPUT_LIMIT (chars, not bytes).
# Match Prime Agent's default IPython output cap (64 KiB).
_DEFAULT_CELL_OUTPUT_LIMIT = 64 * 1024

# Prime bash tool: 50KB or 2000 lines (whichever first).
_BASH_OUTPUT_MAX_CHARS = 50 * 1024
_BASH_OUTPUT_MAX_LINES = 2000


def _truncate_bash_output(text: str, *, spill_root: str = "") -> str:
    """Cap %%bash stdout/stderr like Prime's bash tool (50KB / 2000 lines).

    When truncated, spill the *full* body under ``spill_root/offload/`` and append
    ``[Full output: offload/…]`` (same recoverability pattern as Prime's
    ``fullOutputPath``).
    """
    original = str(text or "")
    if not original:
        return original
    raw = original
    truncated = False
    lines = raw.splitlines(keepends=True)
    if len(lines) > _BASH_OUTPUT_MAX_LINES:
        omitted = len(lines) - _BASH_OUTPUT_MAX_LINES
        lines = lines[-_BASH_OUTPUT_MAX_LINES:]
        truncated = True
        raw = "".join(lines)
        raw = (
            f"...[{omitted} earlier lines omitted; full dump on disk]...\n"
            + raw
        )
    if len(raw) > _BASH_OUTPUT_MAX_CHARS:
        omitted = len(raw) - _BASH_OUTPUT_MAX_CHARS
        truncated = True
        raw = (
            raw[:_BASH_OUTPUT_MAX_CHARS]
            + f"\n...[truncated {omitted} chars from preview; full dump on disk]...\n"
        )
    if not truncated:
        return raw
    rel = ""
    root = str(spill_root or "").strip()
    if root:
        try:
            from src.orchestrators.rlm.runtime.tool_offload import (
                spill_content_to_offload,
            )

            rel = spill_content_to_offload(root, original, kind="bash") or ""
        except Exception:
            rel = ""
    if rel:
        return raw.rstrip() + f"\n\n[Full output: {rel}]"
    return (
        raw.rstrip()
        + "\n...[truncated; no spill dir — use head/tail/grep instead of dumping]...\n"
    )


def _cell_output_limit() -> int:
    raw = (os.environ.get("AIV2_RLM_CELL_OUTPUT_LIMIT") or "").strip()
    if not raw:
        return _DEFAULT_CELL_OUTPUT_LIMIT
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_CELL_OUTPUT_LIMIT
    return max(4 * 1024, min(val, 16 * 1024 * 1024))


def _clip_text(value: Any, *, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return (
        value[:limit]
        + f"\n...[truncated {omitted} chars; use seek/head/grep instead of printing whole files]...\n"
    )


def _prepare_emit(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy with oversized stdout/stderr/result clipped."""
    limit = _cell_output_limit()
    out = dict(obj)
    for key in ("stdout", "stderr", "error"):
        if key in out:
            out[key] = _clip_text(out.get(key), limit=limit)
    if "result" in out and isinstance(out.get("result"), str):
        out["result"] = _clip_text(out.get("result"), limit=limit)
    return out


def _emit(obj: Dict[str, Any]) -> None:
    # Always write protocol lines to the real process stdout — cell execution
    # may temporarily redirect sys.stdout into a StringIO capture buffer.
    stream = sys.__stdout__
    stream.write(json.dumps(_prepare_emit(obj), ensure_ascii=False) + "\n")
    stream.flush()


def _read_line() -> Optional[str]:
    line = sys.stdin.readline()
    if line == "":
        return None
    return line.strip()


class _HostProxy:
    """Blocking host_request from worker thread/async cell context."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"hr_{self._counter}"

    async def request(self, request_type: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        rid = self._next_id()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        _emit(
            {
                "op": "host_request",
                "id": rid,
                "type": str(request_type),
                "payload": payload if isinstance(payload, dict) else {},
            }
        )
        return await fut

    def resolve(self, rid: str, *, ok: bool, result: Any = None, error: str = "") -> None:
        fut = self._pending.pop(str(rid), None)
        if fut is None or fut.done():
            return
        if ok:
            fut.set_result(result)
        else:
            fut.set_exception(RuntimeError(error or "host_request_failed"))


def _build_namespace(host: _HostProxy, pools: List[Dict[str, Any]], tools: List[str]) -> Dict[str, Any]:
    import types

    async def host_request(request_type: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        return await host.request(request_type, payload)

    class RLMSpawnHandle(dict):
        @property
        def rlm_child_id(self) -> str:
            return str(self.get("rlm_child_id") or "")

        @property
        def name(self) -> str:
            return str(self.get("name") or "")

        @property
        def session_dir(self) -> str:
            return str(self.get("session_dir") or "")

        @property
        def model(self) -> str:
            return str(self.get("model") or "")

    class _RLM:
        async def __call__(self, prompt: str, *, name: str | None = None, model: str | None = None):
            return await self.run(prompt, name=name, model=model)

        async def run(self, prompt: str, *, name: str | None = None, model: str | None = None):
            raw = await host_request(
                "rlm.run",
                {"prompt": str(prompt or ""), "name": name, "model": model},
            )
            data = raw if isinstance(raw, dict) else {}
            return RLMSpawnHandle(data)

        async def list_subagents(self):
            raw = await host_request("rlm.list_subagents", {})
            if isinstance(raw, dict):
                return raw.get("subagents") or []
            return []

        async def delete_subagent(self, target):
            sel = target
            if isinstance(target, dict):
                sel = target.get("name") or target.get("rlm_child_id") or target.get("session_name")
            return await host_request("rlm.delete_subagent", {"target": str(sel or "")})

        async def find_models(self, query: str = "", limit: int = 8):
            raw = await host_request(
                "rlm.find_models", {"query": str(query or ""), "limit": int(limit)}
            )
            if isinstance(raw, dict):
                return raw.get("models") or []
            return []

        async def compact(self, keep_last: int = 12):
            return await host_request("compact.run", {"keep_last": int(keep_last or 12)})

        async def look(
            self,
            *paths: str,
            path: str | None = None,
            filename: str | None = None,
            pool: str | None = None,
        ):
            locators = [str(p).strip() for p in paths if str(p).strip()]
            if path:
                locators.append(str(path).strip())
            if filename:
                locators.append(str(filename).strip())
            payload: Dict[str, Any] = {"locators": locators}
            if pool:
                payload["pool"] = str(pool)
            return await host_request("rlm.look", payload)

        @property
        def harness(self):
            return _Harness()

        @property
        def goal(self):
            return _Goal()

        @property
        def heartbeat(self):
            return _Heartbeat()

    class _Harness:
        async def get(self):
            return await host_request("harness.get", {})

        async def list(self, kind: str | None = None):
            return await host_request("harness.list", {"kind": kind})

        async def upsert_memory(self, content: str, *, title: str = "", entry_id: str | None = None):
            return await host_request(
                "harness.upsert",
                {
                    "kind": "memory",
                    "content": content,
                    "title": title,
                    "entry_id": entry_id,
                },
            )

        async def upsert_note(self, content: str, *, title: str = "", entry_id: str | None = None):
            return await host_request(
                "harness.upsert",
                {"kind": "note", "content": content, "title": title, "entry_id": entry_id},
            )

        async def upsert(self, kind: str, content: str, **kwargs):
            payload = {"kind": kind, "content": content, **kwargs}
            return await host_request("harness.upsert", payload)

        async def delete(self, kind: str, entry_id: str):
            return await host_request(
                "harness.delete", {"kind": kind, "entry_id": entry_id}
            )

    class _Goal:
        async def get(self):
            return await host_request("goal.get", {})

        async def create(self, objective: str, *, token_budget: int = 0):
            return await host_request(
                "goal.create",
                {"objective": str(objective or ""), "token_budget": int(token_budget or 0)},
            )

        async def complete(self):
            return await host_request("goal.complete", {})

    class _Heartbeat:
        async def list(self):
            raw = await host_request("rlm_heartbeat.list", {})
            if isinstance(raw, dict):
                return raw.get("heartbeats") or []
            return []

        async def create(self, *, interval_sec: int = 300, prompt: str = "", name: str = ""):
            return await host_request(
                "rlm_heartbeat.create",
                {
                    "interval_sec": int(interval_sec or 300),
                    "prompt": str(prompt or ""),
                    "name": str(name or ""),
                },
            )

        async def delete(self, target: str):
            return await host_request("rlm_heartbeat.delete", {"target": str(target or "")})

    class _Observe:
        async def list(self):
            return await host_request("agent_observe.list", {})

        async def get(self, target: str = ""):
            return await host_request("agent_observe.get", {"target": str(target or "")})

        async def recent(self, target: str = "", limit: int = 20):
            return await host_request(
                "agent_observe.recent",
                {"target": str(target or ""), "limit": int(limit or 20)},
            )

    class _Refine:
        async def run(
            self,
            instructions: str = "",
            *,
            model: str | None = None,
            write_scope: str | None = None,
            scope: str | None = None,
        ):
            payload: Dict[str, Any] = {"instructions": str(instructions or "")}
            if model is not None:
                payload["model"] = model
            if write_scope is not None or scope is not None:
                payload["write_scope"] = write_scope if write_scope is not None else scope
            return await host_request("refine.run", payload)

    class _AgentMessage:
        async def send(
            self,
            text: str,
            *,
            receiver_role: str = "parent",
            receiver_name: str | None = None,
        ):
            return await host_request(
                "agent_message.send",
                {
                    "text": str(text or ""),
                    "receiver_role": str(receiver_role or "parent"),
                    "receiver_name": receiver_name,
                },
            )

        async def inbox(self):
            raw = await host_request("agent_message.inbox", {})
            if isinstance(raw, dict):
                return raw.get("messages") or []
            return []

        async def list_agents(self):
            raw = await host_request("agent_message.list_agents", {})
            if isinstance(raw, dict):
                return raw.get("agents") or []
            return []

    class _Pool:
        def __init__(self, pool_key: str, label: str):
            self._key = pool_key
            self.label = label
            self.__doc__ = (
                f"Attached data pool {label!r}. Gathers cases, files, and user texts. "
                f"Files are already under pool/files/ and pool/inputs/ — use Path/open/grep. "
                f"Optional: files = await pools.{label}.ls(); "
                f"hits = await pools.{label}.search('query')."
            )

        async def search(self, query: str):
            return await host_request(
                "pools.search", {"pool_key": self._key, "query": str(query or "")}
            )

        async def ls(self):
            raw = await host_request("pools.list", {"pool_key": self._key})
            if isinstance(raw, dict):
                return list(raw.get("files") or [])
            return raw if isinstance(raw, list) else []

        async def resolve_file(self, locator: str):
            return await host_request(
                "pools.resolve_file",
                {"locator": str(locator or ""), "pool_key": self._key},
            )

        def __repr__(self) -> str:
            return (
                f"<data pool {self.label!r}: files on disk under pool/; "
                f"await pools.{self.label}.ls() | .search(query)>"
            )

    class _Pools:
        def __init__(self, pool_defs: List[Dict[str, Any]]):
            self._by_name: Dict[str, _Pool] = {}
            self._by_key: Dict[str, _Pool] = {}
            for p in pool_defs or []:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name") or "").strip()
                key = str(p.get("pool_key") or name).strip()
                if not name:
                    continue
                # A pool must never shadow _Pools' own API (.names/.resolve_file)
                # or a private attribute, or that method disappears for the cell.
                if name.startswith("_") or name in self._by_name or hasattr(self, name):
                    stem = name.lstrip("_") or "pool"
                    n = 2
                    alt = f"{stem}_{n}"
                    while alt in self._by_name or hasattr(self, alt):
                        n += 1
                        alt = f"{stem}_{n}"
                    name = alt
                pool = _Pool(key, name)
                self._by_name[name] = pool
                if key:
                    self._by_key[key] = pool
                setattr(self, name, pool)
            names = ", ".join(self._by_name) or "(none)"
            self.__doc__ = (
                "Context-attached data pools. Files are already on disk under "
                f"`pool/files/` and `pool/inputs/`. Use Path/open/grep, or "
                f"`await pools.<name>.ls()`. Bound: {names}."
            )

        async def resolve_file(self, locator: str, *, pool: str = ""):
            pool_key = ""
            sel = str(pool or "").strip()
            if sel:
                if sel in self._by_name:
                    pool_key = self._by_name[sel]._key
                elif sel in self._by_key:
                    pool_key = sel
                else:
                    raise AttributeError(
                        f"unknown data pool: {sel}; bound: {', '.join(self._by_name) or '(none)'}"
                    )
            return await host_request(
                "pools.resolve_file",
                {"locator": str(locator or ""), "pool_key": pool_key},
            )

        def names(self):
            return list(self._by_name.keys())

        def __getattr__(self, item: str):
            if item.startswith("_"):
                raise AttributeError(item)
            raise AttributeError(
                f"unknown data pool: {item}; bound: {', '.join(self._by_name) or '(none)'}"
            )

        def __repr__(self) -> str:
            names = ", ".join(self._by_name) or "(none)"
            return f"<pools: {names}>"

    class _Tools:
        _rlm_gatherable = True

        def __init__(self, tool_ids: List[str]):
            self._allowed = {str(t).strip() for t in (tool_ids or []) if str(t).strip()}
            self._aliases: Dict[str, str] = {}
            for tid in self._allowed:
                self._aliases[tool_alias(tid)] = tid
                self._aliases[tid] = tid

        def _resolve(self, tool_id: str) -> str:
            tid = self._aliases.get(str(tool_id or "").strip(), str(tool_id or "").strip())
            if tid not in self._allowed:
                raise PermissionError(
                    f"tool_not_allowed:{tool_id}; allowed: {', '.join(sorted(self._allowed)) or '(none)'}"
                )
            return tid

        async def call(self, tool_id: str, **kwargs):
            return await host_request(
                "tools.call", {"tool_id": self._resolve(tool_id), "args": kwargs}
            )

        async def describe(self, tool_id: str):
            return await host_request("tools.describe", {"tool_id": self._resolve(tool_id)})

        def list(self):
            return sorted(self._allowed)

        def __dir__(self):
            return sorted({a for a in self._aliases if a.isidentifier()} | {"call", "describe", "list"})

        def __repr__(self) -> str:
            return f"<tools: {', '.join(sorted(self._allowed)) or '(none)'}>"

        def __getattr__(self, item: str):
            if item.startswith("_"):
                raise AttributeError(item)
            tid = self._aliases.get(str(item or "").strip())
            if not tid:
                raise AttributeError(
                    f"tool_not_allowed:{item}; allowed: {', '.join(sorted(self._allowed)) or '(none)'}"
                )

            async def _caller(**kwargs):
                return await self.call(tid, **kwargs)

            _caller.__name__ = item
            _caller._rlm_tool_call = True
            return _caller

    ns: Dict[str, Any] = {
        "host_request": host_request,
        "rlm": _RLM(),
        "agent_message": _AgentMessage(),
        "agent_observe": _Observe(),
        "refine": _Refine(),
        "pools": _Pools(pools),
        "tools": _Tools(tools),
        "RLMSpawnHandle": RLMSpawnHandle,
    }
    try:
        from src.orchestrators.rlm.runtime import skills as rlm_skills

        ns["skills"] = rlm_skills
    except Exception:
        pass
    # Also expose module-like alias
    ns["aiv2_rlm"] = types.SimpleNamespace(
        rlm=ns["rlm"],
        agent_message=ns["agent_message"],
        agent_observe=ns["agent_observe"],
        refine=ns["refine"],
        pools=ns["pools"],
        tools=ns["tools"],
    )
    return ns


_LAST_VALUE_NAME = "_rlm_last_value_"


def _register_importable(name: str, obj: Any) -> None:
    """Let ``import <name>`` / ``from <name> import <tool>`` resolve to the kernel object.

    Never shadows a real module already loaded under that name.
    """
    current = sys.modules.get(name)
    if current is None or getattr(current, "_rlm_gatherable", False):
        sys.modules[name] = obj


def _auto_gather_enabled() -> bool:
    raw = (os.environ.get("AIV2_RLM_AUTO_GATHER_TOOLS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _auto_await_enabled() -> bool:
    raw = (os.environ.get("AIV2_RLM_AUTO_AWAIT_TOOLS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _await_bare_tool_calls(tree: Any, ns: Dict[str, Any]) -> None:
    """Await top-level tool calls the model forgot to await (``r = tools.x(...)``,
    ``from tools import x; x(...)``), which would otherwise yield an unrun coroutine.

    Only a call that is a whole statement value or a direct ``print`` argument is
    awaited; calls handed to ``gather`` / ``create_task`` etc. are left alone.
    """
    import ast

    def marked(obj: Any, flag: str) -> bool:
        return getattr(obj, flag, False) is True

    tool_ns = {k for k, v in ns.items() if marked(v, "_rlm_gatherable")}
    tool_fns = {k for k, v in ns.items() if marked(v, "_rlm_tool_call")}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if marked(sys.modules.get(a.name), "_rlm_gatherable"):
                    tool_ns.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if marked(sys.modules.get(node.module), "_rlm_gatherable"):
                tool_fns.update(a.asname or a.name for a in node.names if a.name != "*")

    def is_tool_call(expr: Any) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        f = expr.func
        if isinstance(f, ast.Name):
            return f.id in tool_fns
        return (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in tool_ns and f.attr not in ("list",))

    def awaited(expr: Any) -> Any:
        return ast.copy_location(ast.Await(expr), expr) if is_tool_call(expr) else expr

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) and node.value is not None:
            node.value = awaited(node.value)
            v = node.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print"):
                v.args = [awaited(a) for a in v.args]


def _gather_independent_tool_awaits(tree: Any, ns: Dict[str, Any]) -> None:
    """Run independent top-level ``x = await <tool call>`` statements concurrently.

    A tool call is ``<tools>.<id>(...)`` on a namespace marked ``_rlm_gatherable`` or a
    function marked ``_rlm_tool_call`` / imported from such a namespace. A later tool
    await joins the group when its call reads no name bound by the group or by the
    simple statements between them (no awaits, no control flow, e.g. ``print(w)``);
    those statements then run right after the gathered calls, in their original order.
    Dependent chains stay sequential.
    """
    import ast

    tool_fns = {k for k, v in ns.items() if getattr(v, "_rlm_tool_call", False) is True}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if getattr(sys.modules.get(node.module), "_rlm_gatherable", False) is True:
                tool_fns.update(a.asname or a.name for a in node.names if a.name != "*")

    def tool_await(stmt: Any) -> Optional[str]:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Await)):
            return None
        call = stmt.value.value
        if not isinstance(call, ast.Call):
            return None
        f = call.func
        if isinstance(f, ast.Name) and f.id in tool_fns:
            return stmt.targets[0].id
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and getattr(ns.get(f.value.id), "_rlm_gatherable", False)):
            return stmt.targets[0].id
        return None

    def loads(node: Any) -> set:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    def stores(node: Any) -> set:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}

    def movable(stmt: Any) -> bool:
        return (isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign))
                and not any(isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr))
                            for n in ast.walk(stmt)))

    body: List[Any] = []
    run: List[Any] = []
    held: List[Any] = []

    def flush() -> None:
        if len(run) > 1:
            first = run[0]
            gather = ast.Call(
                func=ast.Attribute(
                    value=ast.Call(func=ast.Name("__import__", ast.Load()),
                                   args=[ast.Constant("asyncio")], keywords=[]),
                    attr="gather", ctx=ast.Load()),
                args=[s.value.value for s in run], keywords=[])
            target = ast.Tuple([ast.Name(s.targets[0].id, ast.Store()) for s in run], ast.Store())
            body.append(ast.copy_location(ast.Assign(targets=[target], value=ast.Await(gather)), first))
        else:
            body.extend(run)
        body.extend(held)
        run.clear()
        held.clear()

    for stmt in tree.body:
        name = tool_await(stmt)
        if run and name:
            bound = {s.targets[0].id for s in run}
            between_stores = set().union(*(stores(h) for h in held)) if held else set()
            between_loads = set().union(*(loads(h) for h in held)) if held else set()
            if (not (loads(stmt.value) & (bound | between_stores))
                    and name not in bound | between_stores | between_loads):
                run.append(stmt)
                continue
        elif run and movable(stmt):
            held.append(stmt)
            continue
        flush()
        if name:
            run.append(stmt)
        else:
            body.append(stmt)
    flush()
    tree.body = body


def _compile_cell(source: str, ns: Optional[Dict[str, Any]] = None):
    """Compile a cell like IPython: a trailing expression's value becomes the cell result.

    A trailing ``;`` suppresses the result, as in IPython.
    """
    import ast

    tree = ast.parse(source, "<rlm>", "exec")
    if ns is not None and _auto_await_enabled():
        _await_bare_tool_calls(tree, ns)
        ast.fix_missing_locations(tree)
    if ns is not None and _auto_gather_enabled():
        _gather_independent_tool_awaits(tree, ns)
        ast.fix_missing_locations(tree)
    tail = tree.body[-1] if tree.body else None
    captures = isinstance(tail, ast.Expr) and not source.rstrip().endswith(";")
    if captures:
        tree.body[-1] = ast.copy_location(
            ast.Assign(targets=[ast.Name(_LAST_VALUE_NAME, ast.Store())], value=tail.value),
            tail,
        )
        ast.fix_missing_locations(tree)
    code = compile(tree, "<rlm>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return code, captures


async def _exec_cell(shell: Any, code: str, *, cwd: str) -> Dict[str, Any]:
    import contextlib

    stdout = StringIO()
    stderr = StringIO()
    err_text = None
    result_repr = None
    ns = getattr(shell, "user_ns", None)
    if not isinstance(ns, dict):
        ns = {}

    raw = str(code or "")
    stripped = raw.lstrip()
    if stripped.startswith("%cd"):
        rest = stripped[len("%cd") :].strip()
        root = os.path.realpath(cwd or os.getcwd())
        target_raw = rest or root
        if not os.path.isabs(target_raw):
            target_raw = os.path.join(os.getcwd(), target_raw)
        target = os.path.realpath(target_raw)
        try:
            if target != root and not target.startswith(root + os.sep):
                err_text = "cd_outside_session"
            else:
                os.chdir(target)
                ns["_RLM_CWD"] = os.getcwd()
                stdout.write(os.getcwd() + "\n")
        except Exception as exc:
            err_text = str(exc)
        return {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "error": err_text,
            "result": result_repr,
        }
    if stripped.startswith("%%bash"):
        script = stripped[len("%%bash") :].lstrip("\n")
        # The root comes from the host bootstrap argument, never from the kernel
        # namespace: cell code can rewrite any name it can read.
        session_root = os.path.realpath(cwd or os.getcwd())
        bash_cwd = os.path.realpath(str(ns.get("_RLM_CWD") or session_root))
        if bash_cwd != session_root and not bash_cwd.startswith(session_root + os.sep):
            bash_cwd = session_root
        from src.orchestrators.rlm.runtime.sandbox import bash_script_allowed

        bash_err = bash_script_allowed(script, session_root, cwd=bash_cwd)
        if bash_err:
            return {
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "error": bash_err,
                "result": result_repr,
            }
        try:
            proc = await asyncio.create_subprocess_shell(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=bash_cwd,
                env={
                    k: v
                    for k, v in os.environ.items()
                    if k
                    in {
                        "PATH",
                        "HOME",
                        "LANG",
                        "LC_ALL",
                        "LC_CTYPE",
                        "TZ",
                        "USER",
                        "LOGNAME",
                        "TMPDIR",
                        "TMP",
                        "TEMP",
                        "TERM",
                        "PYTHONPATH",
                        "PYTHONUSERBASE",
                        "PIP_USER",
                        "SSL_CERT_FILE",
                        "REQUESTS_CA_BUNDLE",
                        "CURL_CA_BUNDLE",
                        "http_proxy",
                        "https_proxy",
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "NO_PROXY",
                        "no_proxy",
                    }
                },
            )
            try:
                # Align with cell budget — 0 = no bash wall-clock limit.
                raw_bt = os.environ.get("AIV2_RLM_BASH_TIMEOUT_SEC")
                try:
                    bash_timeout = float(raw_bt) if raw_bt is not None and str(raw_bt).strip() else 0.0
                except ValueError:
                    bash_timeout = 0.0
                if bash_timeout <= 0:
                    out_b, err_b = await proc.communicate()
                else:
                    out_b, err_b = await asyncio.wait_for(
                        proc.communicate(), timeout=bash_timeout
                    )
            except asyncio.TimeoutError:
                proc.kill()
                err_text = "bash_timeout"
                out_b, err_b = b"", b""
            stdout.write(out_b.decode("utf-8", errors="replace"))
            stderr.write(err_b.decode("utf-8", errors="replace"))
            if proc.returncode and not err_text:
                err_text = f"bash_exit_{proc.returncode}"
        except Exception as exc:
            err_text = "".join(traceback.format_exception_only(type(exc), exc))
        return {
            "stdout": _truncate_bash_output(
                stdout.getvalue(), spill_root=session_root
            ),
            "stderr": _truncate_bash_output(
                stderr.getvalue(), spill_root=session_root
            ),
            "error": err_text,
            "result": result_repr,
        }

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if hasattr(shell, "transform_cell"):
                try:
                    raw = shell.transform_cell(raw)
                except Exception:
                    pass
            compiled, captures = _compile_cell(raw, ns)
            # Top-level await: eval(compiled) returns a coroutine module body.
            result = eval(compiled, ns, ns)  # noqa: S307
            if asyncio.iscoroutine(result):
                result = await result
            if captures:
                result = ns.pop(_LAST_VALUE_NAME, None)
                if result is not None:
                    ns["_"] = result
            if result is not None:
                result_repr = repr(result)
            # Keep IPython shell ns in sync when present.
            if hasattr(shell, "user_ns") and shell.user_ns is not ns:
                shell.user_ns.update(ns)
    except Exception as exc:
        err_text = "".join(traceback.format_exception_only(type(exc), exc))
    return {
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "error": err_text,
        "result": result_repr,
    }


async def _main_async() -> None:
    host = _HostProxy()
    user_ns: Dict[str, Any] = {}
    shell = None
    exec_lock = asyncio.Lock()

    try:
        from IPython.core.interactiveshell import InteractiveShell

        shell = InteractiveShell.instance(user_ns=user_ns)
        # Ensure top-level await works against this event loop.
        if hasattr(shell, "enable_gui"):
            try:
                shell.enable_gui("asyncio")
            except Exception:
                pass
    except Exception:
        class _Fallback:
            def __init__(self, ns: Dict[str, Any]):
                self.user_ns = ns

            def run_cell(self, code: str, store_history: bool = True):
                class _R:
                    error_in_exec = None
                    result = None

                r = _R()
                try:
                    compiled = compile(code, "<rlm>", "exec")
                    exec(compiled, self.user_ns, self.user_ns)  # noqa: S102
                except Exception as exc:
                    r.error_in_exec = exc
                return r

            async def run_cell_async(
                self,
                code: str,
                store_history: bool = True,
                transformed_cell: str | None = None,
                preprocessing_exc_tuple=None,
            ):
                return self.run_cell(transformed_cell or code, store_history=store_history)

            def transform_cell(self, code: str) -> str:
                return code

        shell = _Fallback(user_ns)

    _emit({"op": "ready"})
    loop = asyncio.get_event_loop()
    session_cwd = os.getcwd()

    async def handle_exec(msg: Dict[str, Any]) -> None:
        eid = str(msg.get("id") or "")
        code = str(msg.get("code") or "")
        async with exec_lock:
            try:
                result = await _exec_cell(shell, code, cwd=session_cwd)
                ok = not result.get("error")
                _emit(
                    {
                        "op": "exec_result",
                        "id": eid,
                        "ok": ok,
                        "stdout": result.get("stdout") or "",
                        "stderr": result.get("stderr") or "",
                        "error": result.get("error"),
                        "result": result.get("result"),
                    }
                )
            except Exception as exc:
                _emit(
                    {
                        "op": "exec_result",
                        "id": eid,
                        "ok": False,
                        "stdout": "",
                        "stderr": "",
                        "error": str(exc),
                        "result": None,
                    }
                )

    while True:
        line = await loop.run_in_executor(None, _read_line)
        if line is None:
            break
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        op = str(msg.get("op") or "")
        if op == "shutdown":
            break
        if op == "host_response":
            # Must be handled even while a cell is awaiting host_request.
            host.resolve(
                str(msg.get("id") or ""),
                ok=bool(msg.get("ok", True)),
                result=msg.get("result"),
                error=str(msg.get("error") or ""),
            )
            continue
        if op == "bootstrap":
            pools = msg.get("pools") if isinstance(msg.get("pools"), list) else []
            tools = msg.get("tools") if isinstance(msg.get("tools"), list) else []
            cwd = str(msg.get("cwd") or "").strip()
            if cwd:
                try:
                    os.chdir(cwd)
                    session_cwd = cwd
                except Exception:
                    pass
            # Landlock must install BEFORE the Python FS guard patches os.open;
            # otherwise rule setup cannot open /usr and stdlib imports get EACCES.
            try:
                from src.orchestrators.rlm.runtime.landlock_jail import (
                    install_landlock_fs_jail,
                )

                install_landlock_fs_jail(session_cwd)
            except Exception:
                pass
            from src.orchestrators.rlm.runtime.sandbox import install_process_fs_guard

            install_process_fs_guard(session_cwd)
            # Keep temp files inside the session so Landlock allows them.
            sess_tmp = os.path.join(session_cwd, "tmp")
            try:
                os.makedirs(sess_tmp, exist_ok=True)
                os.environ["TMPDIR"] = sess_tmp
                os.environ["TMP"] = sess_tmp
                os.environ["TEMP"] = sess_tmp
                os.environ["HOME"] = session_cwd
            except Exception:
                pass
            try:
                from src.orchestrators.rlm.runtime.session_pip import (
                    apply_session_pip_env,
                )

                apply_session_pip_env(session_cwd)
            except Exception:
                pass
            boot_ns = _build_namespace(host, pools, [str(t) for t in tools])
            bound = surface_namespace({"namespace": msg.get("tools_namespace")})
            boot_ns[bound] = boot_ns["tools"]
            _register_importable(bound, boot_ns["tools"])
            boot_ns["_RLM_CWD"] = os.getcwd()
            boot_ns["_RLM_SESSION_ROOT"] = session_cwd
            user_ns.update(boot_ns)
            if hasattr(shell, "user_ns"):
                shell.user_ns.update(boot_ns)
            _emit({"op": "bootstrap_ok", "id": msg.get("id")})
            continue
        if op == "set_tools":
            tools = msg.get("tools") if isinstance(msg.get("tools"), list) else []
            fresh = _build_namespace(host, [], [str(t) for t in tools])["tools"]
            bound = surface_namespace({"namespace": msg.get("tools_namespace")})
            for target in (user_ns, getattr(shell, "user_ns", None)):
                if isinstance(target, dict):
                    target["tools"] = fresh
                    target[bound] = fresh
                    alias_ns = target.get("aiv2_rlm")
                    if alias_ns is not None:
                        alias_ns.tools = fresh
            _register_importable(bound, fresh)
            continue
        if op == "exec":
            # Concurrent with stdin reader so host_response can arrive mid-cell.
            asyncio.create_task(handle_exec(msg))
            continue
        if op == "snapshot_dump":
            from src.orchestrators.rlm.runtime.snapshot import dump_snapshot

            path = str(msg.get("path") or "")
            ns = getattr(shell, "user_ns", user_ns)
            ok = dump_snapshot(ns, path) if path else False
            _emit({"op": "snapshot_ok" if ok else "snapshot_fail", "id": msg.get("id")})
            continue
        if op == "snapshot_restore":
            from src.orchestrators.rlm.runtime.snapshot import load_snapshot

            path = str(msg.get("path") or "")
            data = load_snapshot(path) if path else None
            ok = False
            if isinstance(data, dict):
                user_ns.update(data)
                if hasattr(shell, "user_ns"):
                    shell.user_ns.update(data)
                ok = True
            _emit({"op": "snapshot_ok" if ok else "snapshot_fail", "id": msg.get("id")})
            continue
        if op == "snapshot_ns":
            keys = [
                k
                for k in list(getattr(shell, "user_ns", user_ns).keys())
                if isinstance(k, str) and not k.startswith("_")
            ]
            _emit({"op": "snapshot_ns_result", "id": msg.get("id"), "keys": keys})
            continue


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
