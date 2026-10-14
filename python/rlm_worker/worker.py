"""RLM kernel worker: a persistent Python REPL speaking JSONL over stdin/stdout.

Host -> worker (one JSON object per line):
  {"op":"bootstrap","tools":[{"id":"fs.read_file","params":["path",...],"doc":"..."}],"output_limit":16000}
  {"op":"exec","id":"c1","code":"..."}
  {"op":"host_response","id":"hr_1","ok":true,"result":...,"error":null}
  {"op":"interrupt"}
  {"op":"reset"}
  {"op":"shutdown"}

Worker -> host:
  {"op":"ready","pid":123,"python":"3.12.3"}
  {"op":"stream","id":"c1","stream":"stdout"|"stderr","text":"..."}
  {"op":"exec_result","id":"c1","ok":true,"stdout":"...","stderr":"...","result":"...","error":null,"interrupted":false}
  {"op":"host_request","id":"hr_1","type":"tool"|"spawn","payload":{...}}

Cells run on the main thread so SIGINT / interrupt_main raise KeyboardInterrupt inside user
code. A reader thread routes host responses so tool calls can block the cell.
"""

from __future__ import annotations

import _thread
import ast
import builtins
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
import types
from typing import Any, Dict, List, Optional

from rewrite_guard import RewriteGuard

_REAL_STDOUT = sys.__stdout__
_EMIT_LOCK = threading.Lock()
# Stream chunks are flushed at most this often so a tight print loop does not flood the host.
_STREAM_FLUSH_SEC = 0.05


def emit(obj: Dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with _EMIT_LOCK:
        _REAL_STDOUT.write(line + "\n")
        _REAL_STDOUT.flush()


class ToolError(Exception):
    """Raised in the kernel when a tool call fails; the message explains why."""


class _HostChannel:
    def __init__(self) -> None:
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._n = 0

    def request(self, rtype: str, payload: Dict[str, Any]) -> Any:
        with self._lock:
            self._n += 1
            rid = f"hr_{self._n}"
            ev = threading.Event()
            self._events[rid] = ev
        emit({"op": "host_request", "id": rid, "type": rtype, "payload": payload})
        while not ev.wait(0.1):
            pass
        with self._lock:
            self._events.pop(rid, None)
            msg = self._pending.pop(rid, {})
        if not msg.get("ok"):
            raise ToolError(str(msg.get("error") or "host request failed"))
        return msg.get("result")

    def resolve(self, msg: Dict[str, Any]) -> None:
        rid = str(msg.get("id"))
        with self._lock:
            ev = self._events.get(rid)
            if ev is None:
                return
            self._pending[rid] = msg
        ev.set()


class _StreamWriter:
    """Replaces sys.stdout/stderr during a cell: streams chunks to the host and keeps a copy."""

    def __init__(self, cell_id: str, name: str, limit: int) -> None:
        self.cell_id = cell_id
        self.name = name
        self.limit = limit
        self.parts: List[str] = []
        self.size = 0
        self.pending: List[str] = []
        self.last_flush = 0.0
        self.encoding = "utf-8"

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        if self.size < self.limit:
            self.parts.append(s)
        self.size += len(s)
        self.pending.append(s)
        now = time.monotonic()
        if now - self.last_flush >= _STREAM_FLUSH_SEC or "\n" in s and len(self.pending) > 64:
            self.flush()
        return len(s)

    def flush(self) -> None:
        if self.pending:
            text = "".join(self.pending)
            self.pending = []
            self.last_flush = time.monotonic()
            emit({"op": "stream", "id": self.cell_id, "stream": self.name, "text": text[: self.limit]})

    def isatty(self) -> bool:
        return False

    def value(self) -> str:
        text = "".join(self.parts)
        if self.size > self.limit:
            text = text[: self.limit] + f"\n[truncated {self.size - self.limit} chars; print less, use line ranges or grep]"
        return text


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n[truncated {len(s) - limit} chars]"


class Kernel:
    def __init__(self) -> None:
        self.host = _HostChannel()
        self.tools: List[Dict[str, Any]] = []
        self.output_limit = 16000
        self.ns: Dict[str, Any] = {}
        self.busy = False
        self.guard: Optional[RewriteGuard] = None

    def build_namespace(self) -> None:
        ns: Dict[str, Any] = {"__name__": "__main__", "ToolError": ToolError}
        if self.guard:
            ns["open"] = self.guard.open
        host = self.host
        groups: Dict[str, types.ModuleType] = {}
        for t in self.tools:
            tid = str(t["id"])
            if "." not in tid:
                continue
            group, name = tid.split(".", 1)
            params = list(t.get("params") or [])

            def make(tool_id: str, params: List[str]):
                def call(*args: Any, **kwargs: Any) -> Any:
                    if args:
                        if len(args) > len(params):
                            raise ToolError(f"{tool_id} takes at most {len(params)} positional arguments")
                        for p, v in zip(params, args):
                            kwargs.setdefault(p, v)
                    res = host.request("tool", {"tool_id": tool_id, "args": kwargs})
                    if isinstance(res, dict) and res.get("ok") is False:
                        raise ToolError(f"{tool_id}: {res.get('error')}")
                    if isinstance(res, dict) and "result" in res:
                        return res["result"]
                    return res

                call.__name__ = tool_id.split(".", 1)[1]
                call.__doc__ = str(t.get("doc") or "")
                return call

            ns_obj = groups.setdefault(group, types.ModuleType(group, f"RLM tools: {group}"))
            setattr(ns_obj, name, make(tid, params))
        ns.update(groups)
        # Models often write `import fs` / `from fs import read_file`; make the namespaces importable.
        for old in getattr(self, "_tool_modules", set()) - set(groups):
            sys.modules.pop(old, None)
        for group, mod in groups.items():
            if group not in sys.modules or sys.modules[group] is mod or getattr(sys.modules[group], "__doc__", "").startswith("RLM tools:"):
                sys.modules[group] = mod
        self._tool_modules = set(groups)

        def spawn_agents(tasks: List[str], tool_ids: Optional[List[str]] = None) -> List[str]:
            """Run read-only sub-agents in parallel; returns their final answers in order."""
            if isinstance(tasks, str):
                tasks = [tasks]
            res = host.request("spawn", {"tasks": [str(x) for x in tasks], "tool_ids": tool_ids})
            if isinstance(res, dict) and res.get("ok") is False:
                raise ToolError(str(res.get("error")))
            return res.get("result") if isinstance(res, dict) and "result" in res else res

        ns["spawn_agents"] = spawn_agents
        self.ns = ns

    def execute(self, cell_id: str, code: str) -> None:
        out = _StreamWriter(cell_id, "stdout", self.output_limit)
        err = _StreamWriter(cell_id, "stderr", self.output_limit)
        result_text = ""
        error: Optional[str] = None
        interrupted = False
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        self.busy = True
        if self.guard:
            self.guard.install()
        try:
            tree = ast.parse(code, mode="exec")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), self.ns)
            if last_expr is not None:
                value = eval(compile(last_expr, "<cell>", "eval"), self.ns)
                if value is not None:
                    self.ns["_"] = value
                    result_text = _clip(repr(value), self.output_limit)
        except KeyboardInterrupt:
            interrupted = True
            error = "KeyboardInterrupt: the cell was interrupted"
        except SyntaxError as e:
            error = f"SyntaxError: {e.msg} (line {e.lineno})\n{(e.text or '').rstrip()}"
        except ToolError as e:
            error = f"ToolError: {e}"
        except BaseException:  # noqa: BLE001 - report every failure to the model
            etype, evalue, tb = sys.exc_info()
            frames = [f for f in traceback.extract_tb(tb) if f.filename == "<cell>"]
            lines = traceback.format_list(frames) + traceback.format_exception_only(etype, evalue)
            error = _clip("Traceback (most recent call last):\n" + "".join(lines), 6000)
        finally:
            self.busy = False
            refused = self.guard.uninstall() if self.guard else None
            if refused and (error is None or refused not in error):
                error = f"PermissionError: {refused}" if error is None else f"{error}\n{refused}"
            try:
                out.flush()
                err.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = old_out, old_err
        emit({
            "op": "exec_result",
            "id": cell_id,
            "ok": error is None,
            "stdout": out.value(),
            "stderr": err.value(),
            "result": result_text,
            "error": error,
            "interrupted": interrupted,
        })


def apply_resource_limits() -> None:
    """Windows: put this process (and its children) in a Job Object with kill-on-close and a
    memory cap. The worker owns the only handle, so when it dies its children die too."""
    mb = int(os.environ.get("RLM_MEMORY_MB") or "0")
    if sys.platform != "win32" or mb <= 0:
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        job = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = mb * 1024 * 1024
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
        globals()["_JOB_HANDLE"] = job
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"job object setup failed: {e}\n")


def main() -> None:
    apply_resource_limits()
    kernel = Kernel()
    work: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    def on_sigint(_signum: int, _frame: Any) -> None:
        if kernel.busy:
            raise KeyboardInterrupt()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, on_sigint)

    def reader() -> None:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            op = msg.get("op")
            if op == "host_response":
                kernel.host.resolve(msg)
            elif op == "interrupt":
                if kernel.busy:
                    _thread.interrupt_main()
            else:
                work.put(msg)
        # The host is gone: a running cell may be blocked on a host call that can never be answered.
        if kernel.busy:
            os._exit(0)
        work.put({"op": "shutdown"})

    threading.Thread(target=reader, daemon=True).start()
    kernel.build_namespace()
    emit({"op": "ready", "pid": os.getpid(), "python": sys.version.split()[0]})

    while True:
        try:
            msg = work.get()
        except KeyboardInterrupt:
            continue
        op = msg.get("op")
        try:
            if op == "bootstrap":
                kernel.tools = list(msg.get("tools") or [])
                kernel.output_limit = int(msg.get("output_limit") or kernel.output_limit)
                rg = msg.get("rewrite_guard")
                if rg:
                    kernel.guard = RewriteGuard(os.getcwd(), int(rg["min_lines"]), float(rg["max_changed"]), builtins.open)
                kernel.build_namespace()
                emit({"op": "bootstrapped"})
            elif op == "exec":
                kernel.execute(str(msg.get("id")), str(msg.get("code") or ""))
            elif op == "reset":
                kernel.build_namespace()
                emit({"op": "reset_done"})
            elif op == "shutdown":
                break
        except KeyboardInterrupt:
            emit({"op": "exec_result", "id": str(msg.get("id")), "ok": False, "stdout": "", "stderr": "", "result": "", "error": "KeyboardInterrupt", "interrupted": True})


if __name__ == "__main__":
    main()
