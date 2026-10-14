"""Host-side persistent RLM kernel (subprocess worker + JSONL RPC)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from src.orchestrators._shared.tool_surface import DEFAULT_NAMESPACE

logger = logging.getLogger(__name__)

HostHandler = Callable[[str, Dict[str, Any]], Awaitable[Any]]

# asyncio.StreamReader defaults to 64KiB. One JSONL exec_result that embeds a
# large print()/file dump exceeds that and kills the whole reader loop
# ("Separator is not found, and chunk exceed the limit" → rlm_kernel_exited).
# Bound protocol frames per kernel. A large default multiplied by dozens of
# children can exhaust a worker even when each child is otherwise healthy.
# The worker clips individual outputs before they reach this reader.
_DEFAULT_STDOUT_LIMIT = 8 * 1024 * 1024
_MAX_STDOUT_LIMIT = 64 * 1024 * 1024


def _stdout_stream_limit() -> int:
    raw = (os.environ.get("AIV2_RLM_STDOUT_LIMIT") or "").strip()
    if not raw:
        return _DEFAULT_STDOUT_LIMIT
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_STDOUT_LIMIT
    # Floor at the asyncio default so we never silently regress to fragile 64KiB.
    return max(64 * 1024, min(val, _MAX_STDOUT_LIMIT))

_ENV_ALLOW = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "USER",
        "LOGNAME",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONUSERBASE",
        "PIP_USER",
        "AIV2_RLM_ALLOW_PIP",
        "AIV2_RLM_AUTO_GATHER_TOOLS",
        "AIV2_RLM_AUTO_AWAIT_TOOLS",
    }
)


def worker_env(
    *,
    session_dir: Optional[str] = None,
    bash_timeout_sec: Optional[float] = None,
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key in _ENV_ALLOW:
        val = os.environ.get(key)
        if val:
            env[key] = val
    # The kernel changes cwd to its isolated workspace, so it cannot import the
    # worker module unless the application root remains explicit.
    app_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    import_roots = [
        path if os.path.isabs(path) else os.path.abspath(path)
        for path in str(env.get("PYTHONPATH") or "").split(os.pathsep)
        if path
    ]
    if app_root not in import_roots:
        import_roots.insert(0, app_root)
    env["PYTHONPATH"] = os.pathsep.join(import_roots)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if bash_timeout_sec is not None:
        bt = float(bash_timeout_sec)
        env["AIV2_RLM_BASH_TIMEOUT_SEC"] = str(int(bt) if bt > 0 else 0)
    elif "AIV2_RLM_BASH_TIMEOUT_SEC" not in env:
        env["AIV2_RLM_BASH_TIMEOUT_SEC"] = os.environ.get("AIV2_RLM_BASH_TIMEOUT_SEC", "0")
    if session_dir:
        sess = os.path.realpath(session_dir)
        env["HOME"] = sess
        tmp = os.path.join(sess, "tmp")
        try:
            os.makedirs(tmp, exist_ok=True)
        except OSError:
            tmp = sess
        env["TMPDIR"] = tmp
        env["TMP"] = tmp
        env["TEMP"] = tmp
        from src.orchestrators.rlm.runtime.session_pip import (
            configure_session_pip,
        )

        pip_env = configure_session_pip(sess)
        if pip_env.get("PYTHONPATH"):
            # Keep app_root import path ahead of / merged with session site-packages.
            site_first = [
                p
                for p in str(pip_env["PYTHONPATH"]).split(os.pathsep)
                if p and p not in import_roots
            ]
            env["PYTHONPATH"] = os.pathsep.join(site_first + import_roots)
        for key, val in pip_env.items():
            if key == "PYTHONPATH":
                continue
            if key == "PYTHONNOUSERSITE" and not val:
                env.pop("PYTHONNOUSERSITE", None)
                continue
            if key == "PATH":
                host_path = str(env.get("PATH") or os.environ.get("PATH") or "")
                parts = [p for p in str(val).split(os.pathsep) if p]
                for p in host_path.split(os.pathsep):
                    if p and p not in parts:
                        parts.append(p)
                env["PATH"] = os.pathsep.join(parts)
                continue
            env[key] = val
    return env


class RlmKernel:
    """Owns a worker subprocess and routes host_request callbacks."""

    def __init__(
        self,
        *,
        host_handler: HostHandler,
        timeout_sec: float = 0.0,
        python_executable: Optional[str] = None,
        cwd: Optional[str] = None,
        snapshot_path: Optional[str] = None,
    ) -> None:
        self._host_handler = host_handler
        self.timeout_sec = float(timeout_sec)
        self.python_executable = python_executable or sys.executable
        self.cwd = cwd or os.getcwd()
        self.snapshot_path = snapshot_path
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._pending_exec: Dict[str, asyncio.Future] = {}
        self._pending_boot: Dict[str, asyncio.Future] = {}
        self._pending_snap: Dict[str, asyncio.Future] = {}
        self._host_tasks: Set[asyncio.Task] = set()
        self._counter = 0
        self._ready = asyncio.Event()
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._pools: List[Dict[str, Any]] = []
        self._tools: List[str] = []
        self._tool_namespace: str = DEFAULT_NAMESPACE

    @property
    def alive(self) -> bool:
        return (
            not self._closed
            and self._proc is not None
            and self._proc.returncode is None
        )

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    async def start(self) -> None:
        if self.alive:
            return
        self._closed = False
        self._ready = asyncio.Event()
        from src.orchestrators.rlm.runtime.bwrap_jail import resolve_worker_argv

        argv, jailed = resolve_worker_argv(
            python_executable=self.python_executable,
            session_dir=self.cwd,
        )
        if jailed:
            logger.info("rlm kernel spawning inside bwrap session=%s", self.cwd)
        else:
            logger.info("rlm kernel spawning without bwrap session=%s", self.cwd)
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=worker_env(session_dir=self.cwd, bash_timeout_sec=self.timeout_sec),
            cwd=None if jailed else self.cwd,
            limit=_stdout_stream_limit(),
            # Bash/tool subprocesses inherit this group. Cancellation can then
            # terminate the complete tree instead of leaking descendants.
            start_new_session=True,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError as exc:
            # Surface worker stderr to help diagnose bwrap/import failures.
            err_tail = ""
            try:
                if self._proc and self._proc.stderr:
                    # Non-blocking best-effort drain is hard; rely on shutdown path.
                    pass
            except Exception:
                pass
            await self.shutdown()
            raise RuntimeError(f"rlm_kernel_start_timeout{err_tail}") from exc

    async def bootstrap(
        self,
        *,
        pools: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[str]] = None,
        tool_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start worker, restore snapshot BEFORE bootstrap (Prime order), then boot bridges.

        Returns restore metadata for ipython_state notices.
        """
        self._pools = list(pools or [])
        self._tools = list(tools or [])
        if tool_namespace:
            self._tool_namespace = tool_namespace
        await self.start()
        restored = False
        restored_names: List[str] = []
        if self.snapshot_path:
            restored = await self.restore_snapshot(self.snapshot_path)
            if restored:
                try:
                    restored_names = await self.list_namespace_names()
                except Exception:
                    restored_names = []
        bid = self._next_id("boot")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_boot[bid] = fut
        await self._send(
            {
                "op": "bootstrap",
                "id": bid,
                "pools": self._pools,
                "tools": self._tools,
                "tools_namespace": self._tool_namespace,
                "cwd": self.cwd,
            }
        )
        await asyncio.wait_for(fut, timeout=30.0)
        return {"restored": restored, "names": restored_names}

    async def set_tools(self, tools: Optional[List[str]], *, namespace: Optional[str] = None) -> None:
        """Rebind the kernel's tools allowlist (kept for restore/re-bootstrap)."""
        self._tools = list(tools or [])
        if namespace:
            self._tool_namespace = namespace
        if self.alive:
            await self._send(
                {"op": "set_tools", "tools": self._tools, "tools_namespace": self._tool_namespace}
            )

    async def list_namespace_names(self) -> List[str]:
        if not self.alive:
            return []
        sid = self._next_id("ns")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_snap[sid] = fut
        await self._send({"op": "snapshot_ns", "id": sid})
        try:
            keys = await asyncio.wait_for(fut, timeout=5.0)
            if isinstance(keys, list):
                return [str(k) for k in keys if k]
            return []
        except Exception:
            self._pending_snap.pop(sid, None)
            return []

    async def execute(self, code: str, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        await self.start()
        eid = self._next_id("exec")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_exec[eid] = fut
        await self._send({"op": "exec", "id": eid, "code": str(code or "")})
        timeout = float(timeout_sec if timeout_sec is not None else self.timeout_sec)
        # 0 / negative ⇒ no wall-clock limit (Prime-style).
        if timeout <= 0:
            return await fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return await self._on_exec_timeout(eid, fut)

    async def interrupt_soft(self) -> bool:
        """Ask the worker to abort the current cell via SIGINT (no process restart)."""
        if not self._proc or self._proc.returncode is not None:
            return False
        try:
            import signal

            self._proc.send_signal(signal.SIGINT)
            return True
        except Exception as exc:
            logger.warning("rlm soft interrupt failed: %s", exc)
            return False

    async def _on_exec_timeout(
        self, eid: str, fut: asyncio.Future
    ) -> Dict[str, Any]:
        """Soft-interrupt first; restart the worker only if the cell stays stuck."""
        await self.interrupt_soft()
        try:
            result = await asyncio.wait_for(fut, timeout=8.0)
            if isinstance(result, dict):
                out = dict(result)
                out["ok"] = False
                err = str(out.get("error") or "").strip()
                if not err or "KeyboardInterrupt" in err:
                    out["error"] = "code_exec_timeout"
                elif "code_exec_timeout" not in err:
                    out["error"] = f"code_exec_timeout; {err}"
                return out
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        self._pending_exec.pop(eid, None)
        if not fut.done():
            fut.cancel()
        # Last resort: cell ignored SIGINT — restart so the next turn can run.
        logger.warning("rlm cell still stuck after SIGINT; restarting worker")
        await self.interrupt_and_restore()
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "code_exec_timeout",
            "result": None,
        }

    async def dump_snapshot(self, path: str) -> bool:
        if not self.alive:
            return False
        sid = self._next_id("snap")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_snap[sid] = fut
        await self._send({"op": "snapshot_dump", "id": sid, "path": path})
        try:
            ok = await asyncio.wait_for(fut, timeout=5.0)
            return bool(ok)
        except Exception:
            self._pending_snap.pop(sid, None)
            return False

    async def restore_snapshot(self, path: str) -> bool:
        if not path or not (os.path.exists(path) or os.path.exists(path + ".json")):
            return False
        if not self.alive:
            return False
        sid = self._next_id("snap")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_snap[sid] = fut
        await self._send({"op": "snapshot_restore", "id": sid, "path": path})
        try:
            return bool(await asyncio.wait_for(fut, timeout=5.0))
        except Exception:
            self._pending_snap.pop(sid, None)
            return False

    async def interrupt_and_restore(self) -> None:
        """Kill the worker and bring a fresh one back with the last snapshot."""
        await self.shutdown(dump_snapshot=False)
        self._closed = False
        try:
            # bootstrap restores snapshot before re-injecting bridges.
            await self.bootstrap(pools=self._pools, tools=self._tools)
        except Exception as exc:
            logger.warning("rlm kernel restore after interrupt failed: %s", exc)

    async def shutdown(self, *, dump_snapshot: bool = True) -> None:
        if self._closed and not self.alive:
            return
        if dump_snapshot and self.snapshot_path and self.alive:
            try:
                await self.dump_snapshot(self.snapshot_path)
            except Exception:
                pass
        self._closed = True
        try:
            if self._proc and self._proc.returncode is None and self._proc.stdin:
                await self._send({"op": "shutdown"})
        except Exception:
            pass
        if self._proc and self._proc.returncode is None:
            try:
                try:
                    os.killpg(self._proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except Exception:
                try:
                    try:
                        os.killpg(self._proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        self._proc.kill()
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except Exception:
                    pass
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except Exception:
                pass
        for task in list(self._host_tasks):
            task.cancel()
        for fut in (
            list(self._pending_exec.values())
            + list(self._pending_boot.values())
            + list(self._pending_snap.values())
        ):
            if not fut.done():
                fut.set_exception(RuntimeError("rlm_kernel_shutdown"))
        self._pending_exec.clear()
        self._pending_boot.clear()
        self._pending_snap.clear()
        self._proc = None

    async def _send(self, obj: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("rlm_kernel_not_started")
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()

    async def _discard_oversize_line(self) -> bool:
        """Skip one oversize JSONL frame after LimitOverrunError.

        Returns False when the stream is closed. asyncio may already have
        cleared the internal buffer; we still scan for the next newline so a
        single huge frame does not tear down the whole kernel reader.
        """
        assert self._proc and self._proc.stdout
        stream = self._proc.stdout
        # Prefer readuntil with an elevated temporary limit when available.
        try:
            await asyncio.wait_for(
                stream.readuntil(separator=b"\n"),
                timeout=30.0,
            )
            return True
        except asyncio.LimitOverrunError as exc:
            # Still too large even for the raised limit — drop consumed bytes
            # and keep eating until newline or EOF.
            drop = max(int(getattr(exc, "consumed", 0) or 0), 1)
            chunk = await stream.read(drop)
            if not chunk:
                return False
        except asyncio.IncompleteReadError:
            return False
        except (ValueError, asyncio.TimeoutError):
            pass
        except Exception as exc:
            logger.warning("rlm kernel oversize discard failed: %s", exc)
            return False
        # Byte-scan fallback: read until newline without relying on readuntil.
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return False
            if b"\n" in chunk:
                return True

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                try:
                    line = await self._proc.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError) as exc:
                    # ValueError wraps LimitOverrunError inside readline().
                    msg = str(exc)
                    if "chunk" not in msg.lower() and "separator" not in msg.lower():
                        raise
                    logger.warning(
                        "rlm kernel reader skipped oversize JSONL frame: %s",
                        exc,
                    )
                    if not await self._discard_oversize_line():
                        break
                    # Fail only the in-flight exec that produced the huge line;
                    # keep the reader alive for sibling cells / parallel agents.
                    for fut in list(self._pending_exec.values()):
                        if not fut.done():
                            fut.set_result(
                                {
                                    "ok": False,
                                    "stdout": "",
                                    "stderr": "",
                                    "error": "rlm_kernel_stdout_too_large",
                                    "result": None,
                                }
                            )
                    self._pending_exec.clear()
                    continue
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("rlm worker non-json: %s", text[:200])
                    continue
                if not isinstance(msg, dict):
                    continue
                await self._dispatch_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("rlm kernel reader failed: %s", exc)
        finally:
            for fut in list(self._pending_exec.values()):
                if not fut.done():
                    fut.set_result(
                        {
                            "ok": False,
                            "stdout": "",
                            "stderr": "",
                            "error": "rlm_kernel_exited",
                            "result": None,
                        }
                    )
            self._pending_exec.clear()

    async def _dispatch_message(self, msg: Dict[str, Any]) -> None:
        op = str(msg.get("op") or "")
        if op == "ready":
            self._ready.set()
            return
        if op == "bootstrap_ok":
            bid = str(msg.get("id") or "")
            fut = self._pending_boot.pop(bid, None)
            if fut and not fut.done():
                fut.set_result(True)
            return
        if op == "exec_result":
            eid = str(msg.get("id") or "")
            fut = self._pending_exec.pop(eid, None)
            if fut and not fut.done():
                fut.set_result(
                    {
                        "ok": bool(msg.get("ok")),
                        "stdout": str(msg.get("stdout") or ""),
                        "stderr": str(msg.get("stderr") or ""),
                        "error": msg.get("error"),
                        "result": msg.get("result"),
                    }
                )
            return
        if op in ("snapshot_ok", "snapshot_fail"):
            sid = str(msg.get("id") or "")
            fut = self._pending_snap.pop(sid, None)
            if fut and not fut.done():
                fut.set_result(op == "snapshot_ok")
            return
        if op == "snapshot_ns_result":
            sid = str(msg.get("id") or "")
            fut = self._pending_snap.pop(sid, None)
            if fut and not fut.done():
                keys = msg.get("keys") if isinstance(msg.get("keys"), list) else []
                fut.set_result(keys)
            return
        if op == "host_request":
            # Served off the reader loop so concurrent kernel awaits (asyncio.gather) overlap.
            task = asyncio.create_task(self._serve_host_request(msg))
            self._host_tasks.add(task)
            task.add_done_callback(self._host_tasks.discard)
            return

    async def _serve_host_request(self, msg: Dict[str, Any]) -> None:
        rid = str(msg.get("id") or "")
        rtype = str(msg.get("type") or "")
        payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
        try:
            reply = {"op": "host_response", "id": rid, "ok": True,
                     "result": await self._host_handler(rtype, payload)}
        except Exception as exc:
            logger.warning("rlm host_request failed type=%s err=%s", rtype, exc)
            reply = {"op": "host_response", "id": rid, "ok": False, "error": str(exc), "result": None}
        try:
            await self._send(reply)
        except Exception as exc:
            logger.warning("rlm host_response send failed type=%s err=%s", rtype, exc)
