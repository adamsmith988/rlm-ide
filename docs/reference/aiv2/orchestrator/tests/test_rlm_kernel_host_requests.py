import asyncio
import time

import pytest

from src.orchestrators.rlm.runtime.kernel import RlmKernel
from src.orchestrators.rlm.runtime.worker import _compile_cell, _register_importable


@pytest.mark.asyncio
async def test_host_requests_are_served_concurrently():
    async def handler(rtype, payload):
        await asyncio.sleep(0.3)
        return {"type": rtype, **payload}

    kernel = RlmKernel(host_handler=handler)
    sent = []

    async def fake_send(obj):
        sent.append((time.monotonic(), obj))

    kernel._send = fake_send
    t0 = time.monotonic()
    for i in range(2):
        await kernel._dispatch_message({"op": "host_request", "id": f"r{i}", "type": "tools.call",
                                        "payload": {"n": i}})
    await asyncio.gather(*list(kernel._host_tasks))

    assert sorted(o["id"] for _, o in sent) == ["r0", "r1"]
    assert all(o["ok"] for _, o in sent)
    assert max(t for t, _ in sent) - t0 < 0.55


class _FakeTools:
    _rlm_gatherable = True

    def __init__(self):
        self.log = []

    def __getattr__(self, name):
        async def call(**kw):
            self.log.append(("start", name, time.monotonic()))
            await asyncio.sleep(0.2)
            self.log.append(("end", name, time.monotonic()))
            return {"ok": True, "name": name, **kw}

        return call


async def _run_cell(source, ns):
    code, _ = _compile_cell(source, ns)
    result = eval(code, ns, ns)
    if asyncio.iscoroutine(result):
        await result


@pytest.mark.asyncio
async def test_independent_tool_awaits_run_together(monkeypatch):
    monkeypatch.delenv("AIV2_RLM_AUTO_GATHER_TOOLS", raising=False)
    ns = {"tools": _FakeTools()}
    await _run_cell("a = await tools.weather(city='Oslo')\nb = await tools.rate(base='EUR')\nc = 1", ns)
    starts = [t for kind, _, t in ns["tools"].log if kind == "start"]
    ends = [t for kind, _, t in ns["tools"].log if kind == "end"]
    assert max(starts) < min(ends)
    assert ns["a"]["name"] == "weather" and ns["b"]["name"] == "rate" and ns["c"] == 1


@pytest.mark.asyncio
async def test_dependent_tool_awaits_stay_sequential(monkeypatch):
    monkeypatch.delenv("AIV2_RLM_AUTO_GATHER_TOOLS", raising=False)
    ns = {"tools": _FakeTools()}
    await _run_cell("a = await tools.lookup(e='x')\nb = await tools.cancel(order=a['name'])", ns)
    log = ns["tools"].log
    assert [k for k, _, _ in log] == ["start", "end", "start", "end"]
    assert ns["b"]["order"] == "lookup"


@pytest.mark.asyncio
async def test_prints_between_independent_awaits_still_gather(monkeypatch, capsys):
    import sys

    monkeypatch.delenv("AIV2_RLM_AUTO_GATHER_TOOLS", raising=False)
    tools = _FakeTools()
    monkeypatch.setitem(sys.modules, "rlm_test_tools3", tools)
    ns = {"tools": tools}
    await _run_cell(
        "from rlm_test_tools3 import weather\n"
        "a = await weather(city='Oslo')\nprint('A', a['name'])\n"
        "b = await tools.rate(base='EUR')\nprint('B', b['name'])",
        ns,
    )
    starts = [t for kind, _, t in tools.log if kind == "start"]
    ends = [t for kind, _, t in tools.log if kind == "end"]
    assert max(starts) < min(ends)
    assert capsys.readouterr().out.splitlines() == ["A weather", "B rate"]


@pytest.mark.asyncio
async def test_statement_feeding_a_later_call_keeps_order(monkeypatch):
    monkeypatch.delenv("AIV2_RLM_AUTO_GATHER_TOOLS", raising=False)
    ns = {"tools": _FakeTools()}
    await _run_cell("a = await tools.lookup(e='x')\ncode = a['name'] + '-1'\nb = await tools.cancel(order=code)", ns)
    assert [k for k, _, _ in ns["tools"].log] == ["start", "end", "start", "end"]
    assert ns["b"]["order"] == "lookup-1"


@pytest.mark.asyncio
async def test_auto_gather_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AIV2_RLM_AUTO_GATHER_TOOLS", "0")
    ns = {"tools": _FakeTools()}
    await _run_cell("a = await tools.x()\nb = await tools.y()", ns)
    assert [k for k, _, _ in ns["tools"].log] == ["start", "end", "start", "end"]


@pytest.mark.asyncio
async def test_unawaited_tool_calls_are_awaited(monkeypatch):
    import sys

    monkeypatch.delenv("AIV2_RLM_AUTO_AWAIT_TOOLS", raising=False)
    tools = _FakeTools()
    monkeypatch.setitem(sys.modules, "rlm_test_tools2", tools)
    ns = {"tools": tools}
    await _run_cell(
        "from rlm_test_tools2 import status\n"
        "a = tools.weather(city='Oslo')\n"
        "b = status(code='T1')\n"
        "pending = [tools.rate(base='EUR')]",
        ns,
    )
    assert ns["a"]["name"] == "weather" and ns["b"]["code"] == "T1"
    assert asyncio.iscoroutine(ns["pending"][0])
    ns["pending"][0].close()


def test_tools_namespace_is_importable_without_shadowing(monkeypatch):
    import sys

    tools = _FakeTools()
    monkeypatch.delitem(sys.modules, "rlm_test_tools", raising=False)
    _register_importable("rlm_test_tools", tools)
    from rlm_test_tools import lookup  # noqa: F401

    assert sys.modules["rlm_test_tools"] is tools
    _register_importable("json", tools)
    assert sys.modules["json"] is not tools
    monkeypatch.delitem(sys.modules, "rlm_test_tools")


def test_tool_usage_hint_comes_from_config():
    from src.orchestrators._shared.manifest_config import manifest_defaults, package_manifest
    from src.orchestrators.rlm.runtime import session as rlm_session
    from src.orchestrators.rlm.runtime.prompts import format_tool_surface

    default = manifest_defaults(package_manifest(rlm_session.__file__)).get("rlm_tool_usage_hint")
    assert default and "{namespace}" in default
    on = format_tool_surface([], ["e2e_x"], namespace="kit", usage_hint=default)
    assert "kit.<x>" in on and "{namespace}" not in on
    off = format_tool_surface([], ["e2e_x"], namespace="kit", usage_hint="")
    assert "asyncio.gather" not in off


@pytest.mark.asyncio
async def test_host_request_failure_is_reported():
    async def handler(rtype, payload):
        raise RuntimeError("boom")

    kernel = RlmKernel(host_handler=handler)
    sent = []

    async def fake_send(obj):
        sent.append(obj)

    kernel._send = fake_send
    await kernel._dispatch_message({"op": "host_request", "id": "r", "type": "tools.call", "payload": {}})
    await asyncio.gather(*list(kernel._host_tasks))
    assert sent == [{"op": "host_response", "id": "r", "ok": False, "error": "boom", "result": None}]
