from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_ROOT, "..", ".."))
for _path in (SERVICE_ROOT, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.config import settings
from src.utils.registry_wrapper import ToolRegistryWrapper
from src.workloads import WorkloadRegistry, WorkloadSpec, load_from_tool_registry


def _write_fake_plugin(root: Path) -> None:
    plugin = root / "fake_plugin"
    tool = plugin / "run"
    tool.mkdir(parents=True)
    (plugin / "plugin.json").write_text(
        json.dumps({"id": "fake_plugin", "name": "Fake plugin"}),
        encoding="utf-8",
    )
    (tool / "tool.py").write_text(
        "def run(*, app=None, **kwargs):\n"
        "    return {'ok': True, 'data': kwargs}\n",
        encoding="utf-8",
    )
    (tool / "tool.json").write_text(
        json.dumps(
            {
                "id": "plugin_fake_run",
                "entry": "tool:run",
                "description": "Fake detached plugin",
                "args": {},
                "workload": {
                    "execution_mode": "detached",
                    "backend": "test_queue",
                    "job_id_fields": ["task_id"],
                    "owner_namespace": "fake",
                    "resource_class": "cpu.large",
                    "weight": 2.5,
                    "timeouts": {"soft_sec": 10, "hard_sec": 20},
                    "retry": {
                        "max_attempts": 3,
                        "backoff_sec": 1,
                        "retry_on": ["transient"],
                    },
                    "idempotency": {
                        "mode": "keyed",
                        "key_fields": ["task_id"],
                        "ttl_sec": 60,
                    },
                    "affinity": {
                        "selectors": ["owner"],
                        "id_fields": ["task_id"],
                        "tool_ids": ["plugin_fake_status"],
                    },
                    "checkpoint": {
                        "mode": "periodic",
                        "interval_sec": 5,
                        "resume": True,
                    },
                    "cancel": {"mode": "cooperative", "grace_sec": 2},
                    "sandbox": {
                        "mode": "container",
                        "network": "deny",
                        "filesystem": "readonly",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_wrapper_automatically_loads_fake_plugin_workload(
    tmp_path: Path, monkeypatch
) -> None:
    plugins = tmp_path / "plugins"
    _write_fake_plugin(plugins)
    monkeypatch.setattr(settings, "TOOLS_MANIFEST_PATH", str(tmp_path / "manifest.py"))

    wrapper = ToolRegistryWrapper(
        tools_path=str(Path(REPO_ROOT) / "modules" / "tools"),
        plugins_path=str(plugins),
    )
    assert wrapper.initialize()

    from src.workloads import workload_registry

    spec = workload_registry.spec_for("plugin_fake_run")
    assert spec is not None
    assert spec.execution_mode == "detached"
    assert spec.backend == "test_queue"
    assert spec.job_id_fields == ("task_id",)
    assert spec.owner_namespace == "fake"
    assert spec.resource_class == "cpu.large"
    assert spec.weight == 2.5
    assert (spec.soft_timeout_sec, spec.hard_timeout_sec) == (10, 20)
    assert spec.retry.max_attempts == 3
    assert spec.idempotency.key_fields == ("task_id",)
    assert spec.affinity.id_fields == ("task_id",)
    assert spec.checkpoint.resume is True
    assert spec.cancel.grace_sec == 2
    assert spec.sandbox.network == "deny"
    workload_registry.clear()


def test_manifest_adapter_owns_persist_stamping_hook() -> None:
    calls = []

    class Adapter:
        meta = {
            "id": "plugin_fake_run",
            "workload": {
                "execution_mode": "detached",
                "adapter": f"{__name__}:register_fake_adapter",
            },
        }

    class Registry:
        def list(self):
            return {"plugin_fake_run": {}}

        def get_adapter(self, tool_id):
            return Adapter()

    global _FAKE_ADAPTER_CALLS
    _FAKE_ADAPTER_CALLS = calls
    target = load_from_tool_registry(Registry(), target=WorkloadRegistry())
    result = {"persisted": [{"artifact_id": "a"}]}
    asyncio.run(target.stamp_persisted("plugin_fake_run", result))
    assert calls == ["plugin_fake_run"]
    assert result["stamped"] is True


def test_workload_event_stream_hook_receives_generic_job_context() -> None:
    spec = WorkloadSpec.from_manifest(
        "plugin_fake_run",
        {
            "workload": {
                "execution_mode": "detached",
                "job_id_fields": ["run_id"],
            }
        },
    )
    registry = WorkloadRegistry()

    async def events(job_id, context):
        yield f"{job_id}:{context['user_id']}"

    registry.register(spec, event_stream_hook=events)

    async def collect():
        stream = registry.event_stream(
            "plugin_fake_run",
            "run-1",
            {"user_id": "user-1"},
        )
        return [value async for value in stream]

    assert asyncio.run(collect()) == ["run-1:user-1"]


def register_fake_adapter(*, registry, spec, metadata) -> None:
    del metadata

    def stamp(result):
        _FAKE_ADAPTER_CALLS.append(spec.tool_id)
        result["stamped"] = True

    registry.register(spec, persist_stamp_hook=stamp)


def test_invalid_timeout_contract_is_rejected() -> None:
    metadata = {
        "workload": {
            "execution_mode": "detached",
            "job_id_fields": ["job_id"],
            "soft_timeout_sec": 20,
            "hard_timeout_sec": 10,
        }
    }
    try:
        WorkloadSpec.from_manifest("bad", metadata)
    except ValueError as exc:
        assert "soft timeout" in str(exc)
    else:
        raise AssertionError("invalid workload was accepted")


def test_subprocess_contract_requires_explicit_sandbox_limits() -> None:
    incomplete = {
        "workload": {
            "execution_mode": "detached",
            "backend": "subprocess",
            "owner_namespace": "terminal",
            "sandbox": {"filesystem": "workspace"},
        }
    }
    try:
        WorkloadSpec.from_manifest("terminal_run", incomplete)
    except ValueError as exc:
        assert "executable_allowlist" in str(exc)
    else:
        raise AssertionError("unrestricted subprocess workload was accepted")

    spec = WorkloadSpec.from_manifest(
        "terminal_run",
        {
            "workload": {
                "execution_mode": "detached",
                "backend": "subprocess",
                "owner_namespace": "terminal",
                "resource_class": "subprocess",
                "sandbox": {
                    "filesystem": "workspace",
                    "executable_allowlist": ["python3"],
                    "env_allowlist": ["PATH", "LANG"],
                    "output_limit_bytes": 8192,
                    "required_capability": "sandbox",
                },
            }
        },
    )
    assert spec.sandbox.executable_allowlist == ("python3",)
    assert spec.sandbox.env_allowlist == ("PATH", "LANG")
    assert spec.sandbox.output_limit_bytes == 8192
    assert spec.sandbox.required_capability == "sandbox"
