from __future__ import annotations

import asyncio
import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.job_runtime import AsyncTaskHandle, JobRuntime, ProcessGroupHandle  # noqa: E402
from src.job_queue import Job  # noqa: E402
from src.workloads import WorkloadSpec  # noqa: E402


def test_async_job_status_events_and_cancel() -> None:
    async def _run() -> None:
        runtime = JobRuntime()
        runtime.create(
            job_id="job-1",
            tool_id="plugin_fake_run",
            owner_namespace="fake",
            tenant_id="tenant-1",
            record_id=None,
            resource_class="cpu",
        )
        subscriber = runtime.subscribe("job-1")
        await subscriber.get()
        task = asyncio.create_task(asyncio.sleep(30))
        runtime.attach("job-1", AsyncTaskHandle(task))
        event = await subscriber.get()
        assert event["status"] == "running"
        assert await runtime.cancel("job-1", grace_seconds=0.1)
        assert runtime.get("job-1")["status"] == "cancelled"
        assert task.cancelled()

    asyncio.run(_run())


def test_process_group_handle_is_bounded_and_killable(tmp_path) -> None:
    async def _run() -> None:
        short = await ProcessGroupHandle.spawn(
            [sys.executable, "-c", "print('x' * 20000)"],
            cwd=str(tmp_path),
            env={"PATH": os.environ.get("PATH", "")},
            output_limit_bytes=4096,
        )
        result = await short.wait()
        assert result["returncode"] == 0
        assert len(result["stdout"].encode()) <= 4096
        assert result["output_truncated"] is True

        long = await ProcessGroupHandle.spawn(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(tmp_path),
            env={"PATH": os.environ.get("PATH", "")},
        )
        await long.cancel(grace_seconds=0.1)
        assert long.done

    asyncio.run(_run())


def test_runtime_drain_checkpoints_then_cancels() -> None:
    async def _run() -> None:
        calls = []

        async def checkpoint() -> None:
            calls.append("checkpoint")

        runtime = JobRuntime()
        runtime.create(
            job_id="job-2",
            tool_id="plugin_fake_run",
            owner_namespace="fake",
            tenant_id=None,
            record_id=None,
            resource_class="default",
        )
        task = asyncio.create_task(asyncio.sleep(30))
        runtime.attach("job-2", AsyncTaskHandle(task, checkpoint_fn=checkpoint))
        report = await runtime.drain(
            wait_seconds=0.01,
            checkpoint_seconds=1,
            cancel_seconds=0.1,
        )
        assert report["checkpointed"] == 1
        assert report["cancelled"] == 1
        assert calls == ["checkpoint"]
        assert task.cancelled()

    asyncio.run(_run())


def test_terminal_workload_uses_namespaced_workspace(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        from src.job_runtime import get_job_runtime
        from src.routes.tools import _execute_process

        monkeypatch.setenv("TOOL_RUNNER_WORKSPACE_ROOT", str(tmp_path))
        spec = WorkloadSpec.from_manifest(
            "plugin_terminal_run",
            {
                "workload": {
                    "execution_mode": "detached",
                    "backend": "subprocess",
                    "owner_namespace": "terminal",
                    "resource_class": "subprocess",
                    "sandbox": {
                        "filesystem": "workspace",
                        "executable_allowlist": [sys.executable],
                        "env_allowlist": ["PATH"],
                        "output_limit_bytes": 4096,
                        "required_capability": "sandbox",
                    },
                }
            },
        )
        job = Job(
            job_id="terminal-job",
            tool_id=spec.tool_id,
            owner_namespace=spec.owner_namespace,
            tenant_id="tenant-a",
            record_id="record-a",
            resource_class=spec.resource_class,
            metadata={"required_capability": "sandbox"},
            args={
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('artifact.txt').write_text('ok'); print('done')",
                ],
                "env": {"PATH": os.environ.get("PATH", ""), "SECRET": "blocked"},
            },
        )
        runtime = get_job_runtime()
        runtime.create(
            job_id=job.job_id,
            tool_id=job.tool_id,
            owner_namespace=job.owner_namespace,
            tenant_id=job.tenant_id,
            record_id=job.record_id,
            resource_class=job.resource_class,
        )
        result = await _execute_process(job, spec)
        assert result["returncode"] == 0
        assert result["stdout"].strip() == "done"
        artifact = (
            tmp_path
            / "terminal"
            / "tenant-a"
            / "record-a"
            / "terminal-job"
            / "artifact.txt"
        )
        assert artifact.read_text() == "ok"

    asyncio.run(_run())
