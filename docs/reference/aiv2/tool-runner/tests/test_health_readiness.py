from __future__ import annotations

import asyncio
import json
import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_ROOT, "..", ".."))
for _p in (SERVICE_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import job_queue  # noqa: E402
from src.job_queue import JobQueue  # noqa: E402
from src.config import settings  # noqa: E402
from src.routes import health as health_route  # noqa: E402
from tests.fake_redis import FakeRedis  # noqa: E402


class _Registry:
    def __init__(self, ready: bool):
        self.is_ready = ready


def _run(coro):
    return asyncio.run(coro)


def test_healthz_stays_live_during_redis_outage(monkeypatch):
    async def _t():
        monkeypatch.setattr(settings, "TOOL_RUNNER_ROLE", "api")
        monkeypatch.setattr(settings, "TOOL_JOB_REDIS_URL", "redis://down")
        job_queue.set_queue_for_tests(None)
        assert await health_route.healthz() == {"status": "ok"}

    _run(_t())


def test_readyz_requires_registry_and_api_producer(monkeypatch):
    async def _t():
        monkeypatch.setattr(settings, "TOOL_RUNNER_ROLE", "api")
        monkeypatch.setattr(settings, "TOOL_JOB_REDIS_URL", "redis://fake")
        monkeypatch.setattr(health_route, "get_registry", lambda: _Registry(True))
        job_queue.set_queue_for_tests(None)
        response = await health_route.readyz()
        assert response.status_code == 503
        assert json.loads(response.body)["queue"]["producer"] is False

        q = JobQueue(FakeRedis(), pod="api")
        await q.connect()
        job_queue.set_queue_for_tests(q)
        response = await health_route.readyz()
        assert response.status_code == 200

        monkeypatch.setattr(health_route, "get_registry", lambda: _Registry(False))
        assert (await health_route.readyz()).status_code == 503
        await job_queue.close_queue()

    _run(_t())


def test_readyz_worker_requires_consumer(monkeypatch):
    async def _t():
        monkeypatch.setattr(settings, "TOOL_RUNNER_ROLE", "worker")
        monkeypatch.setattr(settings, "TOOL_JOB_REDIS_URL", "redis://fake")
        monkeypatch.setattr(health_route, "get_registry", lambda: _Registry(True))
        q = JobQueue(FakeRedis(), pod="worker")
        await q.connect()
        job_queue.set_queue_for_tests(q)
        assert (await health_route.readyz()).status_code == 503

        async def _exec(_job):
            return None

        await job_queue.start_worker(_exec)
        assert (await health_route.readyz()).status_code == 200
        await job_queue.shutdown_queue()
        await job_queue.close_queue()

    _run(_t())


def test_metrics_are_prometheus_text(monkeypatch):
    async def _t():
        async def fake_health():
            return {
                "job_queue": {
                    "depth": {"shared": 2, "pod": 1},
                    "consumer": True,
                },
                "admission": {"running": 3, "queued": 4},
                "event_loop": {"lag_seconds": 0.1, "max_lag_seconds": 0.2},
            }

        monkeypatch.setattr(health_route, "_job_health", fake_health)
        response = await health_route.prometheus_metrics()
        body = response.body.decode()
        assert "tool_job_queue_depth 3" in body
        assert "tool_job_admission_running 3" in body
        assert "python_event_loop_lag_seconds 0.100000" in body

    _run(_t())
