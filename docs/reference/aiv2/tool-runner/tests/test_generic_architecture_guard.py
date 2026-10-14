from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = re.compile(
    r"agent[_-]flow|plugin_agent|modules\.plugins\.(?!discovery\b)",
    re.IGNORECASE,
)


def _generic_platform_files() -> list[Path]:
    tool_src = ROOT / "services" / "tool-runner" / "src"
    gateway_src = ROOT / "services" / "api-gateway" / "src"
    files = [
        *sorted((tool_src / "workloads").glob("*.py")),
        *sorted(tool_src.glob("job_*.py")),
        tool_src / "main.py",
        tool_src / "config" / "settings.py",
        tool_src / "utils" / "registry_wrapper.py",
        tool_src / "routes" / "tools.py",
        tool_src / "routes" / "health.py",
        tool_src / "routes" / "jobs.py",
        gateway_src / "workload_affinity.py",
        gateway_src / "routes" / "jobs.py",
        gateway_src / "middleware" / "bulk_transfer.py",
        ROOT / "deploy" / "k8s" / "base" / "tool-runner.yaml",
        ROOT / "deploy" / "k8s" / "observability" / "keda-tool-runner-worker.yaml",
        ROOT / "deploy" / "k8s" / "observability" / "queue-metrics-exporter.yaml",
        ROOT
        / "deploy"
        / "k8s"
        / "overlays"
        / "admin-system"
        / "tool-runner-worker-profiles.yaml",
    ]
    return [path for path in files if path.exists()]


def test_generic_platform_has_no_plugin_specific_coupling() -> None:
    offenders: list[str] = []
    for path in _generic_platform_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if FORBIDDEN.search(line):
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line_number}:{line.strip()[:120]}"
                )
    assert not offenders, "plugin-specific coupling in generic platform:\n" + "\n".join(
        offenders
    )


def test_workload_contract_is_manifest_driven() -> None:
    workload_files = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (ROOT / "services" / "tool-runner" / "src" / "workloads").glob("*.py")
        )
    )
    assert "from_manifest" in workload_files
    assert "owner_namespace" in workload_files
    assert "resource_class" in workload_files
    assert "job_id_fields" in workload_files
    assert "persist_stamp_hook" in workload_files
