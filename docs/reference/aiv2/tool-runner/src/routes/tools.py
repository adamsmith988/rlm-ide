"""
Tool Execution Endpoints
اجرای ابزارها
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Mapping, Optional, Sequence
import asyncio
import logging
import time
import os
import shutil
from dataclasses import asdict

from ..config import settings
from ..authz import authorize_tool_execute, authorize_tool_view, filter_executable_tool_ids
from ..utils import get_registry
from ..gpu_client import get_gpu_client
from ..job_admission import AdmissionDraining, get_admission
from ..job_queue import Job, TerminalOutcome, get_queue, get_role, start_owner_heartbeat
from ..job_runtime import AsyncTaskHandle, ProcessGroupHandle, get_job_runtime
from ..workloads import WorkloadSpec, extract_job_id, workload_registry
from ..tool_resource import resolve_tool_resource_from_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])

_OWNER_HEARTBEAT_TASKS: Dict[str, "asyncio.Task[Any]"] = {}


def _runner_should_acquire_gpu(adapter: Any) -> bool:
    """Mirror ToolAdapter.runner_should_acquire_gpu without importing modules.tools at load time."""
    if adapter is None:
        return True
    exec_cfg = (getattr(adapter, "meta", None) or {}).get("execution") or {}
    mode = str(exec_cfg.get("mode", "sync")).lower()
    gpu_at = str(exec_cfg.get("gpu_acquire", "runner")).lower()
    if mode == "deferred" or gpu_at in ("worker", "none"):
        return False
    vram_mb = int(getattr(adapter, "vram_mb", 0) or 0)
    backend = str(getattr(adapter, "backend", "local") or "local").lower()
    if vram_mb <= 0 or backend in ("triton", "vllm", "cpu", "worker"):
        return False
    return True


async def _execute_with_admission(
    *,
    tool_id: str,
    adapter: Any,
    tenant_id: str,
    run_body,
) -> Any:
    """Queue sync tool processes via JobAdmission (tool-agnostic resource_class).

    LLM capacity remains in llm-proxy AIMD; this only bounds concurrent tool
    process slots in tool-runner. See case-importer concurrency_contract.
    """
    import uuid as _uuid

    admission = get_admission()
    resource_class, resource_weight = resolve_tool_resource_from_adapter(adapter)
    job_id = f"sync-{tool_id}-{_uuid.uuid4().hex[:16]}"
    ticket = await admission.submit(
        job_id,
        owner_namespace="tool-runner",
        tenant_id=str(tenant_id or "anonymous"),
        resource_class=resource_class,
        resource_weight=resource_weight,
        metadata={"tool_id": tool_id, "mode": "sync_execute"},
    )
    try:
        await admission.wait(ticket)
        return await run_body()
    finally:
        await admission.release(ticket)


# ============================================================================
# Request/Response Models
# ============================================================================

class ExecuteRequest(BaseModel):
    """Request to execute a tool."""
    args: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    validate_args: bool = True


class ExecuteResponse(BaseModel):
    """Tool execution response."""
    ok: bool
    tool_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PersistRequest(BaseModel):
    """Request to persist tool result."""
    result: Dict[str, Any]
    folder_id: Optional[str] = None


class ToolInfoResponse(BaseModel):
    """Tool information response."""
    id: str
    description: str
    category: str
    capabilities: List[str]
    args: Dict[str, Any]
    parameters: Dict[str, Any]
    enabled: bool
    version: Optional[str]
    timeout_sec: Optional[int]
    supports_persist: bool
    has_ui: bool


class ValidateRequest(BaseModel):
    """Request to validate tool arguments."""
    args: Dict[str, Any]


def _accepted(tool_id: str, job_id: str, status: str, **extra: Any) -> ExecuteResponse:
    data = {
        "job_id": job_id,
        "tool_id": tool_id,
        "status": status,
        "accepted": True,
        **extra,
    }
    return ExecuteResponse(ok=True, tool_id=tool_id, result={"ok": True, "data": data})


def _terminal_outcome(job_id: str) -> TerminalOutcome:
    status, fields = get_job_runtime().terminal_outcome(job_id)
    return TerminalOutcome(status, fields)


async def _execute_process(job: Job, spec: WorkloadSpec) -> Dict[str, Any]:
    args = job.args
    argv = args.get("argv")
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError("subprocess workload requires non-empty argv")
    argv = [str(value) for value in argv]
    executable = shutil.which(argv[0]) or os.path.realpath(argv[0])
    allowed = {
        shutil.which(value) or os.path.realpath(value)
        for value in spec.sandbox.executable_allowlist
    }
    if executable not in allowed:
        raise ValueError("subprocess executable is not allowed by workload spec")

    from modules.tools.run_workspace import workspace_dir

    workspace = workspace_dir(
        job.owner_namespace,
        job.tenant_id,
        job.record_id or "_none",
        job.job_id,
    )
    cwd_value = str(args.get("cwd") or ".")
    if os.path.isabs(cwd_value):
        raise ValueError("subprocess cwd must be workspace-relative")
    cwd = os.path.realpath(os.path.join(workspace, cwd_value))
    raw_env = args.get("env") if isinstance(args.get("env"), Mapping) else {}
    env = {
        key: str(raw_env.get(key, os.environ.get(key, "")))
        for key in spec.sandbox.env_allowlist
        if raw_env.get(key) is not None or os.environ.get(key) is not None
    }
    handle = await ProcessGroupHandle.spawn(
        argv,
        cwd=cwd,
        workspace_root=workspace,
        env=env,
        env_allowlist=spec.sandbox.env_allowlist,
        output_limit_bytes=spec.sandbox.output_limit_bytes,
        timeout_seconds=spec.hard_timeout_sec,
    )
    get_job_runtime().attach(job.job_id, handle)
    return await handle.wait()


async def _start_local(job: Job, spec: WorkloadSpec) -> ExecuteResponse:
    runtime = get_job_runtime()
    existing = runtime.get(job.job_id)
    if existing and existing["status"] not in runtime.TERMINAL:
        return _accepted(job.tool_id, job.job_id, existing["status"], already_running=True)
    admission = get_admission()
    try:
        ticket = await admission.submit(
            job.job_id,
            owner_namespace=job.owner_namespace,
            tenant_id=job.tenant_id,
            resource_class=job.resource_class,
            resource_weight=job.resource_weight,
            metadata=job.metadata,
        )
    except AdmissionDraining as exc:
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": "5"}
        ) from exc
    runtime.create(
        job_id=job.job_id,
        tool_id=job.tool_id,
        owner_namespace=job.owner_namespace,
        tenant_id=job.tenant_id,
        record_id=job.record_id,
        resource_class=job.resource_class,
        status="queued" if ticket.queued else "running",
        queue_position=ticket.position,
    )

    async def worker() -> None:
        try:
            if ticket.queued:
                await admission.wait(ticket)
            runtime.update(job.job_id, status="running", queue_position=0, started_at=time.time())
            if spec.backend in {"process", "subprocess"}:
                result = await _execute_process(job, spec)
            else:
                registry = get_registry()
                adapter = registry.tool_registry.get_adapter(job.tool_id)
                gpu = get_gpu_client()
                if gpu is not None and _runner_should_acquire_gpu(adapter):
                    async with gpu.acquire(job.tool_id):
                        result = await registry.execute(
                            tool_id=job.tool_id, args=job.args, validate=job.validate_args
                        )
                else:
                    result = await registry.execute(
                        tool_id=job.tool_id, args=job.args, validate=job.validate_args
                    )
            if runtime.get(job.job_id).get("status") == "cancelled":
                return
            ok = (
                bool((result or {}).get("ok", True))
                and not (result or {}).get("error")
                and int((result or {}).get("returncode", 0)) == 0
            )
            runtime.update(
                job.job_id,
                status="completed" if ok else "failed",
                result=result,
                error=None if ok else str((result or {}).get("error") or "workload_failed"),
            )
        except asyncio.CancelledError:
            runtime.update(job.job_id, status="cancelled", error="cancelled")
            raise
        except Exception as exc:
            logger.exception("workload failed job_id=%s", job.job_id)
            runtime.update(job.job_id, status="failed", error=str(exc))
        finally:
            await admission.release(ticket)

    task = asyncio.create_task(worker(), name=f"workload:{job.tool_id}:{job.job_id}")
    registration = workload_registry.get(job.tool_id)
    runtime.attach(
        job.job_id,
        AsyncTaskHandle(
            task,
            checkpoint_fn=(
                lambda: workload_registry.checkpoint(job.tool_id, job.job_id)
                if registration and registration.checkpoint_hook
                else None
            ),
            cancel_fn=(
                lambda: workload_registry.cancel(job.tool_id, job.job_id)
                if registration and registration.cancel_hook
                else None
            ),
        ),
        mark_running=not ticket.queued,
    )

    async def affinity_records(_job_id: str) -> Sequence[str]:
        values = [job.record_id] if job.record_id else []
        status = await workload_registry.status(job.tool_id, job.job_id)
        if status:
            values.extend(
                str(status.get(field) or "").strip()
                for field in spec.affinity.dynamic_id_fields
            )
        return tuple(dict.fromkeys(value for value in values if value))

    heartbeat = start_owner_heartbeat(
        job,
        alive=lambda: runtime.alive(job.job_id),
        terminal_source=_terminal_outcome,
        record_source=affinity_records,
    )
    if heartbeat is not None:
        _OWNER_HEARTBEAT_TASKS[job.job_id] = heartbeat
        heartbeat.add_done_callback(
            lambda _task, jid=job.job_id: _OWNER_HEARTBEAT_TASKS.pop(jid, None)
        )
    return _accepted(
        job.tool_id,
        job.job_id,
        "queued" if ticket.queued else "accepted",
        queue_position=ticket.position,
    )


async def submit_workload(
    *,
    spec: WorkloadSpec,
    args: Dict[str, Any],
    validate_args: bool,
) -> ExecuteResponse:
    """Single submission path for every declared workload execution mode."""
    job_id = extract_job_id(spec.tool_id, args)
    args[spec.job_id_fields[0]] = job_id
    tenant_id = str(args.get("user_id") or args.get("tenant_id") or "anonymous")
    record_id = next(
        (str(args.get(field)) for field in spec.affinity.id_fields if args.get(field)),
        None,
    )
    job = Job(
        job_id=job_id,
        tool_id=spec.tool_id,
        owner_namespace=spec.owner_namespace,
        tenant_id=tenant_id,
        record_id=record_id,
        metadata={
            "execution_mode": spec.execution_mode,
            "required_capability": spec.sandbox.required_capability,
        },
        resource_class=spec.resource_class,
        resource_weight=max(1, int(spec.weight)),
        args=args,
        validate_args=validate_args,
    )
    if get_role().enqueues:
        queue = get_queue()
        if queue is None:
            raise HTTPException(
                status_code=503,
                detail="job_queue_unavailable",
                headers={"Retry-After": "5"},
            )
        job.enqueued_by = queue.pod
        position = await queue.enqueue(job)
        return _accepted(
            spec.tool_id, job_id, "queued", queue_position=position, dispatch="redis"
        )
    return await _start_local(job, spec)


async def run_queued_job(job: Job) -> None:
    spec = workload_registry.spec_for(job.tool_id)
    if spec is None:
        raise ValueError(f"workload not registered: {job.tool_id}")
    await _start_local(job, spec)


class ValidateResponse(BaseModel):
    """Validation response."""
    valid: bool
    validated_args: Dict[str, Any]
    errors: List[str]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def list_tools(
    request: Request,
    channel: Optional[str] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all available tools.
    
    Args:
        channel: Filter by channel (llm, ui, embed, rawdata, metadata)
        category: Filter by category (image, audio, text, file, etc.)
    
    Returns:
        Dictionary with tool IDs and basic info
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    tools = registry.list_tools()
    allowed_tool_ids = await filter_executable_tool_ids(request, tools.keys())
    
    # Apply filters
    filtered = {}
    for tool_id, info in tools.items():
        if tool_id not in allowed_tool_ids:
            continue
        # Channel filter
        if channel:
            tool_channels = getattr(info, 'channels', None) or []
            # Empty channels = all channels allowed (backward compatibility)
            if tool_channels and channel not in tool_channels:
                continue
        
        # Category filter
        if category and (info.category or "").lower() != category.lower():
            continue
        
        extra: Dict[str, Any] = {}
        raw_registry = getattr(registry, "tool_registry", None)
        adapter = getattr(raw_registry, "_tools", {}).get(tool_id) if raw_registry else None
        if adapter is not None:
            extra["backend"] = getattr(adapter, "backend", None)
            meta = getattr(adapter, "meta", None) or {}
            if isinstance(meta, dict) and meta.get("llm_profile"):
                extra["llm_profile"] = meta["llm_profile"]

        filtered[tool_id] = {
            "description": info.description,
            "category": info.category,
            "enabled": info.enabled,
            "version": info.version,
            **extra,
        }
    
    return {
        "count": len(filtered),
        "tools": filtered,
        "filters": {
            "channel": channel,
            "category": category,
        }
    }


@router.get("/manifest")
async def get_manifest(request: Request, channel: str = None) -> List[Dict[str, Any]]:
    """
    Get categorized tool manifest for orchestrator.
    
    Args:
        channel: فیلتر بر اساس کانال (llm, ui, api, embed, storage)
    
    Returns:
        List of tool definitions with name, description, arguments, category
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    manifest = registry.get_manifest(channel=channel)
    tool_ids = [item.get("name") or item.get("id") for item in manifest if isinstance(item, dict)]
    allowed_tool_ids = await filter_executable_tool_ids(request, [tid for tid in tool_ids if tid])
    return [
        item
        for item in manifest
        if isinstance(item, dict) and (item.get("name") or item.get("id")) in allowed_tool_ids
    ]


@router.get("/by-file-requirement")
async def get_tools_by_file_requirement(request: Request, channel: str = None) -> Dict[str, Any]:
    """
    Get tools categorized by file requirement.
    
    Args:
        channel: فیلتر بر اساس کانال (llm, ui, api, embed, storage)
    
    دسته‌بندی ابزارها بر اساس نیاز به فایل:
    - file_required: ابزارهایی که حتماً نیاز به فایل دارن
    - no_file_required: ابزارهایی که فایل نمی‌خوان (مثل text_search)
    - file_optional: ابزارهایی که فایل اختیاریه
    
    Returns:
        {"file_required": [...], "no_file_required": [...], "file_optional": [...]}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return registry.get_tools_by_file_requirement(channel=channel)


@router.get("/prompt-info")
async def get_prompt_friendly_info(request: Request, channel: str = "llm") -> Dict[str, str]:
    """
    Get dynamic prompt-friendly tools description.
    
    Args:
        channel: فیلتر بر اساس کانال (default: llm)
    
    برای استفاده در system prompt - داینامیک از tool.json ها تولید میشه.
    
    Returns:
        {"tools_info": "markdown formatted tools description"}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return {"tools_info": registry.get_prompt_friendly_tools_info(channel=channel)}


@router.get("/agent-prompt/{category}")
async def get_agent_prompt(request: Request, category: str, channel: str = "llm") -> Dict[str, Any]:
    """
    Get dynamic agent prompt for a specific category.
    
    این endpoint یه prompt آماده برای agent برمی‌گردونه که از tool.json های
    مربوط به اون category ساخته شده.
    
    Args:
        category: دسته‌بندی (image, audio, text, file, video, face, embed, entity, general)
        channel: کانال (default: llm)
    
    Returns:
        {
            "role": "Image Analysis Specialist",
            "capabilities": ["..."],
            "tools": [{"name": "...", "description": "..."}],
            "prompt": "## Role: ... (full markdown prompt)"
        }
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return registry.get_agent_prompt_for_category(category, channel)


@router.get("/agent-prompts")
async def get_all_agent_prompts(request: Request, channel: str = "llm") -> Dict[str, Dict[str, Any]]:
    """
    Get dynamic prompts for ALL agent types.
    
    برای cache کردن در orchestrator در startup.
    
    Args:
        channel: کانال (default: llm)
    
    Returns:
        {
            "image": {"role": "...", "prompt": "..."},
            "audio": {"role": "...", "prompt": "..."},
            ...
        }
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return registry.get_all_agent_prompts(channel)


@router.get("/openai")
async def get_openai_tools(request: Request, channel: str = "llm") -> List[Dict[str, Any]]:
    """
    Get OpenAI-compatible tool definitions.
    For use with LLM function calling.
    
    Args:
        channel: فیلتر بر اساس کانال (default: llm برای LLM function calling)
    
    Returns:
        List of tools in OpenAI format (filtered by channel)
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    tools = registry.get_openai_tools(channel=channel)
    tool_ids = []
    for item in tools:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            tool_ids.append(fn["name"])
    allowed_tool_ids = await filter_executable_tool_ids(request, tool_ids)
    return [
        item
        for item in tools
        if isinstance(item.get("function"), dict)
        and item["function"].get("name") in allowed_tool_ids
    ]


@router.get("/schema")
async def get_full_schema(request: Request) -> Dict[str, Any]:
    """
    Get full schema for all tools.
    Includes all metadata, examples, errors, etc.
    
    Returns:
        Complete tool schema dictionary
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    schema = registry.get_schema()
    if isinstance(schema, dict):
        tool_keys = schema.get("tools")
        if isinstance(tool_keys, dict):
            allowed_tool_ids = await filter_executable_tool_ids(request, tool_keys.keys())
            schema = dict(schema)
            schema["tools"] = {
                tid: value for tid, value in tool_keys.items() if tid in allowed_tool_ids
            }
            return schema
    await authorize_tool_view(request)
    return schema


@router.get("/agents/manifest")
async def get_multi_agent_manifest(request: Request, channel: str = None) -> Dict[str, Any]:
    """
    Get multi-agent categorized tool manifest.
    
    Args:
        channel: فیلتر بر اساس کانال (llm, ui, api, embed, storage)
    
    دریافت manifest دسته‌بندی شده برای معماری Multi-Agent.
    ابزارها بر اساس نوع agent (vision, text, audio, file, analysis, general) گروه‌بندی می‌شوند.
    
    Returns:
        {
            "agents": {
                "vision": {"description": ..., "tools": [...]},
                "text": {"description": ..., "tools": [...]},
                "audio": {"description": ..., "tools": [...]},
                "file": {"description": ..., "tools": [...]},
                "analysis": {"description": ..., "tools": [...]},
                "general": {"description": ..., "tools": [...]}
            },
            "all_tools": [...],
            "category_mapping": {"tool_name": "agent_type", ...}
        }
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    data = registry.get_multi_agent_manifest(channel=channel)
    all_tools = data.get("all_tools", []) if isinstance(data, dict) else []
    tool_ids = [
        item.get("name") or item.get("id")
        for item in all_tools
        if isinstance(item, dict)
    ]
    allowed_tool_ids = await filter_executable_tool_ids(request, [tid for tid in tool_ids if tid])
    if not isinstance(data, dict):
        return data
    data = dict(data)
    data["all_tools"] = [
        item
        for item in all_tools
        if isinstance(item, dict) and (item.get("name") or item.get("id")) in allowed_tool_ids
    ]
    agents = data.get("agents")
    if isinstance(agents, dict):
        filtered_agents = {}
        for agent_id, agent_data in agents.items():
            if not isinstance(agent_data, dict):
                filtered_agents[agent_id] = agent_data
                continue
            agent_copy = dict(agent_data)
            tools = agent_copy.get("tools", [])
            agent_copy["tools"] = [
                item
                for item in tools
                if isinstance(item, dict) and (item.get("name") or item.get("id")) in allowed_tool_ids
            ]
            filtered_agents[agent_id] = agent_copy
        data["agents"] = filtered_agents
    return data


@router.get("/by-category")
async def get_tools_by_category(
    request: Request,
    category: str = None,
    channel: str = "llm",
    include_general: bool = True
) -> Dict[str, Any]:
    """
    Get tools filtered by category and channel.
    
    مثال: برای فایل عکس → category=image, channel=llm
    
    Args:
        category: دسته‌بندی (image, audio, text, face, file, embed, entity, geo)
        channel: کانال (llm, ui, metadata, rawdata, embed) - default: llm
        include_general: شامل ابزارهای عمومی هم بشه؟ - default: true
    
    Returns:
        {"tools": [...], "count": ..., "category": ..., "channel": ...}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    tools = registry.get_tools_by_category_and_channel(
        category=category,
        channel=channel,
        include_general=include_general
    )
    tool_ids = [item.get("name") or item.get("id") for item in tools if isinstance(item, dict)]
    allowed_tool_ids = await filter_executable_tool_ids(request, [tid for tid in tool_ids if tid])
    tools = [
        item
        for item in tools
        if isinstance(item, dict) and (item.get("name") or item.get("id")) in allowed_tool_ids
    ]
    
    return {
        "tools": tools,
        "count": len(tools),
        "category": category,
        "channel": channel,
        "include_general": include_general
    }


@router.get("/categories")
async def get_available_categories(request: Request) -> Dict[str, Any]:
    """
    Get list of all available tool categories.
    
    Returns:
        {"categories": ["image", "audio", "text", ...]}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return {"categories": registry.get_available_categories()}


@router.get("/category-map")
async def get_category_map(request: Request, channel: str = "llm") -> Dict[str, Any]:
    """
    Get complete mapping of categories to tool IDs.
    
    برای populate کردن cache در orchestrator و سرویس‌های دیگه.
    
    Args:
        channel: فیلتر کانال (default: llm)
    
    Returns:
        {"image": ["vision_info", "image_faces", ...], ...}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    category_map = registry.get_category_map(channel=channel)
    all_ids = []
    if isinstance(category_map, dict):
        for ids in category_map.values():
            if isinstance(ids, list):
                all_ids.extend([tid for tid in ids if isinstance(tid, str)])
    allowed_tool_ids = await filter_executable_tool_ids(request, all_ids)
    return {
        category: [tid for tid in ids if tid in allowed_tool_ids]
        for category, ids in category_map.items()
        if isinstance(ids, list)
    }


@router.get("/file-based-tools")
async def get_file_based_tools(request: Request, channel: str = "llm") -> Dict[str, Any]:
    """
    Get list of tools that require a file input.
    
    برای populate کردن cache در orchestrator.
    
    Args:
        channel: فیلتر کانال (default: llm)
    
    Returns:
        {"tools": ["vision_info", "audio_transcribe", ...]}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return {"tools": registry.get_file_based_tools(channel=channel)}


@router.get("/map-file-type")
async def map_file_type_to_category(request: Request, file_type: str) -> Dict[str, str]:
    """
    Map file type to tool category.
    
    Args:
        file_type: MIME type (image/jpeg) یا extension (.jpg) یا نوع کلی (image)
    
    Returns:
        {"file_type": "image/jpeg", "category": "image"}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return {
        "file_type": file_type,
        "category": registry.map_file_type_to_category(file_type)
    }


@router.get("/agents/{agent_type}/tools")
async def get_agent_tools(request: Request, agent_type: str) -> Dict[str, Any]:
    """
    Get tools for a specific agent type.
    
    دریافت ابزارهای یک agent خاص.
    
    Args:
        agent_type: نوع agent (vision, text, audio, file, analysis, general)
        
    Returns:
        {"agent_type": ..., "tools": [...], "count": ...}
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    tools = registry.get_agent_tools(agent_type)
    
    if not tools and agent_type not in ["vision", "text", "audio", "file", "analysis", "general"]:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent type '{agent_type}' not found. Valid types: vision, text, audio, file, analysis, general"
        )
    
    return {
        "agent_type": agent_type,
        "tools": tools,
        "count": len(tools),
    }


@router.get("/vram-estimates")
async def get_vram_estimates(request: Request) -> Dict[str, Any]:
    """
    Get VRAM estimates for all tools.
    
    برای استفاده در llm-proxy GPU Manager.
    این endpoint لیست تخمین مصرف VRAM هر ابزار رو برمی‌گردونه.
    
    Returns:
        {
            "tools": {"tool_id": {"vram_mb": 500, "backend": "local"}, ...},
            "cpu_only": ["tool_id1", "tool_id2", ...],
            "default_vram_mb": 500
        }
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    return registry.get_vram_estimates()


@router.get("/capabilities")
async def get_workload_capabilities(request: Request) -> Dict[str, Any]:
    """Return the generic detached-workload capability catalog."""
    await authorize_tool_view(request)
    discovered_ids = tuple(get_registry().list_tools())
    policies: Dict[str, Dict[str, Any]] = {}
    for spec in workload_registry.specs():
        affinity_fields = spec.affinity.id_fields or spec.job_id_fields
        owner_policy = {
            "affinity": {
                "enabled": bool(spec.affinity.selectors),
                "job_id_fields": list(affinity_fields),
            },
            "owner_namespace": spec.owner_namespace,
        }
        policies[spec.tool_id] = {
            "tool_id": spec.tool_id,
            "execution_policy": {
                **owner_policy,
                "mode": spec.execution_mode,
                "heavy": spec.detached,
                "resource_class": spec.resource_class,
                "poll_exempt": False,
            },
        }
        related_ids = set(spec.affinity.tool_ids)
        related_ids.update(
            tool_id
            for tool_id in discovered_ids
            if any(
                tool_id.startswith(prefix)
                for prefix in spec.affinity.tool_prefixes
            )
        )
        for related_id in related_ids:
            # Companion operations may identify ownership either by the parent
            # job/run id (progress, cancel) or by a dynamic child/session id.
            # Advertising only dynamic ids makes run-level polling fall back to
            # the API service and lose process-local live state.
            companion_fields = list(
                dict.fromkeys(
                    (
                        *affinity_fields,
                        *spec.affinity.dynamic_id_fields,
                    )
                )
            )
            policies[related_id] = {
                "tool_id": related_id,
                "execution_policy": {
                    "affinity": {
                        "enabled": True,
                        "session_id_fields": companion_fields,
                    },
                    "owner_namespace": spec.owner_namespace,
                    "mode": "inline",
                    # Long-running RLM attach/continue/refine must outlive the
                    # default REQUEST_TIMEOUT (120s). Poll companions stay light.
                    "heavy": related_id
                    in {
                        "plugin_agent_flow_rlm_attach",
                        "plugin_agent_flow_rlm_refine",
                    },
                    "timeout_sec": 10800
                    if related_id
                    in {
                        "plugin_agent_flow_rlm_attach",
                        "plugin_agent_flow_rlm_refine",
                    }
                    else None,
                    "poll_exempt": True,
                },
            }
    return {
        "execution_modes": ["inline", "detached", "deferred", "queued"],
        "backends": ["local", "process", "subprocess"],
        "features": [
            "status",
            "events",
            "cancel",
            "checkpoint",
            "tenant_admission",
            "owner_affinity",
        ],
        "workloads": [
            {
                "tool_id": spec.tool_id,
                "execution_mode": spec.execution_mode,
                "backend": spec.backend,
                "owner_namespace": spec.owner_namespace,
                "resource_class": spec.resource_class,
            }
            for spec in workload_registry.specs()
        ],
        "policies": policies,
    }


@router.get("/for-filetype")
async def get_tools_for_filetype(
    request: Request,
    mime: str,
    channel: str = None
) -> Dict[str, Any]:
    """
    Get tools suitable for a specific file type.
    
    Args:
        mime: MIME type of the file (e.g., "image/jpeg")
        channel: Filter by channel (optional)
    
    Returns:
        List of tools that can process this file type
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    await authorize_tool_view(request)
    tools = registry.get_tools_for_filetype(mime, channel=channel)
    
    return {
        "mime_type": mime,
        "count": len(tools),
        "tools": tools,
    }


@router.get("/{tool_id}")
async def get_tool_info(request: Request, tool_id: str) -> ToolInfoResponse:
    """
    Get detailed info for a specific tool.
    
    Args:
        tool_id: Tool identifier
        
    Returns:
        Tool information including args schema
    """
    registry = get_registry()
    await authorize_tool_view(request)
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    tool = registry.get_tool(tool_id)
    
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    
    return ToolInfoResponse(**asdict(tool))


@router.post("/{tool_id}/validate")
async def validate_tool_args(tool_id: str, request: ValidateRequest, http_request: Request) -> ValidateResponse:
    """
    Validate arguments for a tool without executing.
    
    Args:
        tool_id: Tool identifier
        request: Arguments to validate
        
    Returns:
        Validation result with coerced args
    """
    registry = get_registry()
    await authorize_tool_execute(http_request, tool_id)
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    validated_args, errors = registry.validate_args(tool_id, request.args)
    
    return ValidateResponse(
        valid=len(errors) == 0,
        validated_args=validated_args,
        errors=errors,
    )


@router.post("/{tool_id}/execute", response_model=ExecuteResponse)
async def execute_tool(tool_id: str, request: ExecuteRequest, http_request: Request) -> ExecuteResponse:
    """
    Execute a tool with provided arguments.
    
    Args:
        tool_id: Tool identifier
        request: Execution request with args
        
    Returns:
        Tool execution result
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    # Check tool exists
    tool = registry.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    
    if not tool.enabled:
        raise HTTPException(status_code=400, detail=f"Tool '{tool_id}' is disabled")

    authz_context = await authorize_tool_execute(http_request, tool_id)
    
    # Add session_id to args if provided
    args = dict(request.args)
    if request.session_id:
        args["session_id"] = request.session_id
    if authz_context.user_id:
        args["user_id"] = authz_context.user_id
    if authz_context.group_id:
        args["group_id"] = authz_context.group_id
    if authz_context.user_roles:
        args["user_roles"] = authz_context.user_roles
    tool_ctx = authz_context.as_tool_context()
    incoming_ctx = args.get("_auth_context")
    if isinstance(incoming_ctx, dict):
        # Preserve case-importer fields (artifact_id) needed for MinIO download middleware.
        for key in ("artifact_id", "storage_path", "service_name"):
            if incoming_ctx.get(key) is not None:
                tool_ctx[key] = incoming_ctx[key]
    # File-explorer / gateway callers pass artifact_id at the top level of args
    # (not nested under _auth_context). Promote so MinIOPathResolver can hydrate
    # the file before tools like file_meta check Path.exists().
    for key in ("artifact_id", "storage_path"):
        if args.get(key) is not None and tool_ctx.get(key) is None:
            tool_ctx[key] = args[key]
    if args.get("case_id") and not tool_ctx.get("case_id"):
        tool_ctx["case_id"] = args["case_id"]
    if args.get("session_id") and not tool_ctx.get("session_id"):
        tool_ctx["session_id"] = args["session_id"]
    if args.get("run_id") and not tool_ctx.get("run_id"):
        tool_ctx["run_id"] = args["run_id"]
    args["_auth_context"] = tool_ctx
    
    logger.info(f"Executing tool: {tool_id}")
    
    gpu_client = get_gpu_client()
    if not gpu_client:
        raise HTTPException(
            status_code=503,
            detail="GPU Client not initialized — cannot execute tools without GPU management. "
                   "Ensure llm-proxy is running.",
        )

    workload = workload_registry.spec_for(tool_id)
    # Only detached/queued specs go through the job queue. Inline workload
    # metadata (timeouts/heavy hints) must not divert execution — otherwise
    # tools like rlm_attach get enqueued, collide on run_id, and silently drop.
    if workload is not None and workload.detached:
        return await submit_workload(
            spec=workload,
            args=args,
            validate_args=request.validate_args,
        )

    adapter = registry.tool_registry.get_adapter(tool_id) if registry.tool_registry else None
    needs_gpu_acquire = _runner_should_acquire_gpu(adapter)
    tenant_id = (
        authz_context.user_id
        or authz_context.group_id
        or str(args.get("user_id") or args.get("case_id") or "anonymous")
    )

    async def _run_body():
        if needs_gpu_acquire:
            async with gpu_client.acquire(tool_id):
                return await registry.execute(
                    tool_id=tool_id,
                    args=args,
                    validate=request.validate_args,
                )
        return await registry.execute(
            tool_id=tool_id,
            args=args,
            validate=request.validate_args,
        )

    try:
        result = await _execute_with_admission(
            tool_id=tool_id,
            adapter=adapter,
            tenant_id=str(tenant_id),
            run_body=_run_body,
        )
    except AdmissionDraining:
        raise HTTPException(status_code=503, detail="tool_runner_draining") from None

    # Build response - check ok=True and no error (None counts as no error)
    if result.get("ok", True) and not result.get("error"):
        return ExecuteResponse(
            ok=True,
            tool_id=tool_id,
            result=result,
        )
    else:
        # Keep the full tool payload (especially channels.llm) so the orchestrator
        # can explain structured failures — e.g. DB row exists but blob is missing.
        return ExecuteResponse(
            ok=False,
            tool_id=tool_id,
            error=result.get("error", "Unknown error"),
            result=result,
        )


@router.post("/{tool_id}/persist")
async def persist_tool_result(tool_id: str, request: PersistRequest, http_request: Request) -> Dict[str, Any]:
    """
    Persist a tool's result to storage via OutputPersistenceService.
    
    Args:
        tool_id: Tool identifier
        request: Result to persist
        
    Returns:
        Persistence result
    """
    registry = get_registry()
    authz_context = await authorize_tool_execute(http_request, tool_id)
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    # Check tool exists
    tool = registry.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")

    has_intents = isinstance(request.result, dict) and isinstance(
        request.result.get("output_intents"), list
    ) and bool(request.result.get("output_intents"))
    if not has_intents and not tool.supports_persist:
        raise HTTPException(
            status_code=400, 
            detail=f"Tool '{tool_id}' does not support persistence"
        )
    
    logger.info(f"Persisting result for tool: {tool_id}")
    auth_ctx = authz_context.as_tool_context()
    
    result = await registry.persist(
        tool_id=tool_id,
        result=request.result,
        folder_id=request.folder_id,
        auth_context=auth_ctx,
        strict=True,
    )
    
    return result


@router.get("/category/{category}")
async def list_by_category(request: Request, category: str) -> Dict[str, Any]:
    """
    List tools filtered by category.
    
    Args:
        category: Category name (image, audio, text, etc.)
        
    Returns:
        Tools in the specified category
    """
    registry = get_registry()
    
    if not registry.is_ready:
        raise HTTPException(status_code=503, detail="Registry not ready")
    
    tools = registry.list_tools()
    
    allowed_tool_ids = await filter_executable_tool_ids(request, tools.keys())
    filtered = {
        tool_id: {
            "description": info.description,
            "category": info.category,
            "enabled": info.enabled,
        }
        for tool_id, info in tools.items()
        if info.category == category.lower() and tool_id in allowed_tool_ids
    }
    
    return {
        "category": category,
        "count": len(filtered),
        "tools": filtered,
    }
