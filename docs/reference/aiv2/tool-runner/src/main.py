"""
Tool Runner Service
سرویس اجرای ابزارهای ML - استفاده از ماژول tools اصلی بدون تغییر
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .utils.registry_wrapper import init_registry, get_registry
from .routes import health_router, internal_router, jobs_router, plugins_router, tools_router
from .gpu_client import init_gpu_client, get_gpu_client

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes and cleans up resources.
    """
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    from .config.settings import assert_secure_secrets
    assert_secure_secrets()
    logger.info(f"📁 Tools path: {settings.TOOLS_MODULE_PATH}")
    
    # Initialize the tool registry
    registry = init_registry(app=app)
    
    if registry.is_ready:
        tools = registry.list_tools()
        logger.info(f"✅ Loaded {len(tools)} tools")
        
        # Log categories
        categories = {}
        for tool in tools.values():
            cat = tool.category or "general"
            categories[cat] = categories.get(cat, 0) + 1
        logger.info(f"📊 Categories: {categories}")

        # Register tool permissions in the dynamic catalog so new tools appear
        # in the admin Permissions Matrix automatically (best-effort).
        try:
            from .catalog_sync import sync_tool_catalog

            await sync_tool_catalog(tools)
        except Exception as exc:
            logger.warning("Tool catalog sync error: %s", exc)

        # Register per-user plugin preference namespaces (dynamic tables in Storage).
        try:
            from .prefs_sync import sync_user_pref_namespaces

            await sync_user_pref_namespaces(settings.PLUGINS_MODULE_PATH)
        except Exception as exc:
            logger.warning("Plugin prefs sync error: %s", exc)
    else:
        logger.warning("⚠️ Registry failed to initialize - running in degraded mode")
    
    # Initialize GPU Client (connects to centralized GPU Manager in llm-proxy)
    def _cleanup_tool(tool_id: str) -> bool:
        """Callback for GPU client to cleanup a specific tool's model locally."""
        try:
            reg = get_registry()
            if reg.is_ready:
                return reg.cleanup_tool(tool_id)
        except Exception as e:
            logger.error(f"Tool cleanup failed for {tool_id}: {e}")
        return False

    gpu_client = init_gpu_client(llm_proxy_url=settings.LLM_PROXY_URL)
    gpu_client.set_cleanup_fn(_cleanup_tool)
    await gpu_client.start()
    logger.info("🔗 GPU Client connected to llm-proxy")

    # Inject registry + storage bridge so tools like text_search / image_search work
    try:
        from modules.tools.storage_http_bridge import StorageHttpBridge

        app.state.storage = StorageHttpBridge(settings.STORAGE_URL)
        app.state.registry = get_registry().tool_registry
        logger.info("✅ app.state.storage (HTTP Qdrant) + app.state.registry injected")
    except Exception as e:
        logger.warning("⚠️ Could not inject storage/registry on app.state: %s", e)
    if (settings.TOOL_RUNNER_INTERNAL_SECRET or "").strip():
        logger.info("🔐 Internal LLM gateway: /internal/llm/* requires X-Tool-Runner-Internal-Secret")
    else:
        logger.warning(
            "⚠️ Internal LLM gateway: TOOL_RUNNER_INTERNAL_SECRET unset — only 127.0.0.1 allowed; "
            "do not publish tool-runner port publicly"
        )

    from .job_bus import JobBus, redis_url_from_env
    from .job_ledger import get_job_ledger
    from .job_runtime import get_job_runtime
    from .workloads import workload_registry

    runtime = get_job_runtime()
    job_bus = JobBus(redis_url_from_env(), cancel_callback=runtime.cancel)
    try:
        await job_bus.start()
    except Exception as exc:
        logger.warning("Job bus unavailable; using process-local events: %s", exc)
    runtime.publisher = job_bus.publish
    app.state.job_bus = job_bus

    job_ledger = get_job_ledger()
    app.state.job_ledger = job_ledger
    if job_ledger.configured:
        await job_ledger.start()
        if job_ledger.ready:
            logger.info("🧾 PostgreSQL job ledger ready")
    elif settings.TOOL_JOB_LEDGER_REQUIRED:
        logger.error("PostgreSQL job ledger is required but no DSN is configured")

    from .job_admission import get_admission
    from .job_queue import Role, get_role, start_queue_supervisor
    from .routes.tools import run_queued_job
    from .capacity_poller import start_capacity_poller, stop_capacity_poller

    role = get_role()
    supervisor = await start_queue_supervisor(
        run_queued_job,
        has_capacity=get_admission().has_any_capacity,
    )
    if supervisor is None:
        if role.enqueues:
            logger.error(
                "🚫 TOOL_RUNNER_ROLE=api without Redis; workload submissions return 503"
            )
        elif role is not Role.ALL:
            logger.warning("TOOL_RUNNER_ROLE=%s without Redis", role.value)
        else:
            logger.info("🧩 tool-runner role=all (process-local workloads)")
    else:
        logger.info(
            "🧩 tool-runner role=%s queue supervisor starting",
            role.value,
        )
    await workload_registry.startup()
    await start_capacity_poller()

    from .event_loop_monitor import get_event_loop_monitor

    loop_monitor = get_event_loop_monitor()
    loop_monitor.start()

    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")
    await stop_capacity_poller()
    await loop_monitor.stop()

    try:
        from .job_queue import shutdown_queue

        await shutdown_queue()
    except Exception as e:
        logger.error(f"job queue worker stop error: {e}")

    try:
        get_admission().begin_drain()
        drain_report = await runtime.drain(
            wait_seconds=float(settings.TOOL_JOB_DRAIN_TIMEOUT_SEC),
            checkpoint_seconds=float(settings.TOOL_JOB_DRAIN_CHECKPOINT_TIMEOUT_SEC),
            cancel_seconds=float(settings.TOOL_JOB_DRAIN_CANCEL_TIMEOUT_SEC),
        )
        logger.info("🧯 job drain complete: %s", drain_report)
    except Exception as e:
        logger.error(f"job drain error: {e}")

    try:
        await workload_registry.shutdown()
    except Exception as e:
        logger.error(f"workload shutdown hook error: {e}")

    try:
        from .job_queue import close_queue
        await close_queue()
    except Exception as e:
        logger.error(f"job queue close error: {e}")

    runtime.publisher = None
    await job_bus.stop()

    try:
        await job_ledger.close()
    except Exception as e:
        logger.error(f"job ledger close error: {e}")

    # Stop GPU client
    try:
        client = get_gpu_client()
        if client:
            await client.stop()
            logger.info("🔌 GPU Client disconnected")
    except Exception as e:
        logger.error(f"GPU Client cleanup error: {e}")

    try:
        registry = get_registry()
        if registry.is_ready:
            registry.cleanup_resources()
            logger.info("🧹 Resources cleaned up")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

    try:
        from .authz import aclose_http_client

        await aclose_http_client()
    except Exception as e:
        logger.error(f"Authz HTTP client close error: {e}")



# Create FastAPI app
app = FastAPI(
    title="AI Platform - Tool Runner",
    description="سرویس اجرای ابزارهای ML با استفاده از ماژول tools اصلی",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(tools_router)
app.include_router(plugins_router)
app.include_router(internal_router)
app.include_router(jobs_router)

from .telemetry import configure_telemetry

configure_telemetry(app, settings.SERVICE_NAME)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "docs": "/docs",
        "health": "/health",
        "tools": "/tools",
        "plugins": "/plugins",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
