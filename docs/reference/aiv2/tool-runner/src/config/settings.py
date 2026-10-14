"""
Tool Runner Service Configuration
تنظیمات سرویس اجرای ابزارها
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration settings."""
    
    # Service
    SERVICE_NAME: str = "tool-runner"
    SERVICE_VERSION: str = "2.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = False
    
    # Tools Module Path (original AI tools)
    TOOLS_MODULE_PATH: str = "/app/modules/tools"
    # Optional: plugin bundles (plugin.json + optional tool.json), same discovery as tools/
    PLUGINS_MODULE_PATH: str = "/app/modules/plugins"
    TOOLS_MANIFEST_PATH: Optional[str] = None
    
    # LLM Proxy — فقط tool-runner (دروازهٔ داخلی) مستقیم به آن وصل می‌شود؛ ابزارها نه.
    LLM_PROXY_URL: str = "http://llm-proxy:8005"
    # فوروارد HTTP به llm-proxy از /internal/llm/*
    LLM_FORWARD_TIMEOUT_SEC: float = 540.0
    LLM_EMBED_FORWARD_TIMEOUT_SEC: float = 120.0
    LLM_IMAGE_FORWARD_TIMEOUT_SEC: float = 600.0
    # Infra reconnect retries only (connect refused / TCP blip) — NOT queue retries.
    # Queue wait is handled by llm-proxy ConcurrencyLimiter + LLM_FORWARD_TIMEOUT_SEC.
    LLM_INFRA_RETRY_MAX_ATTEMPTS: int = 3
    LLM_INFRA_RETRY_DELAY_SEC: float = 1.0
    LLM_VALIDATION_CACHE_TTL_SEC: float = 120.0
    # اگر خالی باشد: فقط loopback به /internal/llm/* مجاز است
    TOOL_RUNNER_INTERNAL_SECRET: Optional[str] = None

    # Central RBAC enforcement for tool access
    AUTH_SERVICE_URL: str = "http://auth-service:8003"
    INTERNAL_API_KEY: str = "aiv2-internal-service-key-change-in-production"
    TOOL_RUNNER_RBAC_ENABLED: bool = True
    TOOL_RUNNER_RBAC_FAIL_OPEN: bool = False
    TOOL_RUNNER_RBAC_FINE_GRAINED: bool = True
    TOOL_RUNNER_RBAC_TIMEOUT_SEC: float = 5.0
    TOOL_RUNNER_TRUSTED_SERVICES: str = "api-gateway,case-importer,orchestrator,llm-gateway"
    # When a trusted service calls on behalf of a user (X-User-Id present),
    # enforce that user's effective RBAC permissions instead of trusting the
    # internal hop blindly. Set false to restore legacy bypass behaviour.
    TOOL_RUNNER_RBAC_IMPERSONATION: bool = True
    # Short TTL cache for impersonated effective-permission lookups
    # (keyed by user|roles|group). Only successful auth-service replies are
    # cached; misses still fail closed. Set <= 0 to disable.
    TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC: float = 10.0
    TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_MAX_ENTRIES: int = 4096
    # Permission/section strings (configurable; must match auth-service catalog).
    TOOLS_VIEW_PERMISSION: str = "tools:view"
    TOOLS_EXECUTE_PERMISSION: str = "tools:execute"
    TOOLS_SECTION_ID: str = "tools-panel"
    
    # Storage service for persistence
    STORAGE_URL: str = "http://storage:8004"
    
    # Upload/Data directories
    UPLOAD_DIR: str = "/data/uploads"
    MODELS_DIR: str = "/data/models"
    
    # Tool execution settings
    TOOL_TIMEOUT_DEFAULT: int = 120
    # Generic detached-job execution and drain budgets.
    TOOL_JOB_MAX_WEIGHT: int = 64
    TOOL_JOB_MAX_WEIGHT_PER_TENANT: int = 48
    TOOL_JOB_DRAIN_TIMEOUT_SEC: float = 450.0
    TOOL_JOB_DRAIN_CHECKPOINT_TIMEOUT_SEC: float = 60.0
    TOOL_JOB_DRAIN_CANCEL_TIMEOUT_SEC: float = 20.0
    # Process role, single image:
    #   all    — today's behaviour: serve HTTP and execute heavy tools
    #            in-process (default; safe for the single-pod deployment).
    #            With TOOL_JOB_REDIS_URL it also consumes jobs (api + worker).
    #   api    — serve HTTP; heavy tools are *enqueued* to Redis and never
    #            executed here (requires TOOL_JOB_REDIS_URL). Light tools run inline.
    #   worker — serve generic job control endpoints and consume queued jobs.
    TOOL_RUNNER_ROLE: str = "all"
    # Redis and PostgreSQL backing for generic detached jobs.
    TOOL_JOB_REDIS_URL: Optional[str] = None
    TOOL_JOB_LEDGER_DSN: Optional[str] = None
    TOOL_JOB_LEDGER_REQUIRED: bool = False
    TOOL_JOB_LEDGER_LEASE_SEC: int = 90
    TOOL_JOB_LEDGER_POOL_SIZE: int = 6
    TOOL_JOB_QUEUE_KEY: str = "tool_runner:jobs:queue"
    TOOL_JOB_OWNER_TTL_SEC: int = 90
    TOOL_JOB_OWNER_HEARTBEAT_SEC: float = 30.0
    TOOL_JOB_OWNER_LINGER_SEC: int = 900
    TOOL_JOB_POD_IDENTITY: Optional[str] = None
    TOOL_JOB_WORKER_POLL_SEC: int = 5
    TOOL_JOB_WORKER_BUSY_BACKOFF_SEC: float = 1.0
    TOOL_JOB_QUEUE_GROUP: str = "tool-runner"
    TOOL_JOB_QUEUE_MAX_ATTEMPTS: int = 5
    TOOL_JOB_QUEUE_CLAIM_IDLE_MS: int = 60000
    TOOL_JOB_QUEUE_RECLAIM_INTERVAL_SEC: float = 10.0
    TOOL_JOB_QUEUE_STREAM_MAXLEN: int = 100000
    TOOL_JOB_REDIS_RECONNECT_INITIAL_SEC: float = 0.5
    TOOL_JOB_REDIS_RECONNECT_MAX_SEC: float = 30.0
    TOOL_JOB_REDIS_RECONNECT_JITTER: float = 0.2
    TOOL_JOB_REDIS_SUPERVISOR_PING_SEC: float = 5.0
    TOOL_JOB_RECORD_TTL_SEC: int = 86400
    TOOL_JOB_ADMISSION_LEASE_MS: int = 60000
    TOOL_JOB_CLUSTER_ADMISSION_ENABLED: bool = True
    # Per resource_class caps for sync/async admission (JSON). Bootstrap only —
    # live limits are overwritten from llm-proxy GET /capacity.resource_limits.
    TOOL_JOB_RESOURCE_LIMITS: str = '{"llm":48,"cpu":16,"gpu":4}'
    TOOL_JOB_WORKER_RESOURCE_CLASSES: str = "default,cpu,llm,gpu"
    TOOL_JOB_WORKER_CAPABILITIES: str = ""
    # Follow llm-proxy node-derived capacity (process admission only).
    CAPACITY_POLL_ENABLED: bool = True
    CAPACITY_POLL_INTERVAL_SEC: float = 10.0
    
    # Resource cleanup
    CLEANUP_ON_IDLE: bool = True
    IDLE_TIMEOUT_SEC: int = 300  # 5 minutes

    # GPU management is now centralized in llm-proxy
    # tool-runner uses LLM_PROXY_URL to connect to the GPU Manager

    # CORS
    CORS_ORIGINS: str = "*"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()


DEFAULT_INTERNAL_API_KEY = "aiv2-internal-service-key-change-in-production"


def assert_secure_secrets() -> None:
    """Fail startup if security-critical secrets are left at their known defaults.

    Override only for local development by setting ALLOW_INSECURE_DEFAULTS=true.
    """
    if os.getenv("ALLOW_INSECURE_DEFAULTS", "").lower() in ("1", "true", "yes"):
        return

    insecure = []
    if settings.INTERNAL_API_KEY == DEFAULT_INTERNAL_API_KEY:
        insecure.append("INTERNAL_API_KEY")
    if insecure:
        raise RuntimeError(
            "Refusing to start with default secret(s): "
            + ", ".join(insecure)
            + ". Set them via environment variables, or set "
            "ALLOW_INSECURE_DEFAULTS=true for local development."
        )
