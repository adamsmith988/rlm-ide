"""
GPU Client — کلاینت مرکزی GPU

Thin HTTP client for centralized GPU management via llm-proxy.
Replaces local GPU Manager with remote coordination.

🔄 Flow:
  async with gpu_client.acquire("embed_image") as reservation:
      # 1. POST llm-proxy/gpu/reserve → get slot + eviction list
      # 2. Clean up evicted models locally
      # 3. Execute tool here
      # 4. POST llm-proxy/gpu/release → free slot
      result = await registry.execute(...)
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class GPUClient:
    """
    Thin client for the centralized GPU Manager in llm-proxy.

    کلاینت سبک برای مدیریت مرکزی GPU:
    - رزرو اسلات قبل از اجرای ابزار
    - آزادسازی اسلات بعد از اجرا
    - پاک‌سازی محلی مدل‌های evict شده
    """

    def __init__(self, llm_proxy_url: str):
        self._base_url = llm_proxy_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._cleanup_fn: Optional[Callable[[str], bool]] = None

    def set_cleanup_fn(self, fn: Callable[[str], bool]) -> None:
        """
        Set the function to call for local model cleanup.
        Called when the central GPU manager decides to evict a model.

        Args:
            fn: Function that takes tool_id and returns True if cleanup succeeded
        """
        self._cleanup_fn = fn

    async def start(self) -> None:
        """Initialize HTTP client."""
        from .config import settings

        key = (settings.INTERNAL_API_KEY or "").strip()
        headers = {"X-Internal-Key": key} if key else None
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,   # Reserve can block waiting for semaphore
                write=10.0,
                pool=10.0,
            ),
        )
        logger.info(f"🔗 GPU Client connected to {self._base_url}")

    async def stop(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("🔌 GPU Client disconnected")

    @asynccontextmanager
    async def acquire(self, tool_id: str):
        """
        Acquire GPU resources for tool execution.

        Context manager that handles the full reserve/release cycle:
        1. POST /gpu/reserve → get reservation + eviction list
        2. Clean up evicted models locally
        3. Yield (tool executes)
        4. POST /gpu/release → free slot (always, even on error)

        ⛔ اگه llm-proxy در دسترس نباشه، خطا می‌ده — اجرا بدون GPU management ممنوع
        """
        if not self._client:
            raise RuntimeError(
                "GPU Client not connected — cannot execute tool without GPU management. "
                "Ensure llm-proxy is running and accessible."
            )

        reservation_id = None

        # #region agent log
        import json as _json
        import time as _time
        _t_acq = _time.time()
        try:
            for _path in (
                "/home/fulltime/Desktop/AIv2/.cursor/debug-e29ce2.log",
                "/tmp/aiv2-debug-e29ce2.log",
            ):
                try:
                    with open(_path, "a", encoding="utf-8") as _df:
                        _df.write(
                            _json.dumps(
                                {
                                    "sessionId": "e29ce2",
                                    "hypothesisId": "H-A",
                                    "location": "gpu_client.acquire:before_reserve",
                                    "message": "gpu_reserve_request",
                                    "data": {"tool_id": tool_id},
                                    "timestamp": int(_t_acq * 1000),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except Exception:
                    pass
        except Exception:
            pass
        # #endregion

        try:
            # 1. Reserve GPU slot (may block until semaphore available)
            try:
                resp = await self._client.post(
                    "/gpu/reserve",
                    json={"tool_id": tool_id},
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                raise RuntimeError(
                    f"Cannot reach GPU Manager (llm-proxy) for tool '{tool_id}': {e}. "
                    f"Ensure llm-proxy is running at {self._base_url}"
                ) from e

            # #region agent log
            try:
                for _path in (
                    "/home/fulltime/Desktop/AIv2/.cursor/debug-e29ce2.log",
                    "/tmp/aiv2-debug-e29ce2.log",
                ):
                    try:
                        with open(_path, "a", encoding="utf-8") as _df:
                            _df.write(
                                _json.dumps(
                                    {
                                        "sessionId": "e29ce2",
                                        "hypothesisId": "H-A",
                                        "location": "gpu_client.acquire:after_reserve",
                                        "message": "gpu_reserve_response",
                                        "data": {
                                            "tool_id": tool_id,
                                            "status_code": resp.status_code,
                                            "elapsed_ms": int(
                                                (_time.time() - _t_acq) * 1000
                                            ),
                                        },
                                        "timestamp": int(_time.time() * 1000),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
            except Exception:
                pass
            # #endregion

            if resp.status_code != 200:
                raise RuntimeError(
                    f"GPU reserve failed for tool '{tool_id}': "
                    f"status={resp.status_code}, body={resp.text}"
                )

            data = resp.json()
            reservation_id = data.get("reservation_id")
            evict_tools = data.get("evict_tools", [])

            # 2. Clean up evicted models locally
            if evict_tools and self._cleanup_fn:
                for evict_id in evict_tools:
                    try:
                        self._cleanup_fn(evict_id)
                        logger.info(f"🧹 Local cleanup: {evict_id}")
                    except Exception as e:
                        logger.error(
                            f"Local cleanup failed for {evict_id}: {e}"
                        )

            logger.debug(
                f"🎮 GPU acquired: {tool_id} "
                f"(res={reservation_id[:8] if reservation_id else '?'}..., "
                f"evicted={len(evict_tools)})"
            )

            # 3. Yield — tool executes here
            yield data

        finally:
            # 4. Release GPU slot (always, even on error)
            if reservation_id:
                try:
                    resp = await self._client.post(
                        f"/gpu/release/{reservation_id}"
                    )
                    if resp.status_code != 200:
                        logger.error(
                            f"GPU release failed: {reservation_id[:8]}... "
                            f"({resp.status_code})"
                        )
                    else:
                        logger.debug(
                            f"🔓 GPU released: {tool_id} "
                            f"(res={reservation_id[:8]}...)"
                        )
                except Exception as e:
                    logger.error(f"GPU release error: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get GPU status from central manager."""
        if not self._client:
            return {"ok": False, "error": "GPU Client not connected"}
        try:
            resp = await self._client.get("/gpu/status")
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_gpu_client: Optional[GPUClient] = None


def init_gpu_client(llm_proxy_url: str) -> GPUClient:
    """Initialize the GPU client singleton."""
    global _gpu_client
    if _gpu_client is None:
        _gpu_client = GPUClient(llm_proxy_url)
    return _gpu_client


def get_gpu_client() -> Optional[GPUClient]:
    """Get the GPU client singleton. Returns None if not initialized."""
    return _gpu_client
