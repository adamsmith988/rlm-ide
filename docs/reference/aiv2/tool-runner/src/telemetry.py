"""Optional OpenTelemetry bootstrap with no prompt/body capture."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
_configured = False


def configure_telemetry(app: Any, service_name: str) -> bool:
    global _configured
    if _configured:
        return True
    endpoint = str(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "deployment.environment": os.environ.get("ENVIRONMENT", "production"),
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
            )
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()

        @app.middleware("http")
        async def _trace_ids(request: Any, call_next: Any) -> Any:
            span = trace.get_current_span()
            run_id = request.headers.get("x-run-id") or request.query_params.get("run_id")
            user_id = request.headers.get("x-user-id")
            if run_id:
                span.set_attribute("aiv2.run_id", str(run_id)[:160])
            if user_id:
                span.set_attribute("aiv2.user_id", str(user_id)[:160])
            return await call_next(request)

        _configured = True
        return True
    except Exception:
        logger.warning("OpenTelemetry bootstrap failed", exc_info=True)
        return False
