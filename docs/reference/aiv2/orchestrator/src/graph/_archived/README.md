# Archived graph chat engines

| File | Note |
|------|------|
| `cline_engine.py` | Former Cline XML hub chat engine. Helpers still used via shim ``graph/cline_engine.py`` by ``native_tools_engine``. Not selected as a chat orchestrator. |

LangGraph ReAct stream body remains in ``graph/streaming.py`` as internal
fallback when ``NATIVE_TOOLS_ENGINE_ENABLED`` is false; the drop-in
``langgraph_stream`` orchestrator is under ``orchestrators/_archived/``.
