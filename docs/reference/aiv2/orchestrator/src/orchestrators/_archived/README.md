# Archived chat orchestrators (not discovered by the registry)

Folders here are **not** loaded by ``OrchestratorRegistry`` (names under
``_archived`` / leading ``_`` are skipped).

| Folder | Former role |
|--------|-------------|
| `cline_xml/` | Drop-in chat_engine: Cline-style XML tool protocol |
| `langgraph_stream/` | Drop-in chat_engine: legacy LangGraph ReAct stream |

Active hub chat engines remain: ``native_tools``, ``rlm``.

Related engine implementation for Cline lives in
``src/graph/_archived/cline_engine.py`` (helpers still re-exported via
``src/graph/cline_engine.py`` shim for ``native_tools_engine`` only).
