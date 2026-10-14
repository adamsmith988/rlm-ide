"""
Orchestrator Configuration
تنظیمات سرویس ارکستراتور
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service info
    SERVICE_NAME: str = "orchestrator"
    SERVICE_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # LLM gateway (OpenAI-compatible); model ids come from llm_routes bindings.
    LLM_PROXY_URL: str = "http://llm-proxy:8005"
    # When true, chat/embed calls fail if no DB binding resolves (no env model fallback).
    STRICT_LLM_BINDINGS: bool = True
    LLM_TEMPERATURE: float = 0.2
    LLM_TIMEOUT: int = 120
    # SDK-level quick retries for 429/5xx/connection errors.
    LLM_MAX_RETRIES: int = 2
    # After those, a 429 (gateway rate limit) is waited out: Retry-After when
    # sent, else exponential backoff, within a total wait budget.
    LLM_RATE_LIMIT_BACKOFF_SEC: float = 2.0
    LLM_RATE_LIMIT_MAX_BACKOFF_SEC: float = 30.0
    LLM_RATE_LIMIT_MAX_WAIT_SEC: float = 90.0
    # None = omit max_tokens on API requests (use full model/server completion budget).
    # Set a positive int only if you need an explicit cap (e.g. LLM_MAX_TOKENS=8192).
    LLM_MAX_TOKENS: Optional[int] = None
    # Planning synthesizer: needs a larger completion budget than a single ReAct step
    # when many tasks / long aggregated tool outputs (default was effectively 2048 via bug).
    LLM_SYNTHESIS_MAX_TOKENS: int = 8192
    # Send ``chat_template_kwargs.enable_thinking`` when upstream supports a reasoning parser.
    # Set false if the upstream model has no reasoning parser (saves tokens).
    LLM_ENABLE_THINKING: bool = True
    # Upstream tool-call parser id (logging / leak-detection alignment).
    LLM_TOOL_CALL_PARSER: str = "qwen3_coder"
    
    # Tool Runner Service
    TOOL_RUNNER_URL: str = "http://tool-runner:8002"
    INTERNAL_API_KEY: str = "aiv2-internal-service-key-change-in-production"
    # Same secret as tool-runner ``/internal/llm/*`` (catalog / chat courier).
    TOOL_RUNNER_INTERNAL_SECRET: str = ""
    # Tool arguments the host fills from the authenticated chat for
    # kernel-callable engines (RLM); hidden from their tool schemas and
    # never taken from model-written code (comma-separated).
    TOOL_HOST_OWNED_ARGS: str = "session_id,user_id"

    # Auth Service (RBAC) — used to resolve per-user effective permissions so
    # the chat tool list reflects what each user is actually allowed to run.
    AUTH_SERVICE_URL: str = "http://auth-service:8003"
    # When True, tools offered to the LLM are filtered by the user's RBAC
    # permissions (chat:use_tools gate + per-tool tools:execute:{id}).
    TOOL_PERMISSION_ENFORCEMENT: bool = True
    # Cache TTL (seconds) for a user's resolved permission set.
    TOOL_PERMISSION_CACHE_TTL: int = 60
    # On auth-service failure: True = allow all tools (fail-open), False = no tools.
    # Default fail-CLOSED: a permission-service outage must not silently grant a
    # chat-only user the full tool surface.
    TOOL_PERMISSION_FAIL_OPEN: bool = False
    # Permission strings (configurable so a deployment can rename them without
    # touching code). Must match the auth-service permission catalog.
    CHAT_USE_TOOLS_PERMISSION: str = "chat:use_tools"
    TOOLS_EXECUTE_PERMISSION: str = "tools:execute"

    # Storage Service
    STORAGE_URL: str = "http://storage:8004"
    # Page size for walking session messages (storage caps limit at 500).
    STORAGE_MESSAGES_PAGE_SIZE: int = 500
    
    # Orchestrator Settings
    MAX_STEPS: int = 10  # حداکثر گام‌ها (افزایش برای planning)
    MAX_PARALLEL_TOOLS: int = 5

    # Streaming agent (`/chat/stream` → stream_agent LangGraph)
    # - tool_choice: planning executor uses "auto"; matching that avoids forced
    #   tool JSON when file_context exists but inlined text is enough.
    STREAM_TOOL_CHOICE_REQUIRED_WITH_FILE: bool = False
    # Per ToolMessage text cap sent back into the LLM (was hard-coded 8000).
    STREAM_TOOL_RESULT_MAX_CHARS: int = 24_000
    # Continue-style history prune (protect trailing tool chain, drop old turns).
    STREAM_ENABLE_CONTINUE_STYLE_PRUNE: bool = True
    # Approximate char budget for messages sent to the LLM (≈ context × 4 heuristic).
    STREAM_CONTEXT_CHAR_BUDGET: int = 120_000
    # `/chat/stream/full`: False = same ReAct loop as Continue (`stream_agent`);
    # True = legacy behaviour (planning when PLANNING_ENABLED, else agent graph).
    STREAM_CHAT_FULL_USE_PLANNING_AGENT: bool = False
    # POST `/chat` (non-stream): False = same ReAct as Continue (`run_agent` → `stream_agent`);
    # True = planning when PLANNING_ENABLED, else legacy `run_agent_graph`.
    CHAT_USE_PLANNING_AGENT: bool = False

    # Deprecated: Cline XML chat engine archived under
    # ``orchestrators/_archived/cline_xml`` / ``graph/_archived/cline_engine.py``.
    # Kept for env compatibility only — hub chat resolve ignores this flag.
    CLINE_XML_ENGINE_ENABLED: bool = False

    # Native OpenAI ``tool_calls`` streaming engine (graph/native_tools_engine.py).
    # Default hub chat path (alongside optional orchestrator_id=rlm).
    NATIVE_TOOLS_ENGINE_ENABLED: bool = True
    # Hub notice when an engine turn stops before a final answer ({reason} optional).
    HUB_NOTICE_INCOMPLETE: str = "کار ناتمام ماند. وضعیت جلسه حفظ شده؛ می‌توانید ادامه دهید."
    HUB_NOTICE_INCOMPLETE_REASON: str = "کار ناتمام ماند ({reason}). وضعیت جلسه حفظ شده؛ می‌توانید ادامه دهید."
    # Appended last to the native system prompt each turn; {script} = dominant Unicode
    # script of the user's latest message. Empty disables.
    NATIVE_REPLY_LANGUAGE_HINT: str = (
        "## Reply language\nThe user's latest message is written mainly in {script} script. Write your "
        "entire reply in the language of that message, even when tool output, files, memory or earlier "
        "turns use another language."
    )
    # Appended to the model's copy of the current user message (stored history is
    # untouched), next to where generation starts; {script} as above. Empty disables.
    NATIVE_REPLY_LANGUAGE_USER_NOTE: str = "[Reply language: same as this message ({script} script).]"
    NATIVE_REPAIR_QUOTE_MAX_CHARS: int = 600  # latest user message quoted in the empty-answer repair prompt
    # Repair prompts sent after an empty model reply before raw tool output is shown as the answer.
    NATIVE_EMPTY_ANSWER_REPAIR_RETRIES: int = 1
    # The repair round runs without reasoning (the empty reply is usually reasoning that never ended).
    NATIVE_EMPTY_ANSWER_REPAIR_NO_THINKING: bool = True

    # Chat engines are discovered from src/orchestrators/*/manifest.json
    # (extensions.chat_stream + extensions.stream_entry). These pick which one
    # serves hub chat when no orchestrator_id is sent, and which one serves the
    # floating dock / page mini-chat.
    DEFAULT_CHAT_ORCHESTRATOR: str = "native_tools"
    DOCK_CHAT_ORCHESTRATOR: str = "native_tools"
    # Per-page dock engine: "page=engine_id,..." where page is the envelope's
    # ``surface`` or the page's ``active_plugin``. An ``orchestrator_id`` inside
    # the page envelope itself wins over both.
    DOCK_CHAT_ORCHESTRATOR_BY_PAGE: str = ""
    # ChatRequest.context keys that carry a page envelope (page mini-chat / dock),
    # highest priority first. Their presence routes the turn to the dock engine.
    HOST_PAGE_CONTEXT_KEYS: str = "graph_visual_explorer_v1,native_host_context_v1"
    # How long a page plugin's llm_host_prompt (served by the tool-runner) is cached.
    PLUGIN_HOST_PROMPT_TTL_SEC: float = 300.0
    # Turn-intent vocabulary + intent→tool metadata rules; empty = bundled src/tools/tool_intents.json.
    TOOL_INTENTS_FILE: str = ""
    # Durable per-chat-session state for engines that keep it (e.g. RRSI harness
    # history). Empty = TOOL_RUNNER_WORKSPACE_ROOT/orchestrator_state, else /tmp.
    ORCHESTRATOR_STATE_ROOT: str = ""

    # When False (default), session intent is logged only; tool list uses general
    # Continue-style selection (~8–12 tools) instead of narrowing to 1–2 tools.
    SESSION_INTENT_TOOL_FILTER: bool = False

    UNLOAD_AFTER_RUN: bool = False
    REFLECTION_ENABLED: bool = True  # Enable self-evaluation after tool execution
    REFLECTION_MIN_CONFIDENCE: float = 0.7  # Minimum confidence to skip more tools (0-1)
    
    # Planning Agent Settings
    PLANNING_ENABLED: bool = True  # استفاده از معماری Plan-Execute-Revise
    MAX_REPLANS: int = 3  # حداکثر تعداد re-planning
    
    # Chat memory (Qdrant vectors; embeddings via the llm-gateway binding)
    MEMORY_ENABLED: bool = True
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None        # overrides QDRANT_HOST/PORT when set
    QDRANT_API_KEY: Optional[str] = None
    MEMORY_COLLECTION: str = "orchestrator_memory"  # prefix; suffixed by embed model + dim
    MEMORY_BINDING_SERVICE: str = "orchestrator"  # llm_routes consumer for memory models
    MEMORY_EMBED_PURPOSE: str = "embed"     # binding slot for embeddings
    MEMORY_FACT_PURPOSE: str = "chat"       # binding slot for long-term fact distillation
    MEMORY_EMBED_BATCH: int = 32
    MEMORY_EMBED_PRIORITY: str = "interactive"  # X-AIv2-Priority (batch traffic is sheddable)
    MEMORY_EMBED_RETRIES: int = 3               # on 429/503 from the gateway
    MEMORY_EMBED_RETRY_MAX_WAIT_S: float = 5.0
    MEMORY_TIMEOUT_S: float = 20.0
    MEMORY_EMBED_RETRY_STATUSES: str = "429,502,503,504"  # HTTP statuses retried by the embedder
    MEMORY_EMBED_RETRY_BASE_S: float = 0.5        # exponential backoff base when no Retry-After
    MEMORY_SCROLL_PAGE: int = 256                 # Qdrant scroll page size
    MEMORY_INDEXED_FIELDS: str = "user_id,session_id,memory_type"  # keyword payload indexes (each costs open files)
    MEMORY_QDRANT_SEGMENTS: int = 2               # optimizers_config.default_segment_number (Qdrant default = CPU count)

    MEMORY_SESSION_LIMIT: int = 20      # حداکثر حافظه در هر session
    MEMORY_USER_LIMIT: int = 10         # حداکثر حافظه بلندمدت در جستجو
    MEMORY_AUTO_SAVE: bool = True       # ذخیره خودکار مکالمات
    MEMORY_MIN_SCORE: float = 0.45      # cosine floor for recall
    MEMORY_CHUNK_CHARS: int = 1200      # short-term chunk window (long turns are fully indexed)
    MEMORY_CHUNK_OVERLAP: int = 200
    MEMORY_MAX_CHUNKS_PER_TURN: int = 64
    # A turn whose rendered question+answer fits this many chars is stored as one chunk, so a
    # recall hit on the question also returns its answer. 0 → always chunk per role.
    MEMORY_PAIR_TURN_MAX_CHARS: int = 1200
    MEMORY_PAIR_TURN_TEMPLATE: str = "(user) {user}\n(assistant) {assistant}"
    MEMORY_ASSISTANT_MAX_CHARS: int = 4000
    MEMORY_QUERY_MAX_CHARS: int = 4000
    MEMORY_PROMPT_MAX_CHARS: int = 3000
    MEMORY_PROMPT_ITEM_CHARS: int = 600
    MEMORY_FACT_EXTRACTION: bool = True     # long-term facts distilled per turn (background)
    MEMORY_FACT_MAX_INPUT_CHARS: int = 8000
    # User messages longer than this are treated as pasted material: they are still kept as
    # short-term chunks but never distilled into long-term facts. 0 → no limit.
    MEMORY_FACT_MAX_USER_CHARS: int = 4000
    # Added/updated facts whose number-bearing tokens (codes, dates, amounts) do not occur in
    # the user's own message are dropped: such values came from the assistant or a document.
    MEMORY_FACT_REQUIRE_USER_GROUNDING: bool = True
    MEMORY_FACT_MAX_PER_TURN: int = 8
    MEMORY_FACT_DEDUP_SCORE: float = 0.88   # ≥ this → update the existing fact instead of adding
    MEMORY_FACT_KNOWN_LIMIT: int = 8        # related known facts shown to the distiller
    MEMORY_FACT_MAX_TOKENS: int = 600
    MEMORY_FACT_TEMPERATURE: float = 0.0
    MEMORY_FACT_ENABLE_THINKING: Optional[bool] = False  # None → model/gateway default
    MEMORY_FACT_PROMPT: Optional[str] = None             # overrides the built-in distiller prompt
    MEMORY_HISTORY_KEEP: int = 20           # prior versions kept per updated memory
    MEMORY_PROMPT_HEADER: Optional[str] = None           # overrides the recalled-memory block header
    MEMORY_PROMPT_USER_TITLE: str = "### Known about the user"
    MEMORY_PROMPT_SESSION_TITLE: str = "### Earlier in this chat"

    # Native corpus RAG (preflight ANN + rerank → system prompt)
    # Uses separate bindings service.orchestrator.rag_embed / rag_rerank.
    NATIVE_RAG_ENABLED: bool = True
    NATIVE_RAG_TOP_K: int = 5
    NATIVE_RAG_CANDIDATE_LIMIT: int = 25
    NATIVE_RAG_SCORE_THRESHOLD: Optional[float] = 0.3
    NATIVE_RAG_COLLECTION: str = "tool_embed_text"
    NATIVE_RAG_TIMEOUT_S: float = 20.0
    NATIVE_RAG_MAX_CHARS_PER_HIT: int = 1200
    NATIVE_RAG_QUERY_MAX_CHARS: int = 4000  # retrieval query = head + tail of longer messages
    
    # Chat History Settings (via Storage Service - no direct DB access)
    CHAT_HISTORY_ENABLED: bool = True
    CHAT_HISTORY_LIMIT: int = 20  # تعداد پیام‌های قبلی برای context
    
    # Conversation Intelligence — smart history summarization
    HISTORY_SUMMARY_ENABLED: bool = True
    HISTORY_SUMMARY_CACHE_TTL: int = 300       # seconds — cache for summarized history
    # Increased from 1.5s — at 1.5s the summarizer almost always timed out on
    # quantized models, falling back to a thin rule-based summary which loses
    # goals/constraints/decisions and is the dominant cause of mid-chat drift.
    HISTORY_SUMMARY_LLM_TIMEOUT: float = 8.0
    HISTORY_SUMMARY_LLM_RETRIES: int = 1        # one retry on timeout/failure
    HISTORY_SUMMARY_CACHE_ENABLED: bool = True  # in-process cache by older-block hash
    HISTORY_RECENT_FULL_COUNT: int = 6          # min recent user/assistant msgs kept verbatim
    FILE_REF_LLM_TIMEOUT: float = 2.0           # seconds — timeout for LLM-based file reference resolution

    # SelfCompact (arXiv 2606.23525): rubric-gated compaction of the chat
    # trajectory. Every value can be overridden per chat through
    # ``orchestrator_config`` keys ``self_compact_<name>`` (lower-case).
    SELF_COMPACT_ENABLED: bool = True
    SELF_COMPACT_RUBRIC: str = "chat"              # prompt set under context/self_compact_prompts/<rubric>/
    SELF_COMPACT_PROBE_PERIOD: int = 2             # PERIOD gate: rounds since the last probe
    SELF_COMPACT_MIN_ROUNDS: int = 3               # ROUND gate: rounds since the last compaction
    SELF_COMPACT_PROBE_MIN_RATIO: float = 0.15     # TOKENS gate: prompt ≥ ratio × context window
    SELF_COMPACT_MAX_RUBRIC_PER_TURN: int = 1      # CAP gate: rubric-fired summaries per turn
    SELF_COMPACT_BACKSTOP_RATIO: float = 0.30      # forced COMPRESS at ratio × context window
    # ...only when the summarizable part (not the system prompt / this turn's message)
    # is at least this × window — otherwise re-summarizing a fresh summary gains nothing.
    SELF_COMPACT_BACKSTOP_MIN_TRAJECTORY_RATIO: float = 0.10
    SELF_COMPACT_HARD_RATIO: float = 0.95          # prompt + output reserve must stay under this
    SELF_COMPACT_MAX_RESERVE_RATIO: float = 0.25   # cap on the output reservation vs the window
    SELF_COMPACT_OUTPUT_RESERVE_TOKENS: int = 4096 # used when neither catalog nor client sets max tokens
    SELF_COMPACT_FALLBACK_CONTEXT_TOKENS: int = 0  # 0 → STREAM_CONTEXT_CHAR_BUDGET / chars_per_token
    SELF_COMPACT_CHARS_PER_TOKEN: float = 3.0      # estimate; calibrated from provider usage at runtime
    SELF_COMPACT_IMAGE_TOKENS: int = 1024
    # Learned estimator scale is clamped to [1/MAX, MAX]; BACKOFF grows it when
    # the provider only reports a lower bound ("at least N tokens") or no count.
    SELF_COMPACT_MAX_TOKEN_SCALE: float = 4.0
    SELF_COMPACT_SCALE_BACKOFF: float = 1.5          # estimate per image/video part
    SELF_COMPACT_PROBE_MAX_TOKENS: int = 512
    SELF_COMPACT_SUMMARY_MAX_TOKENS: int = 2048
    # Summarize in turn chunks of at most this share of the window (0 = one
    # call, as in the paper); keeps each call to roughly one long document.
    SELF_COMPACT_SUMMARY_CHUNK_RATIO: float = 0.15
    # Chunk summaries run concurrently (bounded) and are merged in one call;
    # 1 keeps the sequential running-summary behaviour. Small models merging
    # several partial summaries tend to keep only the last one.
    SELF_COMPACT_SUMMARY_PARALLEL: int = 1
    # Summarizer views drop the turn's system prompt + tool schemas when those alone exceed
    # this share of the window (small windows); 0 = always keep them.
    SELF_COMPACT_SUMMARY_HEAD_MAX_RATIO: float = 0.3
    # Lines of the carried summary holding a value (any token with a digit: codes, ids,
    # amounts, dates) that the new summary dropped are re-attached under
    # self_compact_prompts/carried_values.md; small models sometimes lose them in one pass.
    SELF_COMPACT_KEEP_CARRIED_VALUES: bool = True
    # Cap each chat round's max_tokens to the room left in the window, so a served
    # window larger than the configured one cannot run reasoning past it.
    SELF_COMPACT_CAP_COMPLETION: bool = True
    SELF_COMPACT_TEMPERATURE: float = 0.0
    SELF_COMPACT_JUDGE_THINKING: bool = False      # enable_thinking for probe/summarizer calls
    # Recalled memory / RAG blocks are re-retrieved each turn; keep them out of the
    # summarizer's view so stale excerpts cannot overwrite corrected facts.
    SELF_COMPACT_SUMMARY_EXCLUDE_RECALLED: bool = True
    SELF_COMPACT_MAX_OVERFLOW_RETRIES: int = 2     # provider overflow → compact and retry the round
    SELF_COMPACT_HISTORY_LIMIT: int = 200          # stored messages fetched to rebuild the trajectory (storage page max 500)

    # Session compaction — the host decides, before any engine runs, whether the
    # conversation still fits the serving model (catalog context_length and
    # max_output_tokens). RLM-style: older whole turns fold into a structured
    # checkpoint, recent turns stay verbatim, a tool call never loses its result.
    # Engines' own compactors then only guard growth inside the turn.
    # Per chat: orchestrator_config["session_compact_<name>"].
    SESSION_COMPACT_ENABLED: bool = True
    SESSION_COMPACT_HISTORY_SHARE: float = 0.5     # share of the input window (window − output reserve) history may use
    SESSION_COMPACT_TRIGGER_RATIO: float = 0.9     # compact when history ≥ ratio × its share
    SESSION_COMPACT_KEEP_RECENT_RATIO: float = 0.35  # verbatim tail kept, as a ratio of the history share
    SESSION_COMPACT_MAX_OUTPUT_RATIO: float = 0.25   # cap on the catalog output reserve vs the window
    SESSION_COMPACT_MAX_ROWS_RATIO: float = 0.9      # also compact when stored rows ≥ ratio × SELF_COMPACT_HISTORY_LIMIT
    SESSION_COMPACT_SUMMARIZER_FILE: str = ""        # "" → context/session_compact_prompts/summarizer.md

    # Long input — a user message larger than TRIGGER_RATIO × the model window
    # is read by parallel sub-agent workers (one per part) guided by a task
    # checklist; every engine then receives the checklist + per-part notes +
    # the verbatim head/tail instead of the raw text. Storage keeps the
    # original. Per chat: orchestrator_config["long_input_<name>"].
    LONG_INPUT_ENABLED: bool = True
    LONG_INPUT_TRIGGER_RATIO: float = 0.25         # message tokens ≥ ratio × context window
    LONG_INPUT_TRIGGER_TOKENS: int = 0             # > 0 → absolute trigger, overrides the ratio
    LONG_INPUT_CHUNK_RATIO: float = 0.15           # one worker reads at most ratio × window
    LONG_INPUT_CHUNK_OVERLAP_CHARS: int = 400
    LONG_INPUT_MAX_CHUNKS: int = 64                # chunk size grows instead of dropping text
    LONG_INPUT_MAX_PARALLEL: int = 4
    LONG_INPUT_KEEP_HEAD_CHARS: int = 1500         # verbatim head/tail (instructions usually live there)
    LONG_INPUT_KEEP_TAIL_CHARS: int = 1500
    LONG_INPUT_DIGEST_MAX_RATIO: float = 0.30      # digest budget vs the window; over → reduce pass
    LONG_INPUT_MAX_REDUCE_PASSES: int = 2
    LONG_INPUT_MAX_TASKS: int = 24
    LONG_INPUT_WORKER_MAX_TOKENS: int = 1024
    LONG_INPUT_CHECKLIST_MAX_TOKENS: int = 512
    LONG_INPUT_TEMPERATURE: float = 0.0
    LONG_INPUT_ENABLE_THINKING: bool = False
    LONG_INPUT_CHARS_PER_TOKEN: float = 3.0
    LONG_INPUT_FALLBACK_CONTEXT_TOKENS: int = 0    # 0 → STREAM_CONTEXT_CHAR_BUDGET / chars_per_token
    LONG_INPUT_BINDING_SERVICE: str = "orchestrator"
    LONG_INPUT_PURPOSE: str = "chat"
    LONG_INPUT_PROMPTS_DIR: str = ""               # "" → context/long_input_prompts/

    # Pinned Conversation Facts — deterministic anchor that never disappears
    # from the prompt regardless of how aggressively older messages are
    # summarized. Prevents the model from "forgetting the goal" mid-chat.
    PINNED_FACTS_ENABLED: bool = True
    PINNED_FACTS_MAX_CONSTRAINTS: int = 6
    PINNED_FACTS_MAX_DECISIONS: int = 6
    PINNED_FACTS_LLM_ENABLED: bool = False      # optional LLM enrichment (off by default)
    PINNED_FACTS_LLM_TIMEOUT: float = 6.0

    # Anti-loop / repetition guard — proactive system block injected into the
    # prompt once the conversation is long enough that drift becomes likely,
    # plus a reactive detector that flags repeats so the UI/caller can react.
    ANTI_LOOP_GUARD_ENABLED: bool = True
    ANTI_LOOP_TURN_THRESHOLD: int = 4            # inject anti-loop block from turn N
    REPETITION_DETECTION_ENABLED: bool = True
    REPETITION_SIMILARITY_THRESHOLD: float = 0.82

    # Native engine: a call to a tool that was not offered this turn is never dispatched;
    # the model gets this message as the tool result instead.
    NATIVE_REJECT_UNOFFERED_TOOLS: bool = True
    NATIVE_UNKNOWN_TOOL_TEMPLATE: str = (
        "error: `{name}` is not a tool available to you, so nothing was executed. "
        "Available tools: {available}. Do not pass `{name}` as an argument to another tool; "
        "tell the user this action is not available."
    )

    # Tools declaring "side_effects": true in tool.json (src/tools/side_effects.py): checked
    # against the conversation before they run (prompt: src/tools/prompts/action_check.md),
    # run once per turn for identical calls, simulated in rehearsal runs.
    TOOL_SIDE_EFFECTS_POLICY: bool = True
    TOOL_ACTION_CHECK_ENABLED: bool = True
    TOOL_ACTION_CHECK_ON_ERROR: str = "deny"          # "allow" | "deny" when the check fails
    TOOL_ACTION_CHECK_MAX_TOKENS: int = 400
    TOOL_ACTION_CHECK_HISTORY_MESSAGES: int = 16      # most recent chat messages shown (0 = all)
    TOOL_ACTION_CHECK_MESSAGE_CHARS: int = 1200       # per message / tool result (0 = full)
    TOOL_ACTION_CHECK_OBSERVATIONS: int = 12          # tool results of the turn shown
    TOOL_ACTION_HELD_TEMPLATE: str = (
        "Not executed: {reason}. Ask the user for what is missing or for confirmation; "
        "do not say the action was done."
    )
    TOOL_SIDE_EFFECTS_REHEARSAL_MESSAGE: str = (
        "Rehearsal run: the action was simulated, nothing was changed. Treat it as accepted."
    )

    # Chat turns run detached from the HTTP stream (src/chat_turns.py): leaving the page does
    # not stop them. With CHAT_TURN_CONTINUE_ON_DISCONNECT off, a disconnect stops the turn.
    CHAT_TURN_CONTINUE_ON_DISCONNECT: bool = True
    CHAT_TURN_REPLAY_MAX_EVENTS: int = 20000     # buffered SSE events per turn for reattach
    CHAT_TURN_RETAIN_S: float = 600.0            # keep a finished turn for reattach this long
    CHAT_TURN_WAIT_PREVIOUS_S: float = 0.0       # new message waits for a running turn (0 = no limit)
    # Explicit stop: the partial turn is saved, the engine generator is closed and (when
    # enabled) every engine's live in-memory state for the chat is released.
    CHAT_STOP_RELEASE_ENGINE_STATE: bool = True
    CHAT_STOP_CLEANUP_TIMEOUT_S: float = 15.0

    # Per-turn guidance blocks for every engine (src/context/turn_guidance.py).
    # Empty template → block disabled.
    HOST_CLOCK_TIMEZONE: str = "UTC"               # IANA zone used when the client sends none
    HOST_CLOCK_CONTEXT_KEY: str = "timezone"       # ChatRequest.context key carrying the client's zone
    HOST_CLOCK_TEMPLATE: str = (
        "## Current date and time\n"
        "Today is {weekday}, {date} ({timezone}); the time is {time}. Resolve relative dates "
        "(today, tomorrow, next Friday, …) from this date and pass dates to tools as YYYY-MM-DD."
    )
    HOST_PERSONA_TEMPLATE: str = (
        "## Instructions for this chat\n"
        "Set by the operator of this chat. Follow them for the whole conversation, however long it "
        "gets, unless they conflict with safety rules. Anything they mark as secret or not to be "
        "revealed must never appear in your replies, not even inside a refusal or a quote.\n\n"
        "{instructions}"
    )
    HOST_PERSONA_MAX_CHARS: int = 8000
    USER_LOOP_REPEATS: int = 3                     # same user message N times in a row → loop guard
    USER_LOOP_SIMILARITY: float = 0.8              # word-set Jaccard for "same message"
    USER_LOOP_TEMPLATE: str = (
        "## Conversation loop detected\n"
        "The user has sent essentially the same message {count} times in a row, so your earlier "
        "answers did not resolve it. Do not repeat them. Say briefly that you seem to be going in "
        "circles, try one clearly different approach (or ask what exactly is unclear), and offer to "
        "escalate to a human support agent."
    )

    # Text-file inlining — embed small text/code/markdown files directly into
    # the system prompt so the LLM can reason about their content without
    # making a `text.read` / `session_read` tool call. Mirrors what we already
    # do for images via multimodal data: URLs.
    INLINE_TEXT_FILES_ENABLED: bool = True
    INLINE_TEXT_FILE_THRESHOLD_BYTES: int = 32_768  # per-file size cap for auto-inlining
    INLINE_TEXT_PER_FILE_CHARS: int = 8_000          # per-file character cap in the prompt
    INLINE_TEXT_TOTAL_CHARS: int = 32_000            # total character cap across all files
    INLINE_TEXT_MAX_FILES: int = 6                   # at most this many text files inlined per turn

    # Markdown session rules (Continue-style: YAML frontmatter + globs + regex + per-rule on/off)
    RULES_ENABLED: bool = True
    # Comma-separated list of directories to scan for ``*.md`` rule files.
    # Empty → use packaged ``agent_rules`` under the orchestrator service if present.
    RULES_MARKDOWN_DIRS: str = ""
    RULES_CACHE_TTL_SEC: float = 30.0
    
    # Redis for state persistence
    REDIS_URL: str = "redis://redis:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
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


def llm_gateway_v1_url() -> str:
    """OpenAI-compatible base URL for the in-cluster llm-gateway."""
    base = (settings.LLM_PROXY_URL or "http://llm-proxy:8005").rstrip("/")
    return f"{base}/v1"


def get_llm_config() -> dict:
    """Transport settings for GatewayLLMClient (model resolved via binding_client)."""
    # In-cluster llm-gateway authenticates with INTERNAL_API_KEY (Bearer or X-Internal-Key).
    api_key = (settings.INTERNAL_API_KEY or "").strip() or "not-needed"
    return {
        "base_url": llm_gateway_v1_url(),
        "api_key": api_key,
        "temperature": settings.LLM_TEMPERATURE,
        "timeout": settings.LLM_TIMEOUT,
        "max_retries": settings.LLM_MAX_RETRIES,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "tool_call_parser": settings.LLM_TOOL_CALL_PARSER,
    }
