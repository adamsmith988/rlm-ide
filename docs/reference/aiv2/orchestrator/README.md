# Orchestrator Service v2.0

سرویس ارکستراتور هوش مصنوعی با **llm-gateway** و **LangGraph**

## 📋 معماری

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator Service                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   FastAPI    │───▶│   LangGraph     │───▶│ Gateway LLM     │ │
│  │   Endpoints  │    │   Agent Graph   │    │ Client (OpenAI) │ │
│  └──────────────┘    └─────────────────┘    └─────────────────┘ │
│         │                    │                                   │
│         │                    ▼                                   │
│         │           ┌─────────────────┐    ┌─────────────────┐ │
│         │           │  Tool Executor  │───▶│  Tool Runner    │ │
│         │           │                 │    │  (External)     │ │
│         │           └─────────────────┘    └─────────────────┘ │
│         │                    │                                   │
│         ▼                    ▼                                   │
│  ┌──────────────┐    ┌─────────────────┐                        │
│  │   Schemas    │    │  Memory (Qdrant)│                        │
│  │   Pydantic   │    │    + Qdrant     │                        │
│  └──────────────┘    └─────────────────┘                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 شروع سریع

### روش ۱: Docker

```bash
docker-compose up orchestrator
```

### روش ۲: Virtual Environment جداگانه

```bash
# راه‌اندازی محیط مجازی
./setup_venv.sh

# فعال‌سازی
source .venv/bin/activate

# کپی تنظیمات
cp .env.example .env

# اجرا
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

## 📁 ساختار پروژه

```
orchestrator/
├── Dockerfile
├── requirements.txt
├── setup_venv.sh          # اسکریپت راه‌اندازی venv
├── .env.example
├── README.md
└── src/
    ├── __init__.py
    ├── main.py             # FastAPI application
    ├── config.py           # Settings (pydantic-settings)
    ├── graph/
    │   └── __init__.py     # LangGraph agent workflow
    ├── llm/
    │   ├── __init__.py
    │   └── gateway_llm_client.py  # OpenAI-compat client for llm-gateway
    ├── memory/
    │   └── __init__.py     # Qdrant chat memory
    ├── prompts/
    │   ├── __init__.py
    │   └── system.py       # System prompts
    ├── schemas/
    │   └── __init__.py     # Pydantic models
    ├── tools/
    │   ├── __init__.py
    │   └── executor.py     # Tool execution via tool-runner
    └── utils/
        ├── __init__.py
        └── file_context.py # File type detection
```

## 🔌 API Endpoints

| Method | Endpoint | توضیح |
|--------|----------|-------|
| GET | `/health` | بررسی سلامت سرویس |
| POST | `/chat` | پردازش پیام چت |
| POST | `/chat/stream` | استریم پاسخ (SSE) |
| GET | `/tools` | لیست ابزارهای موجود |
| POST | `/tools/run` | اجرای مستقیم ابزار |

### مثال درخواست Chat

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "این تصویر را توصیف کن",
    "session_id": "test-session-123",
    "artifacts": [
      {"path": "/data/uploads/image.jpg", "mime_type": "image/jpeg"}
    ]
  }'
```

## ⚙️ تنظیمات

| متغیر | پیش‌فرض | توضیح |
|-------|---------|-------|
| `LLM_PROXY_URL` | `http://llm-proxy:8005` | آدرس llm-gateway (مدل از `llm_routes` / binding resolve) |
| `STRICT_LLM_BINDINGS` | `true` | بدون binding معتبر، chat خطا می‌دهد (بدون fallback از env) |
| `TOOL_RUNNER_URL` | `http://localhost:8002` | آدرس سرویس ابزارها |
| `MAX_STEPS` | `3` | حداکثر گام‌های agent |
| `MEMORY_ENABLED` | `true` | فعال‌سازی حافظه |

## 🔄 گردش کار LangGraph

```
                    ┌──────────────────┐
                    │    Initialize    │
                    │  (file context)  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
            ┌──────│    Call LLM      │──────┐
            │      │  (gateway + tools)│      │
            │      └────────┬─────────┘      │
            │               │                │
            ▼               ▼                ▼
      ┌───────────┐  ┌────────────┐  ┌──────────────┐
      │   END     │  │ Execute    │  │ Force Final  │
      │ (answer)  │  │   Tools    │  │   (max_steps)│
      └───────────┘  └─────┬──────┘  └──────────────┘
                           │
                           └──────────▶ (loop back to LLM)
```

## 🧪 تست

```bash
# بررسی سلامت
curl http://localhost:8001/health

# لیست ابزارها
curl http://localhost:8001/tools

# تست چت ساده
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "سلام", "session_id": "test"}'
```

## 📝 تفاوت با نسخه قبلی

| ویژگی | v1 (قدیمی) | v2 (جدید) |
|-------|------------|-----------|
| LLM Client | Qwen-Agent + Ollama | llm-gateway (OpenAI-compatible) |
| Agent Loop | JSON-based manual | LangGraph StateGraph |
| Tool Execution | Local ToolRegistry | Remote tool-runner service |
| Memory | Qdrant | embeddings via llm-gateway binding |
| Streaming | Limited | SSE + LangGraph |

## 🔗 وابستگی‌ها

- **llm-gateway**: باید روی پورت 8005 در دسترس باشد (LB به node/external endpoints)
- **Tool Runner**: سرویس ابزارها روی پورت 8002
- **Qdrant** (اختیاری): برای حافظه چت (کوتاه‌مدت و بلندمدت)

---
**نسخه**: 2.0.0  
**تاریخ**: دسامبر 2025
