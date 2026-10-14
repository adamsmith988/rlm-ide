# معماری Plan-Execute-Revise برای ارکستراتور

## 🎯 خلاصه تغییرات

این تغییرات یک معماری **Multi-Agent Planning** جدید به ارکستراتور اضافه می‌کند که قابلیت‌های زیر را دارد:

1. **تفکر چند مرحله‌ای**: به جای یک حلقه ساده ReAct، سیستم ابتدا **برنامه‌ریزی** می‌کند
2. **ارزیابی خودکار**: یک **Critic** نتایج هر مرحله را ارزیابی می‌کند
3. **Re-planning**: در صورت نیاز، برنامه به طور خودکار **بازبینی** می‌شود
4. **ترکیب نتایج**: یک **Synthesizer** نتایج را به پاسخ نهایی تبدیل می‌کند

## 🏗️ معماری جدید

```
                    ┌───────────────────────────────────────────────┐
                    │                                               │
START → assess_complexity ──┬── (simple) ──→ simple_response → END │
                            │                                       │
                            └── (complex) ─→ planner ───────────────┤
                                                ↓                   │
                                            executor ←──────────────┤
                                                ↓                   │
                                             critic ────────────────┤
                                                ↓                   │
                                    ┌───────────┴────────┐          │
                                    ↓                    ↓          │
                              replanner              synthesizer    │
                                    ↓                    ↓          │
                                executor               END          │
                                    │                               │
                                    └───────────────────────────────┘
```

## 📁 فایل‌های جدید

### 1. `src/graph/planning_state.py`
State جدید با فیلدهای Planning:
- `execution_plan`: برنامه اجرایی
- `task_results`: نتایج هر sub-task
- `evaluations`: ارزیابی‌های Critic
- `replan_count`: تعداد re-planning ها

### 2. `src/graph/planning_nodes.py`
گره‌های جدید:
- `assess_complexity_node`: ارزیابی پیچیدگی درخواست
- `planner_node`: ساخت برنامه اجرایی
- `executor_node`: اجرای هر sub-task
- `critic_node`: ارزیابی نتایج
- `replanner_node`: بازبینی برنامه
- `synthesizer_node`: ترکیب نتایج نهایی

### 3. `src/graph/planning_agent.py`
گراف جدید با routing هوشمند

### 4. `src/prompts/planning_prompts.py`
پرامپت‌های تخصصی برای هر Agent

## ⚙️ تنظیمات

در `config.py`:

```python
PLANNING_ENABLED: bool = True  # فعال/غیرفعال کردن Planning Agent
MAX_REPLANS: int = 3           # حداکثر تعداد re-planning
MAX_STEPS: int = 10            # حداکثر گام‌ها (افزایش یافته)
```

## 🔄 جریان اجرا

### مثال: "تحلیل این تصویر و پیدا کردن افراد و متن‌ها"

1. **Assess Complexity**: تشخیص می‌دهد که کار پیچیده است (چون فایل دارد)

2. **Planner**: برنامه می‌سازد:
   ```json
   {
     "tasks": [
       {"id": "task_1", "description": "تشخیص چهره‌ها", "tools_needed": ["image.face_detection"]},
       {"id": "task_2", "description": "استخراج متن", "tools_needed": ["image.ocr"]},
       {"id": "task_3", "description": "ترکیب نتایج", "depends_on": ["task_1", "task_2"]}
     ]
   }
   ```

3. **Executor** (task_1): اجرای face_detection

4. **Critic**: ارزیابی - آیا چهره‌ها یافت شدند؟ confidence: 0.85

5. **Executor** (task_2): اجرای OCR

6. **Critic**: ارزیابی - آیا متن استخراج شد؟ confidence: 0.90

7. **Synthesizer**: ترکیب نتایج به پاسخ نهایی

## 📊 مزایای معماری جدید

| ویژگی | ReAct قبلی | Plan-Execute-Revise |
|-------|-----------|---------------------|
| تفکر | یک مرحله‌ای | چند مرحله‌ای |
| خطایابی | Retry ساده | Re-planning هوشمند |
| پیچیدگی | کارهای ساده | کارهای پیچیده |
| شفافیت | کم | بالا (نمایش plan) |
| کیفیت | متوسط | بالا |

## 🧪 تست

```bash
# فعال کردن Planning Agent
export PLANNING_ENABLED=true

# تست با curl
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "تحلیل کامل این تصویر",
    "session_id": "test-session",
    "artifacts": [{"file_id": "...", "type": "image", "path": "/files/test.jpg"}]
  }'
```

## 🔧 غیرفعال کردن (برگشت به حالت قبل)

```bash
export PLANNING_ENABLED=false
```

یا در `.env`:
```
PLANNING_ENABLED=false
```
