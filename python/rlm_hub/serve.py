"""OpenAI-compatible server for a downloaded Hugging Face model (transformers or ONNX Runtime).

Endpoints: GET /v1/models, POST /v1/chat/completions (streaming and not), POST /v1/completions,
POST /v1/embeddings, GET /health. Every /v1 request needs `Authorization: Bearer <key>`, the key
coming from the RLM_HUB_API_KEY environment variable (never from the command line).

The model is described only by its own files: config.json decides the model class, the tokenizer's
chat template formats chats, and sentence-transformers pooling files decide how embeddings pool.
"""
import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import import_guard

import_guard.install()

CTX_KEYS = ("max_position_embeddings", "n_positions", "max_sequence_length", "max_seq_len", "seq_length", "n_ctx", "model_max_length")


def log(msg):
    print(f"[serve] {msg}", file=sys.stderr, flush=True)


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def context_from_config(cfg):
    if not isinstance(cfg, dict):
        return None
    for scope in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {}):
        for k in CTX_KEYS:
            v = scope.get(k)
            if isinstance(v, int) and 0 < v < 10_000_000:
                return v
    return None


class Engine:
    def __init__(self, a):
        self.a = a
        self.dir = Path(a.model_dir)
        self.cfg = read_json(self.dir / "config.json") or {}
        self.lock = threading.Lock()
        self.kind = a.kind
        import torch  # noqa: F401  (fail early with a clear error)
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(self.dir), trust_remote_code=a.trust_remote_code)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        self.device = self.pick_device()
        self.model = self.load_onnx() if a.backend == "onnx" else self.load_transformers()
        if hasattr(self.model, "eval"):
            self.model.eval()
        ctx = a.max_context or context_from_config(self.cfg)
        tml = getattr(self.tok, "model_max_length", None)
        if not ctx and isinstance(tml, int) and tml < 10_000_000:
            ctx = tml
        st = read_json(self.dir / "sentence_bert_config.json")
        if self.kind == "embed" and st and isinstance(st.get("max_seq_length"), int):
            ctx = min(ctx or st["max_seq_length"], st["max_seq_length"])
        self.ctx = ctx or 2048
        self.pooling, self.normalize = self.pooling_config()
        log(f"loaded {self.dir.name} backend={a.backend} kind={self.kind} device={self.device} ctx={self.ctx}")

    def pick_device(self):
        import torch

        if self.a.device == "cpu":
            return "cpu"
        if self.a.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch sees no GPU")
        return "cuda" if torch.cuda.is_available() and self.a.device in ("cuda", "auto") else "cpu"

    def dtype(self):
        import torch

        name = self.a.dtype
        if name == "auto":
            return torch.float16 if self.device == "cuda" else torch.float32
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]

    def load_transformers(self):
        import transformers as tf

        kw = {"trust_remote_code": self.a.trust_remote_code}
        q = self.a.quant
        if q != "none":
            if self.device != "cuda":
                raise SystemExit(f"bitsandbytes quantization ({q}) needs a CUDA GPU; choose device cuda or no quantization")
            kw["quantization_config"] = tf.BitsAndBytesConfig(
                load_in_8bit=q == "bnb-8bit",
                load_in_4bit=q.startswith("bnb-4bit"),
                bnb_4bit_quant_type="fp4" if q == "bnb-4bit-fp4" else "nf4",
                bnb_4bit_compute_dtype=self.dtype() if self.a.dtype != "auto" else None,
            )
            kw["device_map"] = "auto"
        else:
            kw["dtype"] = self.dtype()
        if self.kind == "embed":
            m = tf.AutoModel.from_pretrained(str(self.dir), **kw)
        elif self.cfg.get("is_encoder_decoder"):
            m = tf.AutoModelForSeq2SeqLM.from_pretrained(str(self.dir), **kw)
        else:
            m = None
            errors = []
            for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
                auto = getattr(tf, cls, None)
                if auto is None:
                    continue
                try:
                    m = auto.from_pretrained(str(self.dir), **kw)
                    break
                except (ValueError, KeyError) as e:
                    errors.append(f"{cls}: {e}")
            if m is None:
                raise SystemExit("no text-generation class of transformers accepts this model: " + " | ".join(errors)[:800])
        if q == "none":
            m.to(self.device)
        return m

    def load_onnx(self):
        try:
            import optimum.onnxruntime as ort
        except ImportError as e:
            raise SystemExit(f"the ONNX backend needs optimum[onnxruntime]: {e}")
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        kw = {"provider": provider}
        if self.a.onnx_file:
            p = Path(self.a.onnx_file)
            kw["file_name"] = p.name
            if p.parent != Path("."):
                kw["subfolder"] = str(p.parent)
        if self.kind == "embed":
            cls = ort.ORTModelForFeatureExtraction
        elif self.cfg.get("is_encoder_decoder"):
            cls = ort.ORTModelForSeq2SeqLM
        else:
            cls = ort.ORTModelForCausalLM
        return cls.from_pretrained(str(self.dir), **kw)

    def pooling_config(self):
        """sentence-transformers layout when present; mean pooling + L2 norm otherwise."""
        modules = read_json(self.dir / "modules.json") or []
        mode, normalize = "mean", self.kind == "embed"
        if modules:
            normalize = any(m.get("type", "").endswith("Normalize") for m in modules)
            for m in modules:
                if m.get("type", "").endswith("Pooling"):
                    pc = read_json(self.dir / m.get("path", "") / "config.json") or {}
                    for key, name in (("pooling_mode_cls_token", "cls"), ("pooling_mode_lasttoken", "last"), ("pooling_mode_max_tokens", "max"), ("pooling_mode_mean_tokens", "mean")):
                        if pc.get(key):
                            mode = name
                            break
        return mode, normalize

    # ---- chat -------------------------------------------------------------------------------
    def prompt_ids(self, messages):
        msgs = [{"role": m.get("role", "user"), "content": text_of(m.get("content"))} for m in messages]
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False)
            except Exception as e:  # noqa: BLE001  some templates refuse a system role
                if not any(m["role"] == "system" for m in msgs):
                    raise
                log(f"chat template refused the system role ({e}); merging it into the first user message")
                sys_text = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
                rest = [m for m in msgs if m["role"] != "system"]
                if rest and rest[0]["role"] == "user":
                    rest[0] = {"role": "user", "content": f"{sys_text}\n\n{rest[0]['content']}"}
                else:
                    rest.insert(0, {"role": "user", "content": sys_text})
                return self.tok.apply_chat_template(rest, add_generation_prompt=True, return_tensors="pt", return_dict=False)
        text = "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant:"
        return self.tok(text, return_tensors="pt").input_ids

    def generate(self, ids, max_tokens, temperature, top_p, stop, on_text):
        """Streams decoded text through on_text; returns (completion_tokens, finish_reason)."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        ids = ids.to(self.model.device) if hasattr(self.model, "device") else ids
        room = max(1, self.ctx - ids.shape[-1])
        max_new = max(1, min(max_tokens or room, room))
        stop_flag = threading.Event()

        class Stop(StoppingCriteria):
            def __call__(self, *_a, **_k):
                return stop_flag.is_set()

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "max_new_tokens": max_new,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([Stop()]),
            "pad_token_id": self.tok.pad_token_id,
        }
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=float(temperature), top_p=float(top_p or 1.0))
        else:
            kw.update(do_sample=False)
        result = {}

        def run():
            try:
                with torch.inference_mode():
                    result["out"] = self.model.generate(**kw)
            except Exception as e:  # noqa: BLE001
                result["error"] = e
                stop_flag.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        text, sent, finish = "", 0, "length"
        for piece in streamer:
            text += piece
            cut = min((text.find(s) for s in stop if s and s in text), default=-1)
            if cut >= 0:
                text = text[:cut]
                finish = "stop"
                stop_flag.set()
            if len(text) > sent:
                on_text(text[sent:])
                sent = len(text)
            if stop_flag.is_set():
                break
        t.join()
        if "error" in result:
            raise result["error"]
        out = result.get("out")
        n = int(out.shape[-1] - ids.shape[-1]) if out is not None and not self.cfg.get("is_encoder_decoder") else len(self.tok(text).input_ids)
        if finish != "stop" and n < max_new:
            finish = "stop"
        return n, finish

    # ---- embeddings ---------------------------------------------------------------------------
    def embed(self, texts):
        import torch

        enc = self.tok(texts, padding=True, truncation=True, max_length=self.ctx, return_tensors="pt")
        dev = getattr(self.model, "device", "cpu")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "cls":
            vec = hidden[:, 0]
        elif self.pooling == "last":
            idx = enc["attention_mask"].sum(dim=1) - 1
            vec = hidden[torch.arange(hidden.shape[0]), idx]
        elif self.pooling == "max":
            vec = (hidden + (mask - 1) * 1e9).max(dim=1).values
        else:
            vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vec = vec.float()
        if self.normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
        return vec.cpu().tolist(), int(enc["attention_mask"].sum().item())


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def make_handler(engine, key, served):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def log_request(self, code="-", size="-"):
            log(f"request: {self.command} {self.path} {code}")

        def send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def error(self, code, msg, typ="invalid_request_error"):
            self.send_json(code, {"error": {"message": msg, "type": typ, "code": code}})

        def authorized(self):
            got = self.headers.get("Authorization", "")
            want = f"Bearer {key}"
            if hmac.compare_digest(got.encode(), want.encode()):
                return True
            self.error(401, "invalid or missing API key", "authentication_error")
            return False

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"status": "ok"})
            if not self.path.startswith("/v1"):
                return self.error(404, "not found")
            if not self.authorized():
                return
            if self.path.rstrip("/") == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [{"id": served, "object": "model", "owned_by": "rlm-hub", "max_model_len": engine.ctx, "kind": engine.kind}]})
            self.error(404, "not found")

        def do_POST(self):
            if not self.authorized():
                return
            try:
                req = self.body()
            except Exception:  # noqa: BLE001
                return self.error(400, "body is not JSON")
            path = self.path.rstrip("/")
            try:
                if path == "/v1/embeddings":
                    return self.embeddings(req)
                if path in ("/v1/chat/completions", "/v1/completions"):
                    if engine.kind != "chat":
                        return self.error(400, "this deployment serves embeddings only")
                    return self.complete(req, chat=path.endswith("chat/completions"))
                self.error(404, "not found")
            except Exception as e:  # noqa: BLE001
                log(f"request failed: {e!r}")
                try:
                    self.error(500, f"{type(e).__name__}: {e}"[:2000], "server_error")
                except Exception:  # noqa: BLE001
                    pass

        def embeddings(self, req):
            if engine.kind != "embed":
                return self.error(400, "this deployment serves chat only")
            inp = req.get("input")
            texts = [inp] if isinstance(inp, str) else list(inp or [])
            if not texts or not all(isinstance(t, str) for t in texts):
                return self.error(400, "input must be a string or a list of strings")
            with engine.lock:
                vecs, n = engine.embed(texts)
            self.send_json(200, {"object": "list", "model": served, "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)], "usage": {"prompt_tokens": n, "total_tokens": n}})

        def complete(self, req, chat):
            if chat:
                ids = engine.prompt_ids(req.get("messages") or [])
            else:
                ids = engine.tok(text_of(req.get("prompt")), return_tensors="pt").input_ids
            n_in = int(ids.shape[-1])
            if n_in >= engine.ctx:
                return self.error(400, f"This model's maximum context length is {engine.ctx} tokens. However, your prompt contains at least {n_in} input tokens.")
            stop = req.get("stop") or []
            stop = [stop] if isinstance(stop, str) else list(stop)
            max_tokens = req.get("max_tokens") or req.get("max_completion_tokens")
            if max_tokens and n_in + int(max_tokens) > engine.ctx:
                return self.error(400, f"This model's maximum context length is {engine.ctx} tokens. However, you requested {max_tokens} output tokens and your prompt contains at least {n_in} input tokens, for a total of at least {n_in + int(max_tokens)} tokens.")
            rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            obj = "chat.completion" if chat else "text_completion"
            args = (ids, max_tokens, req.get("temperature", 0.7), req.get("top_p", 1.0), stop)
            if not req.get("stream"):
                parts = []
                with engine.lock:
                    n_out, finish = engine.generate(*args, parts.append)
                text = "".join(parts)
                choice = {"index": 0, "finish_reason": finish}
                choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
                return self.send_json(200, {"id": rid, "object": obj, "created": created, "model": served, "choices": [choice], "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def event(payload):
                data = f"data: {payload}\n\n".encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def chunk(delta, finish=None):
                c = {"index": 0, "finish_reason": finish}
                c.update({"delta": delta} if chat else {"text": delta.get("content", "")})
                event(json.dumps({"id": rid, "object": f"{obj}.chunk" if chat else obj, "created": created, "model": served, "choices": [c]}))

            if chat:
                chunk({"role": "assistant", "content": ""})
            with engine.lock:
                n_out, finish = engine.generate(*args, lambda t: chunk({"content": t}))
            chunk({}, finish)
            if (req.get("stream_options") or {}).get("include_usage"):
                event(json.dumps({"id": rid, "object": f"{obj}.chunk", "created": created, "model": served, "choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}))
            event("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--backend", choices=["transformers", "onnx"], default="transformers")
    p.add_argument("--kind", choices=["chat", "embed"], required=True)
    p.add_argument("--served-name", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--quant", choices=["none", "bnb-8bit", "bnb-4bit-nf4", "bnb-4bit-fp4"], default="none")
    p.add_argument("--onnx-file", default="")
    p.add_argument("--max-context", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    key = os.environ.get("RLM_HUB_API_KEY", "")
    if len(key) < 16:
        raise SystemExit("RLM_HUB_API_KEY must hold the API key (16+ chars)")
    engine = Engine(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine, key, a.served_name))
    log(f"listening on http://{a.host}:{a.port}/v1")
    srv.serve_forever()


if __name__ == "__main__":
    main()
