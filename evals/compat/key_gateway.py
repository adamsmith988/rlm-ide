"""Bearer-key gate in front of a keyless OpenAI-compatible server (e.g. Ollama).

Eval-only. Logs one JSON line per request with a SHA-256 fingerprint of the presented key
(never the key itself), so the harness can check which key reached which server.

With --tap the gateway checks nothing: it forwards the Authorization header untouched and only
records the fingerprint, so the upstream server's own key check is what gets tested.

usage: key_gateway.py --port 8203 --upstream http://127.0.0.1:11434 --key-file keys.txt --log requests.jsonl [--tap]
"""

import argparse
import hashlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "authorization", "host", "content-length"}


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12] if key else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--key-file")
    ap.add_argument("--log", required=True)
    ap.add_argument("--tap", action="store_true")
    a = ap.parse_args()
    up = urlsplit(a.upstream)
    lock = threading.Lock()

    def allowed() -> set[str]:
        with open(a.key_file) as f:
            return {line.strip() for line in f if line.strip()}

    def record(entry: dict) -> None:
        with lock, open(a.log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # quiet
            pass

        def handle_any(self) -> None:
            auth = self.headers.get("Authorization", "")
            key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            ok = True if a.tap else bool(key) and key in allowed()
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            entry = {"t": time.time(), "method": self.command, "path": self.path, "key_fp": fingerprint(key), "authorized": None if a.tap else ok}
            if not ok:
                msg = json.dumps({"error": {"message": "invalid or missing API key", "type": "invalid_request_error", "code": "invalid_api_key"}}).encode()
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                record({**entry, "status": 401})
                return
            conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            if a.tap and auth:
                headers["Authorization"] = auth
            conn.request(self.command, self.path, body=body, headers=headers)
            res = conn.getresponse()
            self.send_response(res.status)
            for k, v in res.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = res.read1(65536) if hasattr(res, "read1") else res.read(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            conn.close()
            record({**entry, "status": res.status})

        do_GET = handle_any
        do_POST = handle_any

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"key gateway on :{a.port} -> {a.upstream}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
