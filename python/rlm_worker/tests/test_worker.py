import json
import os
import signal
import subprocess
import sys
import time

WORKER = os.path.join(os.path.dirname(__file__), "..", "worker.py")


class Proc:
    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-u", WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.read()["op"] == "ready"

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def read(self):
        return json.loads(self.p.stdout.readline())

    def until(self, op, on_request=None):
        while True:
            m = self.read()
            if m["op"] == "host_request" and on_request:
                self.send(on_request(m))
                continue
            if m["op"] == op:
                return m

    def close(self):
        self.send({"op": "shutdown"})
        self.p.wait(timeout=5)


def test_variables_persist_and_last_expression():
    k = Proc()
    k.send({"op": "exec", "id": "a", "code": "x = 40\nprint('hi')"})
    r = k.until("exec_result")
    assert r["ok"] and r["stdout"] == "hi\n"
    k.send({"op": "exec", "id": "b", "code": "x + 2"})
    r = k.until("exec_result")
    assert r["result"] == "42"
    k.close()


def test_errors_are_reported():
    k = Proc()
    k.send({"op": "exec", "id": "a", "code": "1/0"})
    r = k.until("exec_result")
    assert not r["ok"] and "ZeroDivisionError" in r["error"]
    k.close()


def test_tool_callable_round_trip():
    k = Proc()
    k.send({"op": "bootstrap", "tools": [{"id": "fs.read_file", "params": ["path"]}]})
    k.until("bootstrapped")
    k.send({"op": "exec", "id": "a", "code": "print(fs.read_file(path='a.txt'))"})

    def answer(m):
        assert m["type"] == "tool"
        assert m["payload"] == {"tool_id": "fs.read_file", "args": {"path": "a.txt"}}
        return {"op": "host_response", "id": m["id"], "ok": True, "result": {"ok": True, "result": "content!"}}

    r = k.until("exec_result", answer)
    assert r["stdout"] == "content!\n"
    k.close()


def test_tool_namespaces_are_importable():
    k = Proc()
    k.send({"op": "bootstrap", "tools": [{"id": "fs.read_file", "params": ["path"]}]})
    k.until("bootstrapped")
    k.send({"op": "exec", "id": "a", "code": "import fs\nfrom fs import read_file\nprint(read_file('a.txt'), fs.read_file is read_file)"})
    r = k.until("exec_result", lambda m: {"op": "host_response", "id": m["id"], "ok": True, "result": {"ok": True, "result": "hi"}})
    assert r["ok"], r
    assert r["stdout"] == "hi True\n"
    k.close()


def test_tool_error_raises_tool_error():
    k = Proc()
    k.send({"op": "bootstrap", "tools": [{"id": "fs.read_file", "params": ["path"]}]})
    k.until("bootstrapped")
    k.send({"op": "exec", "id": "a", "code": "try:\n    fs.read_file('x')\nexcept ToolError as e:\n    print('caught', e)"})
    r = k.until("exec_result", lambda m: {"op": "host_response", "id": m["id"], "ok": True, "result": {"ok": False, "error": "not allowed"}})
    assert "caught fs.read_file: not allowed" in r["stdout"]
    k.close()


def test_interrupt_runaway_loop():
    k = Proc()
    k.send({"op": "exec", "id": "a", "code": "while True:\n    pass"})
    time.sleep(0.5)
    k.p.send_signal(signal.SIGINT)
    r = k.until("exec_result")
    assert r["interrupted"]
    k.send({"op": "exec", "id": "b", "code": "1+1"})
    assert k.until("exec_result")["result"] == "2"
    k.close()
