import json
import os
import subprocess
import sys

WORKER = os.path.join(os.path.dirname(__file__), "..", "worker.py")
SRC = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(20))


class Kernel:
    def __init__(self, cwd):
        self.p = subprocess.Popen([sys.executable, "-u", WORKER], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        assert self.read()["op"] == "ready"
        self.send({"op": "bootstrap", "tools": [], "output_limit": 16000, "rewrite_guard": {"min_lines": 30, "max_changed": 0.3}})
        assert self.until("bootstrapped")

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def read(self):
        return json.loads(self.p.stdout.readline())

    def until(self, op):
        while True:
            m = self.read()
            if m["op"] == op:
                return m

    def run(self, code):
        self.send({"op": "exec", "id": "c", "code": code})
        return self.until("exec_result")

    def close(self):
        self.send({"op": "shutdown"})
        self.p.wait(timeout=5)


def small_fix(src, n):
    return src.replace("return 3\n", f"return {300 + n}\n")


def test_small_fix_rewrites_warn_then_refuse(tmp_path):
    (tmp_path / "train.py").write_text(SRC)
    k = Kernel(tmp_path)
    r = k.run(f"with open('train.py', 'w') as f:\n    f.write({small_fix(SRC, 1)!r})")
    assert r["ok"], r
    assert "changed only 1 of" in r["stderr"] and "fs.edit_file" in r["stderr"]
    assert "return 301" in (tmp_path / "train.py").read_text()

    r = k.run(f"open('train.py', 'w').write({small_fix(SRC, 2)!r})\nprint('after')")
    assert not r["ok"]
    assert "not written" in r["error"] and 'fs.edit_file(path="train.py"' in r["error"]
    assert "return 301" in (tmp_path / "train.py").read_text()

    r = k.run(f"import pathlib\npathlib.Path('train.py').write_text({small_fix(SRC, 3)!r})")
    assert not r["ok"] and "not written" in r["error"]
    assert "return 301" in (tmp_path / "train.py").read_text()
    k.close()


def test_real_rewrites_new_files_and_data_files_are_untouched(tmp_path):
    (tmp_path / "train.py").write_text(SRC)
    (tmp_path / "results.json").write_text("\n".join(str(i) for i in range(50)))
    k = Kernel(tmp_path)
    new_src = "\n".join(f"x{i} = {i}" for i in range(60))
    r = k.run(f"open('train.py', 'w').write({new_src!r})\nprint(open('train.py').read().count('x'))")
    assert r["ok"] and r["stdout"].strip() == "60", r
    for n in range(3):
        r = k.run(f"open('results.json', 'w').write({chr(10).join(str(i + (n if i == 7 else 0)) for i in range(50))!r})")
        assert r["ok"], r
    r = k.run("open('fresh.py', 'w').write('print(1)\\n' * 40)\nopen('fresh.py', 'a').write('# more\\n')")
    assert r["ok"], r
    assert (tmp_path / "fresh.py").read_text().endswith("# more\n")
    k.close()


def test_one_liner_write_is_visible_in_the_same_cell(tmp_path):
    (tmp_path / "app.py").write_text(SRC)
    k = Kernel(tmp_path)
    r = k.run(f"open('app.py', 'w').write({small_fix(SRC, 1)!r})\nprint('return 301' in open('app.py').read())")
    assert r["ok"] and r["stdout"].strip() == "True", r
    k.close()
