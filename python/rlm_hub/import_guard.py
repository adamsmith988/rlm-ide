"""Treat packages that crash while importing as not installed.

Environments that reuse the base Python's site-packages can carry a package built for another torch or
missing its native libraries; it then raises RuntimeError/OSError on import. Libraries guard optional
dependencies with `except ImportError`, so the crash is re-raised as ImportError for top-level packages.
"""
import importlib.abc
import sys

_failed = {}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        try:
            self._inner.exec_module(module)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            _failed[self._name] = f"{type(e).__name__}: {e}"
            print(f"[rlm_hub] {self._name} failed to load ({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}); treating it as not installed", file=sys.stderr, flush=True)
            raise ImportError(f"{self._name} failed to load: {e}", name=self._name) from e

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if "." in name:
            return None
        if name in _failed:
            raise ImportError(f"{name} failed to load: {_failed[name]}", name=name)
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module") and spec.origin not in (None, "built-in", "frozen"):
                spec.loader = _Loader(spec.loader, name)
            return spec
        return None


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
