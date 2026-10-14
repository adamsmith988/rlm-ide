---
name: python-style
description: Hints for Python modules
globs:
  - "*.py"
alwaysApply: false
---

- Prefer explicit type hints on public functions and class methods.
- Use `pathlib.Path` instead of string paths when reasonable.
- Keep functions focused; extract helpers when a block grows long.
