"""Kernel environment awareness for RLM sessions.

Real runs burned whole turns on ``import openpyxl`` → ``ModuleNotFoundError``
→ ``import xlrd`` → … before the model fell back to ``zipfile`` + XML. The host
probes which optional packages exist and, when pip is allowed, how to install
missing ones into the session userbase.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.orchestrators.rlm.runtime.session_pip import pip_install_allowed

KERNEL_ENV_MARKER = "__RLM_KERNEL_ENV__"

# Optional third-party packages worth advertising. Keep short: this lands in
# the system prompt of every run.
PROBE_PACKAGES: List[str] = [
    "openpyxl",
    "xlrd",
    "xlsxwriter",
    "pandas",
    "numpy",
    "PIL",
    "cv2",
    "pytesseract",
    "docx",
    "pptx",
    "pypdf",
    "pdfplumber",
    "fitz",
    "reportlab",
    "bs4",
    "lxml",
    "yaml",
    "requests",
    "httpx",
    "dateutil",
    "phonenumbers",
    "chardet",
    "tabulate",
    "markdown",
    "matplotlib",
    "sklearn",
    "scipy",
    "sqlalchemy",
]

# Stdlib fallbacks for the packages the model most often reaches for.
STDLIB_FALLBACKS: Dict[str, str] = {
    "openpyxl": ".xlsx = zip: `zipfile` + `xml.etree.ElementTree` (xl/sharedStrings.xml, xl/worksheets/sheet*.xml)",
    "xlrd": ".xlsx = zip: `zipfile` + `xml.etree.ElementTree`; legacy .xls is not parseable here",
    "xlsxwriter": "write .csv with `csv`, or Markdown tables",
    "pandas": "`csv`, `json`, `sqlite3`, `collections.Counter`, `statistics`",
    "numpy": "`statistics`, `math`, plain lists",
    "docx": ".docx = zip: `zipfile` + `xml.etree.ElementTree` (word/document.xml)",
    "pptx": ".pptx = zip: `zipfile` + `xml.etree.ElementTree` (ppt/slides/slide*.xml)",
    "pypdf": "PDF text is not extractable without a library; report that limitation",
    "pdfplumber": "PDF text is not extractable without a library; report that limitation",
    "fitz": "PDF text is not extractable without a library; report that limitation",
    "PIL": "images: `await rlm.look('file.jpg')` shows them to you natively — do not decode pixels",
    "cv2": "images: `await rlm.look('file.jpg')` shows them to you natively — do not decode pixels",
    "pytesseract": "OCR: `await rlm.look('file.jpg')` and read the image yourself",
    "bs4": "`html.parser` from the stdlib (`html.parser.HTMLParser`) or `re`",
    "lxml": "`xml.etree.ElementTree` / `html.parser`",
    "yaml": "`json`, or parse simple YAML with `re`",
    "requests": "`urllib.request` (network may be jailed — do not retry on failure)",
    "httpx": "`urllib.request` (network may be jailed — do not retry on failure)",
    "dateutil": "`datetime.strptime` with explicit formats",
    "phonenumbers": "normalize numbers with `re` (strip spaces/dashes, keep leading +/country code)",
    "chardet": "try `utf-8`, then `utf-8-sig`, `cp1256`, `latin-1` with `errors='replace'`",
    "tabulate": "hand-write Markdown tables",
    "markdown": "write Markdown text directly",
    "matplotlib": "describe the distribution in a table instead of plotting",
}

# Emitted by the probe; parsed by the host.
KERNEL_ENV_PROBE_CODE = (
    "import importlib.util as _rlm_iu, json as _rlm_json, sys as _rlm_sys\n"
    f"_rlm_pkgs = {json.dumps(PROBE_PACKAGES)}\n"
    "_rlm_avail = []\n"
    "for _rlm_p in _rlm_pkgs:\n"
    "    try:\n"
    "        if _rlm_iu.find_spec(_rlm_p) is not None:\n"
    "            _rlm_avail.append(_rlm_p)\n"
    "    except Exception:\n"
    "        pass\n"
    "_rlm_pip = _rlm_iu.find_spec('pip') is not None\n"
    f"print({KERNEL_ENV_MARKER!r} + _rlm_json.dumps({{'python': _rlm_sys.version.split()[0], "
    "'available': _rlm_avail, 'pip': _rlm_pip}))\n"
    "del _rlm_iu, _rlm_json, _rlm_sys, _rlm_pkgs, _rlm_avail, _rlm_p, _rlm_pip\n"
)

_MODULE_NOT_FOUND_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_\.]+)['\"]")


def parse_kernel_env_probe(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON payload printed by :data:`KERNEL_ENV_PROBE_CODE`."""
    text = str(stdout or "")
    idx = text.rfind(KERNEL_ENV_MARKER)
    if idx < 0:
        return None
    raw = text[idx + len(KERNEL_ENV_MARKER) :].strip().splitlines()
    if not raw:
        return None
    try:
        data = json.loads(raw[0])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    avail = [str(x) for x in (data.get("available") or []) if x]
    return {
        "python": str(data.get("python") or ""),
        "available": avail,
        "missing": [p for p in PROBE_PACKAGES if p not in set(avail)],
        "pip": bool(data.get("pip")),
        "pip_install_allowed": pip_install_allowed(),
    }


def format_kernel_env_notice(env: Optional[Dict[str, Any]]) -> str:
    """System-prompt section: what is importable, what is not, and what to use instead."""
    if not isinstance(env, dict):
        return ""
    avail = [str(x) for x in (env.get("available") or [])]
    missing = [str(x) for x in (env.get("missing") or [])]
    py = str(env.get("python") or "").strip()
    allow_pip = bool(env.get("pip_install_allowed", pip_install_allowed()))
    has_pip = bool(env.get("pip"))
    lines = ["## Kernel environment (probed — trust this, do not re-check with imports)"]
    if py:
        lines.append(
            f"- Python {py}; full stdlib available (zipfile, csv, json, sqlite3, re, "
            "xml.etree, html.parser, statistics, datetime)."
        )
    if avail:
        lines.append("- Installed third-party: " + ", ".join(avail) + ".")
    else:
        lines.append("- No optional third-party packages are installed yet.")
    if allow_pip and has_pip:
        lines.append(
            "- `pip install` is ALLOWED into this session "
            "(`python -m pip install --user <pkg>` or `%%bash` with the same). "
            "Packages land in the session userbase and persist for this run."
        )
    elif allow_pip and not has_pip:
        lines.append(
            "- Session pip installs are allowed, but the `pip` module is missing "
            "from this interpreter — ask the host to provide pip."
        )
    if missing:
        if allow_pip and has_pip:
            lines.append(
                "- Not preinstalled (install with pip --user if needed): "
                + ", ".join(missing)
                + "."
            )
        else:
            pip_note = (
                "`pip` exists but installs are disabled for this kernel"
                if has_pip
                else "`pip install` is unavailable in this kernel"
            )
            lines.append(f"- NOT installed ({pip_note}): " + ", ".join(missing) + ".")
    hot = [m for m in ("openpyxl", "pandas", "docx", "PIL", "pypdf") if m in missing]
    if hot and not (allow_pip and has_pip):
        lines.append("- Fallbacks you must use instead of retrying imports:")
        for m in hot:
            lines.append(f"  - {m}: {STDLIB_FALLBACKS[m]}")
    if allow_pip and has_pip:
        lines.append(
            "- Prefer one `pip install --user` then import — do not burn turns on "
            "`try: import X` probes for packages listed above."
        )
    else:
        lines.append(
            "- Each failed import costs a whole turn. Never write `try: import X` probes "
            "for packages listed above; go straight to the fallback."
        )
    return "\n".join(lines)


def module_not_found_hint(result_text: str) -> str:
    """Return a ``[env]`` hint line when an ipython result died on a missing module."""
    m = _MODULE_NOT_FOUND_RE.search(str(result_text or ""))
    if not m:
        return ""
    mod = m.group(1).split(".")[0]
    fallback = STDLIB_FALLBACKS.get(mod)
    if pip_install_allowed():
        base = (
            f"[env] `{mod}` is not installed yet. Install with "
            f"`python -m pip install --user {mod}` (session userbase), then import again."
        )
        if fallback:
            return base + f" Stdlib fallback if install fails: {fallback}."
        return base
    base = (
        f"[env] `{mod}` is not installed in this kernel and cannot be installed "
        "(pip installs disabled). Do not import it again."
    )
    if fallback:
        return base + f" Use instead: {fallback}."
    return base + " Use the standard library instead."
