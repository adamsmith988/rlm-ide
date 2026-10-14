import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.utils.registry_wrapper import ToolRegistryWrapper


class _FakeRegistry:
    """Minimal stand-in for modules.tools.registry.ToolRegistry."""

    def __init__(self) -> None:
        self.list_calls = 0
        self.tools = {
            "embed_face": {"description": "faces", "category": "face"},
            "vision_info": {"description": "vision", "category": "image"},
        }

    def list(self):
        self.list_calls += 1
        return dict(self.tools)

    def reload(self, root=None, recursive=True, extra_roots=None):
        self.tools = {"only_tool": {"description": "after reload", "category": "general"}}


def _ready_wrapper() -> tuple[ToolRegistryWrapper, _FakeRegistry]:
    wrapper = ToolRegistryWrapper(tools_path="/nonexistent/tools", plugins_path="")
    fake = _FakeRegistry()
    wrapper._registry = fake
    wrapper._initialized = True
    return wrapper, fake


def test_list_tools_is_cached_until_invalidated() -> None:
    wrapper, fake = _ready_wrapper()

    first = wrapper.list_tools()
    second = wrapper.list_tools()
    assert first is second
    assert set(first) == {"embed_face", "vision_info"}
    assert fake.list_calls == 1

    # get_tool (hot path on every execute) must not rebuild the snapshot.
    assert wrapper.get_tool("embed_face") is first["embed_face"]
    assert wrapper.get_tool("missing") is None
    assert fake.list_calls == 1

    wrapper.invalidate_tools_cache()
    third = wrapper.list_tools()
    assert third is not first
    assert fake.list_calls == 2


def test_reload_invalidates_list_tools_cache() -> None:
    wrapper, fake = _ready_wrapper()
    assert set(wrapper.list_tools()) == {"embed_face", "vision_info"}

    assert wrapper.reload() is True
    assert set(wrapper.list_tools()) == {"only_tool"}
    assert wrapper.get_tool("embed_face") is None
    assert wrapper.get_tool("only_tool").description == "after reload"


def test_reload_failure_still_invalidates_cache() -> None:
    wrapper, fake = _ready_wrapper()
    wrapper.list_tools()

    def boom(**_kwargs):
        raise RuntimeError("disk gone")

    fake.reload = boom
    assert wrapper.reload() is False
    assert wrapper._tools_cache is None


def test_list_tools_not_cached_when_not_ready() -> None:
    wrapper = ToolRegistryWrapper(tools_path="/nonexistent/tools", plugins_path="")
    assert wrapper.list_tools() == {}
    assert wrapper._tools_cache is None
