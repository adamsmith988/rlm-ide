"""Qdrant chat memory: chunking, scoping, fact ops, formatting, host-context sections."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Dict, List

import pytest

from src.config import settings
from src.memory import client as mc
from src.memory import facts as mf
from src.memory import format_memory_prompt_block
from src.memory.facts import FactOps, _parse
from src.orchestrators._shared.host_context import prompt_blocks, replace_host_context, split_host_context
from src.orchestrators._shared.text_chunks import chunk_text

_TOKEN = re.compile(r"\w+", re.U)
DIM = 64


def _vec(text: str) -> List[float]:
    v = [0.0] * DIM
    for tok in _TOKEN.findall(text.lower()):
        v[hash(tok) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed(texts: List[str]):
    return "fake-embed", [_vec(t) for t in texts]


def _matches(payload: Dict[str, Any], pid: str, flt: Dict[str, Any]) -> bool:
    for cond in flt.get("must") or []:
        if "has_id" in cond:
            if pid not in cond["has_id"]:
                return False
        elif payload.get(cond["key"]) != cond["match"]["value"]:
            return False
    return True


class FakeStore:
    prefix = "mem"

    def __init__(self):
        self.cols: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def collection_for(self, model, dim):
        return f"{self.prefix}__{model}__{dim}"

    async def collections(self):
        return sorted(self.cols)

    async def ensure(self, name, dim):
        self.cols.setdefault(name, {})

    async def upsert(self, name, points):
        for p in points:
            self.cols[name][p["id"]] = {"id": p["id"], "vector": p["vector"], "payload": dict(p["payload"])}

    async def search(self, name, vector, *, flt, limit, score_threshold=None):
        out = []
        for pid, p in self.cols.get(name, {}).items():
            if not _matches(p["payload"], pid, flt):
                continue
            score = sum(a * b for a, b in zip(vector, p["vector"]))
            if score_threshold is None or score >= score_threshold:
                out.append({"id": pid, "score": score, "payload": p["payload"]})
        out.sort(key=lambda h: h["score"], reverse=True)
        return out[:limit]

    async def scroll(self, name, *, flt, limit):
        return [
            {"id": pid, "payload": p["payload"]}
            for pid, p in self.cols.get(name, {}).items()
            if _matches(p["payload"], pid, flt)
        ][:limit]

    async def delete(self, name, *, flt):
        col = self.cols.get(name, {})
        for pid in [pid for pid, p in col.items() if _matches(p["payload"], pid, flt)]:
            del col[pid]


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.setattr(mc, "embed_texts", fake_embed)
    monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(settings, "MEMORY_MIN_SCORE", 0.05)
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", False)
    monkeypatch.setattr(settings, "MEMORY_CHUNK_CHARS", 200)
    monkeypatch.setattr(settings, "MEMORY_CHUNK_OVERLAP", 40)
    return mc.AdvancedMemoryClient(store=FakeStore())


def run(coro):
    return asyncio.run(coro)


# ---------------- chunking ----------------

def test_chunk_text_covers_everything_and_respects_cap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, max_chars=300, overlap=60, max_chunks=10)
    assert len(chunks) <= 10
    joined = " ".join(c.text for c in chunks)
    for i in (0, 777, 1999):
        assert f"word{i}" in joined


def test_chunk_overlap_keeps_boundary_fact_whole():
    filler = "lorem ipsum dolor sit amet. " * 20
    fact = "The vault code is 7741-ALPHA."
    text = filler + fact + " " + filler
    chunks = chunk_text(text, max_chars=300, overlap=80)
    assert any(fact in c.text for c in chunks)


# ---------------- scoping / recall ----------------

def test_middle_fact_in_long_message_is_recallable(memory):
    filler = "Quarterly logistics notes about trucks and routes. " * 60
    msg = filler + "The secret project codename is BLUE HERON. " + filler
    assert run(memory.save_conversation("u1", "s1", msg, "Noted."))
    got = run(memory.get_relevant_memories("what is the secret project codename", "u1", "s1"))
    assert any("BLUE HERON" in m["content"] for m in got["session"])


def test_short_turn_stored_as_one_question_answer_chunk(memory, monkeypatch):
    run(memory.save_conversation("u1", "s1", "What is 17 times 23?", "391"))
    rows = run(memory.get_all_session_memories("u1", "s1"))
    assert len(rows) == 1 and rows[0]["metadata"]["role"] == "turn"
    assert "17 times 23" in rows[0]["content"] and "391" in rows[0]["content"]
    block = format_memory_prompt_block({"session": rows})
    assert "- (user) What is 17 times 23? (assistant) 391" in block

    monkeypatch.setattr(settings, "MEMORY_PAIR_TURN_MAX_CHARS", 0)
    run(memory.save_conversation("u1", "s2", "What is 17 times 23?", "391"))
    roles = sorted(m["metadata"]["role"] for m in run(memory.get_all_session_memories("u1", "s2")))
    assert roles == ["assistant", "user"]


def test_session_isolation_and_user_isolation(memory):
    run(memory.save_conversation("u1", "s1", "My locker number is 4471.", "ok"))
    run(memory.save_conversation("u1", "s2", "The weather is mild today.", "ok"))
    run(memory.save_conversation("u2", "s9", "My locker number is 9999.", "ok"))
    s2 = run(memory.get_relevant_memories("locker number", "u1", "s2"))
    assert not any("4471" in m["content"] for m in s2["session"])
    u2 = run(memory.get_relevant_memories("locker number", "u2", "s9"))
    assert not any("4471" in m["content"] for m in u2["combined"])


def test_long_term_facts_cross_sessions_and_dedup_update(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_DEDUP_SCORE", 0.8)
    assert run(memory.add_memory("The user's favorite color is red", "u1", "s1", is_long_term=True))
    got = run(memory.get_relevant_memories("favorite color", "u1", "brand-new-session"))
    assert any("red" in m["content"] for m in got["user"])
    assert run(memory.add_memory("The user's favorite color is red indeed", "u1", "s3", is_long_term=True))
    facts = run(memory.get_all_user_memories("u1"))
    assert len(facts) == 1 and facts[0]["content"].endswith("indeed")
    hist = run(memory.get_memory_history(facts[0]["id"], user_id="u1"))
    assert hist and hist[0]["content"] == "The user's favorite color is red"


def test_fact_ops_apply_add_update_delete(memory):
    run(memory.add_memory("The user works at Acme", "u1", "s1", is_long_term=True))
    run(memory.add_memory("The user lives in Oslo", "u1", "s1", is_long_term=True))
    known = run(memory.get_all_user_memories("u1"))
    acme = next(k for k in known if "Acme" in k["content"])
    oslo = next(k for k in known if "Oslo" in k["content"])
    ops = FactOps(add=["The user prefers short answers"], update=[{"id": acme["id"], "fact": "The user works at Globex"}],
                  delete=[oslo["id"]])
    run(memory._apply_fact_ops("u1", "s1", ops))
    contents = sorted(m["content"] for m in run(memory.get_all_user_memories("u1")))
    assert contents == ["The user prefers short answers", "The user works at Globex"]


def test_background_fact_learning(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", True)

    async def fake_distill(user_message, assistant_response, known):
        return FactOps(add=["The user's name is Dara"])

    monkeypatch.setattr(mc, "distill_facts", fake_distill)

    async def go():
        await memory.save_conversation("u7", "s1", "Hi, I'm Dara.", "Hello Dara")
        await memory.drain()
        return await memory.get_all_user_memories("u7")

    facts = run(go())
    assert [f["content"] for f in facts] == ["The user's name is Dara"]


def test_oversized_message_is_material_not_a_fact_source(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", True)
    monkeypatch.setattr(settings, "MEMORY_FACT_MAX_USER_CHARS", 100)
    seen = []

    async def fake_distill(user_message, assistant_response, known):
        seen.append(user_message)
        return FactOps(add=["The user's station code is K99"])

    monkeypatch.setattr(mc, "distill_facts", fake_distill)

    async def go():
        await memory.save_conversation("u8", "s1", "The station code is K99. " + "filler " * 50, "K99")
        await memory.drain()
        return await memory.get_all_user_memories("u8"), await memory.get_all_session_memories("u8", "s1")

    facts, chunks = run(go())
    assert seen == [] and facts == [] and chunks


def test_fact_values_must_come_from_the_user_message(monkeypatch):
    from src.memory import facts as f

    user = "Remind me: my badge is E-4471 and I was born on ۱۲ May."
    assert f.grounded_in_user("The user's badge number is e-4471.", user)
    assert f.grounded_in_user("The user was born on 12 May.", user)
    assert f.grounded_in_user("The user prefers short answers.", user)
    assert not f.grounded_in_user("The user reported a renewal date of 14 March 2027.", user)

    class Reply:
        ok, text, error = True, "", ""

    reply = Reply()
    reply.text = '{"add": ["The user\'s badge is E-4471", "The dairy total is 231.5"], "update": []}'

    def fake_llm():
        async def call(*a, **kw):
            return reply
        return call

    monkeypatch.setattr(f, "_fact_llm", fake_llm)
    ops = run(f.distill_facts(user, "Dairy total: 231.5", []))
    assert ops.add == ["The user's badge is E-4471"]
    monkeypatch.setattr(settings, "MEMORY_FACT_REQUIRE_USER_GROUNDING", False)
    assert len(run(f.distill_facts(user, "Dairy total: 231.5", [])).add) == 2


def test_clear_session_keeps_long_term_and_clear_user_drops_all(memory):
    run(memory.save_conversation("u1", "s1", "Turn content alpha", "reply"))
    run(memory.add_memory("The user is a pilot", "u1", "s1", is_long_term=True))
    assert run(memory.clear_session_memories("u1", "s1"))
    assert run(memory.get_all_session_memories("u1", "s1")) == []
    assert len(run(memory.get_all_user_memories("u1"))) == 1
    assert run(memory.clear_user_memories("u1"))
    assert run(memory.get_all_user_memories("u1")) == []


def test_delete_scoped_to_owner(memory):
    run(memory.add_memory("The user is a pilot", "u1", "s1", is_long_term=True))
    mid = run(memory.get_all_user_memories("u1"))[0]["id"]
    run(memory.delete_memory(mid, user_id="intruder"))
    assert len(run(memory.get_all_user_memories("u1"))) == 1


def test_disabled_and_embed_failure_degrade_gracefully(memory, monkeypatch):
    async def broken(texts):
        raise mc.EmbeddingUnavailable("no model bound")

    monkeypatch.setattr(mc, "embed_texts", broken)
    res = run(memory.get_relevant_memories("anything", "u1", "s1"))
    assert res["combined"] == [] and "no model bound" in res["error"]
    assert run(memory.save_conversation("u1", "s1", "x", "y")) is False
    monkeypatch.setattr(settings, "MEMORY_ENABLED", False)
    assert run(memory.get_relevant_memories("anything", "u1", "s1")) == {"session": [], "user": [], "combined": []}


# ---------------- fact parsing ----------------

def test_parse_fact_ops_maps_refs_and_caps():
    known = [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
    ops = _parse({"add": ["n1", "", 3], "update": [{"ref": 2, "fact": "y2"}, {"ref": 9, "fact": "orphan"}],
                  "delete": [1, 5]}, known, max_items=5)
    assert ops.add == ["n1", "orphan"]
    assert ops.update == [{"id": "b", "fact": "y2"}]
    assert ops.delete == ["a"]


def test_fact_prompt_and_purpose_are_configurable(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_PROMPT", "CUSTOM PROMPT")
    assert mf.fact_prompt() == "CUSTOM PROMPT"
    monkeypatch.setattr(settings, "MEMORY_FACT_PROMPT", None)
    assert mf.fact_prompt() == mf.DEFAULT_FACT_PROMPT


# ---------------- prompt block ----------------

def test_prompt_block_budget_dedup_and_overrides(monkeypatch):
    mems = {
        "user": [{"content": "The user is Dara", "source": "user"}],
        "session": [{"content": "c" * 50, "metadata": {"role": "user"}}] * 3,
    }
    block = format_memory_prompt_block(mems, max_chars=2000)
    assert "The user is Dara" in block and block.count("c" * 50) == 1
    monkeypatch.setattr(settings, "MEMORY_PROMPT_HEADER", "## MEM")
    monkeypatch.setattr(settings, "MEMORY_PROMPT_USER_TITLE", "### U")
    assert format_memory_prompt_block(mems).startswith("## MEM\n\n### U")
    assert format_memory_prompt_block({"user": [], "session": []}) == ""


# ---------------- host context ----------------

def test_host_context_blocks_selection_and_replacement():
    kw = {"rag_prompt_block": "R1", "memory_prompt_block": "M1\n\nmulti-paragraph", "other": "x", "empty_prompt_block": " "}
    assert prompt_blocks(kw, "*") == {"rag": "R1", "memory": "M1\n\nmulti-paragraph"}
    assert prompt_blocks(kw, "memory") == {"memory": "M1\n\nmulti-paragraph"}
    assert prompt_blocks(kw, "") == {}
    sys1 = replace_host_context("BASE\n\n## Project tools\nT", prompt_blocks(kw))
    sys2 = replace_host_context(sys1, {"memory": "M2"})
    assert "M1" not in sys2 and "R1" not in sys2 and sys2.count("M2") == 1
    assert sys2.startswith("BASE\n\n## Project tools\nT")
    assert replace_host_context(sys2, {}) == "BASE\n\n## Project tools\nT"
    base, sections = split_host_context(sys1 + "\n\nPAGE")
    assert base == "BASE\n\n## Project tools\nT\n\nPAGE"
    assert "R1" in sections and "M1\n\nmulti-paragraph" in sections
    assert replace_host_context(base, {}) == base
    assert split_host_context("plain") == ("plain", "")


def test_store_ensure_uses_configured_indexes_and_segments(monkeypatch):
    from src.memory.store import QdrantMemoryStore

    monkeypatch.setattr(settings, "MEMORY_INDEXED_FIELDS", "user_id, session_id")
    monkeypatch.setattr(settings, "MEMORY_QDRANT_SEGMENTS", 2)

    def make(existing):
        calls: List[Any] = []
        store = QdrantMemoryStore("http://q", prefix="mem")

        async def fake_req(method, path, json=None, *, ok404=False):
            calls.append((method, path, json))
            if method == "GET":
                return existing
            return {}

        store._req = fake_req
        return store, calls

    store, calls = make({})
    asyncio.run(store.ensure("c1", 4))
    create = [c for c in calls if c[0] == "PUT" and c[1] == "/collections/c1"][0]
    assert create[2]["optimizers_config"] == {"default_segment_number": 2}
    assert [c[2]["field_name"] for c in calls if c[1].endswith("/index?wait=true")] == ["user_id", "session_id"]

    store, calls = make({"result": {"config": {"optimizer_config": {"default_segment_number": 0}}}})
    asyncio.run(store.ensure("c2", 4))
    assert ("PATCH", "/collections/c2", {"optimizers_config": {"default_segment_number": 2}}) in calls

    store, calls = make({"result": {"config": {"optimizer_config": {"default_segment_number": 2}}}})
    asyncio.run(store.ensure("c3", 4))
    asyncio.run(store.ensure("c3", 4))
    assert [c[0] for c in calls] == ["GET"]
