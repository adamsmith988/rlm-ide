CREATE TABLE IF NOT EXISTS chats (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  user_text TEXT NOT NULL,
  assistant_text TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',
  events TEXT NOT NULL DEFAULT '[]',
  UNIQUE (chat_id, idx)
);
CREATE INDEX IF NOT EXISTS turns_chat ON turns(chat_id, idx);

CREATE TABLE IF NOT EXISTS pinned_facts (
  id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS pinned_chat ON pinned_facts(chat_id);

-- Memory rows. Vectors live in per-model tables named orchestrator_memory__<model>__<dim>.
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT,
  memory_type TEXT NOT NULL,
  kind TEXT NOT NULL,
  category TEXT NOT NULL,
  workspace_id TEXT,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  history TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS memory_user ON memory(user_id);
CREATE INDEX IF NOT EXISTS memory_session ON memory(session_id);
CREATE INDEX IF NOT EXISTS memory_type ON memory(memory_type);
CREATE INDEX IF NOT EXISTS memory_kind ON memory(kind);
CREATE INDEX IF NOT EXISTS memory_category ON memory(category);
CREATE INDEX IF NOT EXISTS memory_workspace ON memory(workspace_id);

CREATE TABLE IF NOT EXISTS edits (
  edit_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  path TEXT NOT NULL,
  before_text TEXT NOT NULL,
  after_text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  state TEXT NOT NULL
);
