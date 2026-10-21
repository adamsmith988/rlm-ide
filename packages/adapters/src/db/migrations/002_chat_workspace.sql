-- Chats belong to the project folder they were started in.
ALTER TABLE chats ADD COLUMN workspace_id TEXT;

-- Existing chats: the folder their turns were saved under in short-term memory.
UPDATE chats SET workspace_id = (
  SELECT m.workspace_id FROM memory m
  WHERE m.session_id = chats.id AND m.workspace_id IS NOT NULL
  ORDER BY m.created_at DESC LIMIT 1
);

CREATE INDEX IF NOT EXISTS idx_chats_workspace ON chats(workspace_id, updated_at);
