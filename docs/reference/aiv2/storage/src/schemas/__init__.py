"""
Database Schemas
اسکیماهای دیتابیس PostgreSQL
"""

# Core schema for PostgreSQL
CORE_SCHEMA = """
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Media type enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'media_kind') THEN
        CREATE TYPE media_kind AS ENUM (
            'image',
            'text',
            'audio',
            'video',
            'document',
            'other'
        );
    END IF;
END$$;

-- Persons table
CREATE TABLE IF NOT EXISTS persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT UNIQUE,
    full_name TEXT NOT NULL,
    aliases JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Locations table
CREATE TABLE IF NOT EXISTS locations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Folders table
CREATE TABLE IF NOT EXISTS folders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id UUID REFERENCES folders(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT folders_unique_name_per_parent UNIQUE (parent_id, name)
);
CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_id);

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    media_type media_kind NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    external_ref TEXT,
    checksum TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    folder_id UUID REFERENCES folders(id) ON DELETE SET NULL,
    original_filename TEXT,
    file_size_bytes BIGINT,
    mime_type TEXT,
    -- 🆕 ارتباط با کاربر و جلسه چت
    user_id UUID,  -- کاربری که فایل را آپلود کرده
    session_id UUID,  -- جلسه چتی که فایل در آن آپلود شده
    ingested_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_artifacts_folder ON artifacts(folder_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_media_type ON artifacts(media_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_user ON artifacts(user_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_updated_at ON artifacts(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_size ON artifacts(file_size_bytes);
CREATE INDEX IF NOT EXISTS idx_artifacts_mime ON artifacts(mime_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_storage_path_pattern ON artifacts(storage_path text_pattern_ops);
CREATE INDEX IF NOT EXISTS idx_artifacts_filename_trgm ON artifacts USING GIN (original_filename gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_artifacts_storage_path_trgm ON artifacts USING GIN (storage_path gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_artifacts_tags_gin ON artifacts USING GIN ((metadata->'tags'));
CREATE INDEX IF NOT EXISTS idx_artifacts_checksum ON artifacts(checksum) WHERE checksum IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_artifacts_import_locator ON artifacts (user_id, (metadata ->> 'case_importer_rel_path'), (metadata ->> 'case_importer_filename'))
    WHERE COALESCE(metadata ->> 'case_importer_rel_path', '') <> '';

-- ============================================================================
-- Group-Scoped File Access (RBAC for artifacts)
-- Minimal subset required by file_manager access logic.
-- ============================================================================

CREATE TABLE IF NOT EXISTS access_groups (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    parent_id   UUID REFERENCES access_groups(id) ON DELETE SET NULL,
    metadata    JSONB DEFAULT '{}'::jsonb,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ag_parent ON access_groups(parent_id);
CREATE INDEX IF NOT EXISTS idx_ag_name   ON access_groups(name);

CREATE TABLE IF NOT EXISTS group_memberships (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id   UUID NOT NULL,
    group_id  UUID NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    role_name TEXT NOT NULL DEFAULT 'user',
    joined_at TIMESTAMPTZ DEFAULT now(),
    added_by  TEXT,
    UNIQUE(user_id, group_id)
);
CREATE INDEX IF NOT EXISTS idx_gm_user  ON group_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_gm_group ON group_memberships(group_id);

CREATE TABLE IF NOT EXISTS group_files (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id    UUID NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    granted_at  TIMESTAMPTZ DEFAULT now(),
    granted_by  TEXT,
    UNIQUE(group_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_gf_group    ON group_files(group_id);
CREATE INDEX IF NOT EXISTS idx_gf_artifact ON group_files(artifact_id);

CREATE TABLE IF NOT EXISTS user_file_overrides (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    permissions TEXT[] NOT NULL DEFAULT ARRAY['read'],
    reason      TEXT,
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT now(),
    created_by  TEXT,
    UNIQUE(user_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_ufo_user     ON user_file_overrides(user_id);
CREATE INDEX IF NOT EXISTS idx_ufo_artifact ON user_file_overrides(artifact_id);

-- ============================================================================
-- Secure artifact share tokens (short-lived, auditable)
-- ============================================================================

CREATE TABLE IF NOT EXISTS artifact_share_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash TEXT NOT NULL UNIQUE,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    created_by UUID NOT NULL,
    scope JSONB DEFAULT '{}'::jsonb,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ast_artifact  ON artifact_share_tokens(artifact_id);
CREATE INDEX IF NOT EXISTS idx_ast_creator   ON artifact_share_tokens(created_by);
CREATE INDEX IF NOT EXISTS idx_ast_expires   ON artifact_share_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_ast_revoked   ON artifact_share_tokens(revoked_at);

CREATE TABLE IF NOT EXISTS artifact_share_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_id UUID REFERENCES artifact_share_tokens(id) ON DELETE SET NULL,
    action TEXT NOT NULL, -- resolve|download|revoke|create|deny|expired
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_asa_token   ON artifact_share_audit(token_id);
CREATE INDEX IF NOT EXISTS idx_asa_action  ON artifact_share_audit(action);
CREATE INDEX IF NOT EXISTS idx_asa_created ON artifact_share_audit(created_at DESC);

-- Case importer: per-user default tool selection (null allowlist = all tools)
CREATE TABLE IF NOT EXISTS case_importer_user_preferences (
    user_id TEXT PRIMARY KEY,
    tool_allowlist JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Artifact faces table
CREATE TABLE IF NOT EXISTS artifact_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    face_key TEXT NOT NULL,
    bbox JSONB,
    confidence REAL,
    person_id UUID REFERENCES persons(id),
    tag TEXT,
    thumbnail_base64 TEXT,
    vector_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT unique_artifact_face UNIQUE (artifact_id, face_key)
);
CREATE INDEX IF NOT EXISTS idx_artifact_faces_person ON artifact_faces(person_id);

-- Artifact images table
CREATE TABLE IF NOT EXISTS artifact_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE UNIQUE,
    width_px INTEGER,
    height_px INTEGER,
    dpi INTEGER,
    color_mode TEXT,
    camera_model TEXT,
    lens TEXT,
    iso INTEGER,
    exposure TEXT,
    focal_length TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Artifact audio table
CREATE TABLE IF NOT EXISTS artifact_audio (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE UNIQUE,
    duration_sec REAL,
    sample_rate INTEGER,
    channels INTEGER,
    codec TEXT,
    bitrate INTEGER,
    transcription TEXT,
    language TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Artifact audio segments
CREATE TABLE IF NOT EXISTS artifact_audio_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    speaker_id TEXT,
    start_sec REAL,
    end_sec REAL,
    text TEXT,
    confidence REAL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audio_segments_artifact ON artifact_audio_segments(artifact_id);

-- Note: tool_runs table has been REMOVED - tool executions are stored in agent_tool_executions table

-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT,
    rating TEXT NOT NULL,
    comment TEXT,
    message_text TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Person-Artifact relations
CREATE TABLE IF NOT EXISTS person_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'subject',
    metadata JSONB DEFAULT '{}'::jsonb,
    folder_id UUID REFERENCES folders(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT unique_person_artifact UNIQUE (person_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_person_artifacts_person ON person_artifacts(person_id);
CREATE INDEX IF NOT EXISTS idx_person_artifacts_artifact ON person_artifacts(artifact_id);

-- ============================================================================
-- LLM Control Plane (super-admin, model registry, endpoint discovery)
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_external_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    scheme TEXT NOT NULL DEFAULT 'http',
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    base_path TEXT NOT NULL DEFAULT '',
    auth JSONB DEFAULT '{}'::jsonb,          -- e.g. {"type":"bearer","token":"..."} (store secret elsewhere in production)
    capabilities JSONB DEFAULT '{}'::jsonb,  -- discovery output (protocols, models, endpoints)
    last_probe JSONB DEFAULT '{}'::jsonb,    -- last probe summary (latency/error)
    -- Deprecated leftover: RPM is stored per credential (endpoint+logical_id).
    requests_per_minute INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_external_endpoints_active ON llm_external_endpoints(is_active);
ALTER TABLE llm_external_endpoints
  ADD COLUMN IF NOT EXISTS requests_per_minute INTEGER;

-- Multiple bearer tokens for one model on one endpoint address.
-- Every row is model-scoped (logical_id required). Endpoint-wide tokens are not stored here;
-- legacy endpoint.auth.token remains only as a discover/fallback auth field.
CREATE TABLE IF NOT EXISTS llm_endpoint_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    endpoint_id UUID NOT NULL REFERENCES llm_external_endpoints(id) ON DELETE CASCADE,
    label TEXT NOT NULL DEFAULT 'default',
    token TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'model',  -- always model (endpoint-wide rows are purged)
    logical_id TEXT,
    upstream_model_id TEXT,
    priority INTEGER NOT NULL DEFAULT 1000,
    weight INTEGER NOT NULL DEFAULT 1,
    -- Per-token window uses this value; Save RPM sets the same number on every
    -- credential matching endpoint + logical_id (or upstream_model_id).
    requests_per_minute INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    last_probe JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
-- Idempotent for DBs created before requests_per_minute existed.
ALTER TABLE llm_endpoint_credentials
  ADD COLUMN IF NOT EXISTS requests_per_minute INTEGER;
CREATE INDEX IF NOT EXISTS idx_llm_endpoint_credentials_ep ON llm_endpoint_credentials(endpoint_id);
-- Uniqueness for LB: same bearer must not register twice for the same model on
-- an endpoint. Label is display-only.
DROP INDEX IF EXISTS idx_llm_endpoint_credentials_ep_label;
DROP INDEX IF EXISTS idx_llm_endpoint_credentials_ep_label_scope;
CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_endpoint_credentials_ep_model_token
  ON llm_endpoint_credentials (
    endpoint_id,
    lower(COALESCE(NULLIF(btrim(logical_id), ''), '')),
    token
  );
-- Credentials table is model-only: drop legacy whole-endpoint rows.
DELETE FROM llm_endpoint_credentials
 WHERE lower(COALESCE(scope, '')) <> 'model'
    OR logical_id IS NULL
    OR btrim(logical_id) = '';
ALTER TABLE llm_endpoint_credentials ALTER COLUMN scope SET DEFAULT 'model';

CREATE TABLE IF NOT EXISTS llm_local_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,               -- model alias used by orchestrator/tools
    backend_kind TEXT NOT NULL,              -- kserve|external
    kserve_namespace TEXT,
    kserve_inferenceservice TEXT,            -- e.g. qwen-chat
    external_endpoint_id UUID REFERENCES llm_external_endpoints(id) ON DELETE SET NULL,
    external_model_id TEXT,                  -- e.g. upstream model id for OpenAI-compatible providers
    version TEXT,                            -- optional semantic version / tag
    metadata JSONB DEFAULT '{}'::jsonb,      -- includes dispatch_key, api_protocol, modalities
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_local_models_active ON llm_local_models(is_active);

CREATE TABLE IF NOT EXISTS llm_routes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    route_key TEXT NOT NULL UNIQUE,          -- e.g. chat.default, tool.vision_info.llm
    -- Platform-wide bindings: no user_id / tenant_id. Super-admin writes via
    -- api-gateway; all users share the same resolved model on the next call.
    -- model_name can be either a physical row name (llm_local_models.name) OR a
    -- logical_id (metadata.logical_id grouping multiple backends). The router
    -- resolves both, so we keep this column unconstrained for flexibility.
    model_name TEXT NOT NULL,
    fallback_model_name TEXT,
    constraints JSONB DEFAULT '{}'::jsonb,   -- e.g. {"tenant":"X","max_cost":...}
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_routes_active ON llm_routes(is_active);

CREATE TABLE IF NOT EXISTS llm_model_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    storage_path TEXT NOT NULL UNIQUE,
    category TEXT,
    name TEXT,
    model_name TEXT,
    deploy_status TEXT NOT NULL DEFAULT 'discovered',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_model_candidates_status ON llm_model_candidates(deploy_status);
CREATE INDEX IF NOT EXISTS idx_llm_model_candidates_category ON llm_model_candidates(category);

CREATE TABLE IF NOT EXISTS llm_pool_policies (
    logical_id TEXT PRIMARY KEY,
    canonical_model_key TEXT,
    min_replicas INTEGER NOT NULL DEFAULT 1,
    max_replicas INTEGER NOT NULL DEFAULT 3,
    target_free_vram_mb INTEGER NOT NULL DEFAULT 4096,
    idle_scale_down_sec INTEGER NOT NULL DEFAULT 1800,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_llm_pool_policies_enabled ON llm_pool_policies(enabled);

CREATE TABLE IF NOT EXISTS llm_nodes (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    models_root TEXT,
    has_gpu BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- node_id on llm_local_models (per-node deployments; same logical_id on many nodes)
ALTER TABLE llm_local_models ADD COLUMN IF NOT EXISTS node_id TEXT DEFAULT 'default';
UPDATE llm_local_models SET node_id = 'default' WHERE node_id IS NULL;
ALTER TABLE llm_local_models ALTER COLUMN node_id SET NOT NULL;
ALTER TABLE llm_local_models DROP CONSTRAINT IF EXISTS llm_local_models_name_key;
ALTER TABLE llm_local_models DROP CONSTRAINT IF EXISTS llm_local_models_node_name_key;
ALTER TABLE llm_local_models ADD CONSTRAINT llm_local_models_node_name_key UNIQUE (node_id, name);
CREATE INDEX IF NOT EXISTS idx_llm_local_models_node ON llm_local_models(node_id);

-- ── Migration (idempotent): drop legacy FK on existing deployments so the
-- column can hold a logical_id that no physical row matches yet.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'llm_routes_model_name_fkey'
    ) THEN
        ALTER TABLE llm_routes DROP CONSTRAINT llm_routes_model_name_fkey;
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'llm_routes_fallback_model_name_fkey'
    ) THEN
        ALTER TABLE llm_routes DROP CONSTRAINT llm_routes_fallback_model_name_fkey;
    END IF;
END$$;
"""

# Auth & Chat Schema
AUTH_CHAT_SCHEMA = """
-- ============================================================================
-- USER ROLES & PERMISSIONS
-- ============================================================================

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    permissions JSONB DEFAULT '[]'::jsonb,
    is_system BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Insert default roles
INSERT INTO roles (name, description, permissions, is_system) VALUES
    ('super-admin', 'مدیر ارشد سیستم - دسترسی کامل', '["*"]', TRUE),
    ('admin', 'مدیر - مدیریت کاربران و تنظیمات', '["users.read", "users.write", "chats.read", "chats.write", "settings.read", "settings.write"]', TRUE),
    ('user', 'کاربر عادی - دسترسی به چت خود', '["chats.own.read", "chats.own.write", "profile.read", "profile.write"]', TRUE)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- USERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keycloak_id TEXT UNIQUE,
    username TEXT NOT NULL UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    display_name TEXT,
    avatar_url TEXT,
    role_id UUID REFERENCES roles(id),
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMPTZ,
    settings JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_keycloak ON users(keycloak_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- User activity log
CREATE TABLE IF NOT EXISTS user_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity_log(user_id);

-- ============================================================================
-- CHAT SESSIONS & MESSAGES
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT DEFAULT 'چت جدید',
    model TEXT,
    system_prompt TEXT,
    is_archived BOOLEAN DEFAULT FALSE,
    is_pinned BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at DESC);

-- Drop legacy hardcoded default model name on existing deployments.
DO $$
BEGIN
    ALTER TABLE chat_sessions ALTER COLUMN model DROP DEFAULT;
EXCEPTION
    WHEN OTHERS THEN
        NULL;
END $$;

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_calls JSONB,
    tool_results JSONB,
    thinking TEXT,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    attachments JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages(created_at);

-- ============================================================================
-- TOOL / PLUGIN SESSIONS (Postgres only — never MinIO blobs / case evidence)
-- ============================================================================

CREATE TABLE IF NOT EXISTS tool_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plugin_id TEXT NOT NULL,
    tool_id TEXT,
    record_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    status TEXT,
    ok BOOLEAN,
    error TEXT,
    output_preview TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, plugin_id, record_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_sessions_user ON tool_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_tool_sessions_plugin_record
    ON tool_sessions(plugin_id, record_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_sessions_user_plugin_record
    ON tool_sessions(user_id, plugin_id, record_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_sessions_run ON tool_sessions(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_sessions_payload_case
    ON tool_sessions ((payload->>'case_id'));
CREATE INDEX IF NOT EXISTS idx_tool_sessions_payload_conv_case
    ON tool_sessions ((payload->'conversation'->>'case_id'));

-- ============================================================================
-- REFRESH TOKENS
-- ============================================================================

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    device_info JSONB DEFAULT '{}'::jsonb,
    ip_address TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires ON refresh_tokens(expires_at);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_chat_sessions_updated_at ON chat_sessions;
CREATE TRIGGER update_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tool_sessions_updated_at ON tool_sessions;
CREATE TRIGGER update_tool_sessions_updated_at
    BEFORE UPDATE ON tool_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""

__all__ = ["CORE_SCHEMA", "AUTH_CHAT_SCHEMA"]
