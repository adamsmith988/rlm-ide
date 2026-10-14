"""
Agent Trace Repository
مدیریت trace های اجرای Agent در PostgreSQL

این ریپوزیتوری برای ذخیره و بازیابی:
- Traces: هر request یک trace
- Steps: مراحل اجرا (planner, executor, critic, etc.)
- Tool Executions: جزئیات اجرای ابزارها
- Stream Events: لاگ events برای replay
"""
from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

log = logging.getLogger(__name__)


def _parse_dt(val):
    """Convert ISO string to datetime object for asyncpg timestamptz params."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()


class TraceRepository:
    """Repository for agent execution traces."""
    
    def __init__(self, postgres):
        self.postgres = postgres
    
    # =========================================================================
    # TRACES
    # =========================================================================
    
    async def create_trace(
        self,
        trace_id: str,
        session_id: str,
        user_message: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a new agent trace."""
        if not self.postgres.pool:
            log.warning("PostgreSQL pool not available")
            return None
        
        try:
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO agent_traces (
                        trace_id, session_id, user_id, user_message, 
                        status, metadata
                    )
                    VALUES ($1, $2::uuid, $3, $4, 'started', $5::jsonb)
                    ON CONFLICT (trace_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        metadata = EXCLUDED.metadata
                    RETURNING id, trace_id, session_id, user_id, user_message,
                              complexity, plan_version, execution_plan, status,
                              final_answer, synthesis_reasoning, overall_confidence,
                              total_steps, total_tool_calls, replan_count,
                              error, started_at, completed_at, total_duration_ms,
                              metadata, created_at
                    """,
                    trace_id, session_id, user_id, user_message,
                    json.dumps(metadata or {})
                )
                
                if row:
                    log.debug(f"Created trace: {trace_id}")
                    return self._row_to_trace(row)
                
        except Exception as e:
            log.error(f"Error creating trace: {e}")
        
        return None
    
    async def update_trace(
        self,
        trace_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update an existing trace."""
        if not self.postgres.pool:
            return None
        
        # فیلدهای قابل به‌روزرسانی
        allowed_fields = {
            'status', 'complexity', 'plan_version', 'execution_plan',
            'final_answer', 'synthesis_reasoning', 'overall_confidence',
            'total_steps', 'total_tool_calls', 'replan_count',
            'error', 'error_details', 'completed_at', 'total_duration_ms'
        }
        
        # فیلتر کردن updates
        valid_updates = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
        
        if not valid_updates:
            return None
        
        try:
            # ساخت query dynamic
            set_clauses = []
            values = []
            param_idx = 1
            
            for field, value in valid_updates.items():
                if field in ['execution_plan', 'error_details']:
                    set_clauses.append(f"{field} = ${param_idx}::jsonb")
                    values.append(json.dumps(value) if isinstance(value, dict) else value)
                elif field == 'completed_at':
                    set_clauses.append(f"{field} = ${param_idx}::timestamptz")
                    values.append(_parse_dt(value))
                else:
                    set_clauses.append(f"{field} = ${param_idx}")
                    values.append(value)
                param_idx += 1
            
            values.append(trace_id)
            
            query = f"""
                UPDATE agent_traces
                SET {', '.join(set_clauses)}
                WHERE trace_id = ${param_idx}
                RETURNING id, trace_id, session_id, status, complexity,
                          total_steps, total_tool_calls, completed_at
            """
            
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)
                if row:
                    return dict(row)
                    
        except Exception as e:
            log.error(f"Error updating trace: {e}")
        
        return None
    
    async def add_feedback(
        self,
        trace_id: str,
        rating: int,
        comment: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add user feedback (thumbs up/down + comment) to a trace.
        
        Args:
            trace_id: ID of the trace
            rating: 1 (thumbs up), -1 (thumbs down), 0 (reset/neutral)
            comment: Optional text feedback
        """
        if not self.postgres.pool:
            return None
        
        try:
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    UPDATE agent_traces
                    SET user_feedback_rating = $2,
                        user_feedback_comment = $3,
                        user_feedback_at = now()
                    WHERE trace_id = $1
                    RETURNING trace_id, user_feedback_rating, user_feedback_comment, user_feedback_at
                    """,
                    trace_id, rating, comment
                )
                if row:
                    log.debug(f"Added feedback to trace {trace_id}: rating={rating}")
                    result = dict(row)
                    if result.get('user_feedback_at'):
                        result['user_feedback_at'] = result['user_feedback_at'].isoformat()
                    return result
                    
        except Exception as e:
            log.error(f"Error adding feedback to trace: {e}")
        
        return None
    
    async def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace by ID."""
        if not self.postgres.pool:
            return None
        
        try:
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, trace_id, session_id, user_id, user_message,
                           complexity, plan_version, execution_plan, status,
                           final_answer, synthesis_reasoning, overall_confidence,
                           total_steps, total_tool_calls, replan_count,
                           error, error_details, started_at, completed_at, 
                           total_duration_ms, metadata, created_at,
                           user_feedback_rating, user_feedback_comment, user_feedback_at
                    FROM agent_traces
                    WHERE trace_id = $1
                    """,
                    trace_id
                )
                
                if row:
                    return self._row_to_trace(row)
                    
        except Exception as e:
            log.error(f"Error getting trace: {e}")
        
        return None
    
    async def get_trace_full(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a trace with all steps and tool executions."""
        trace = await self.get_trace(trace_id)
        if not trace:
            return None
        
        trace["steps"] = await self.list_steps(trace_id)
        trace["tool_executions"] = await self.list_tool_executions(trace_id)
        
        return trace
    
    async def list_traces(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List traces with optional filters."""
        if not self.postgres.pool:
            return []
        
        try:
            conditions = []
            values = []
            param_idx = 1
            
            if session_id:
                conditions.append(f"session_id = ${param_idx}::uuid")
                values.append(session_id)
                param_idx += 1
            
            if user_id:
                conditions.append(f"user_id = ${param_idx}")
                values.append(user_id)
                param_idx += 1
            
            if status:
                conditions.append(f"status = ${param_idx}")
                values.append(status)
                param_idx += 1
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            values.extend([limit, offset])
            
            query = f"""
                SELECT id, trace_id, session_id, user_id, 
                       LEFT(user_message, 100) as user_message,
                       complexity, status, overall_confidence,
                       total_steps, total_tool_calls, replan_count,
                       total_duration_ms, started_at, completed_at, created_at
                FROM agent_traces
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            
            async with self.postgres.pool.acquire() as conn:
                rows = await conn.fetch(query, *values)
                return [dict(row) for row in rows]
                
        except Exception as e:
            log.error(f"Error listing traces: {e}")
        
        return []
    
    # =========================================================================
    # TRACE STEPS
    # =========================================================================
    
    async def add_step(
        self,
        trace_id: str,
        step_number: int,
        node_name: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Add a new step to a trace."""
        if not self.postgres.pool:
            return None
        
        try:
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO agent_trace_steps (
                        trace_id, step_number, node_name, task_id, task_description,
                        status, input_data, plan_generated, planning_reasoning,
                        tool_calls_requested, llm_messages_sent, started_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7::jsonb, $8::jsonb, $9,
                        $10::jsonb, $11::jsonb, $12::timestamptz
                    )
                    RETURNING id, trace_id, step_number, node_name, task_id, status,
                              started_at, created_at
                    """,
                    trace_id, step_number, node_name,
                    kwargs.get("task_id"),
                    kwargs.get("task_description"),
                    kwargs.get("status", "started"),
                    json.dumps(kwargs.get("input_data")) if kwargs.get("input_data") else None,
                    json.dumps(kwargs.get("plan_generated")) if kwargs.get("plan_generated") else None,
                    kwargs.get("planning_reasoning"),
                    json.dumps(kwargs.get("tool_calls_requested")) if kwargs.get("tool_calls_requested") else None,
                    json.dumps(kwargs.get("llm_messages_sent")) if kwargs.get("llm_messages_sent") else None,
                    _parse_dt(kwargs.get("started_at")) or datetime.utcnow()
                )
                
                if row:
                    return dict(row)
                    
        except Exception as e:
            log.error(f"Error adding step: {e}")
        
        return None
    
    async def update_step(
        self,
        trace_id: str,
        step_number: int,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update a step."""
        if not self.postgres.pool:
            return None
        
        allowed_fields = {
            'status', 'input_data', 'output_data', 'plan_generated', 'planning_reasoning',
            'tool_calls_requested', 'evaluation', 'confidence_score',
            'needs_retry', 'needs_replan', 'synthesis_content',
            'llm_messages_sent', 'llm_response', 'llm_tokens_prompt', 'llm_tokens_completion',
            'error', 'error_details', 'completed_at', 'duration_ms'
        }
        
        # JSONB fields need special handling
        jsonb_fields = {
            'input_data', 'output_data', 'plan_generated', 'tool_calls_requested',
            'evaluation', 'llm_messages_sent', 'llm_response', 'error_details'
        }
        
        valid_updates = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
        
        if not valid_updates:
            return None
        
        try:
            set_clauses = []
            values = []
            param_idx = 1
            
            for field, value in valid_updates.items():
                if field in jsonb_fields:
                    set_clauses.append(f"{field} = ${param_idx}::jsonb")
                    values.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
                elif field == 'completed_at':
                    set_clauses.append(f"{field} = ${param_idx}::timestamptz")
                    values.append(_parse_dt(value))
                else:
                    set_clauses.append(f"{field} = ${param_idx}")
                    values.append(value)
                param_idx += 1
            
            values.extend([trace_id, step_number])
            
            query = f"""
                UPDATE agent_trace_steps
                SET {', '.join(set_clauses)}
                WHERE trace_id = ${param_idx} AND step_number = ${param_idx + 1}
                RETURNING id, trace_id, step_number, status, completed_at, duration_ms
            """
            
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)
                if row:
                    return dict(row)
                    
        except Exception as e:
            log.error(f"Error updating step: {e}")
        
        return None
    
    async def list_steps(
        self,
        trace_id: str,
        node_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all steps of a trace."""
        if not self.postgres.pool:
            return []
        
        try:
            query = """
                SELECT id, trace_id, step_number, node_name, task_id, task_description,
                       status, input_data, output_data, plan_generated, planning_reasoning,
                       tool_calls_requested, evaluation, confidence_score,
                       needs_retry, needs_replan, synthesis_content,
                       llm_messages_sent, llm_response, llm_tokens_prompt, llm_tokens_completion,
                       error, error_details, started_at, completed_at, duration_ms, created_at
                FROM agent_trace_steps
                WHERE trace_id = $1
            """
            values = [trace_id]
            
            if node_name:
                query += " AND node_name = $2"
                values.append(node_name)
            
            query += " ORDER BY step_number ASC"
            
            async with self.postgres.pool.acquire() as conn:
                rows = await conn.fetch(query, *values)
                return [self._row_to_step(row) for row in rows]
                
        except Exception as e:
            log.error(f"Error listing steps: {e}")
        
        return []
    
    # =========================================================================
    # TOOL EXECUTIONS
    # =========================================================================
    
    async def add_tool_execution(
        self,
        trace_id: str,
        tool_name: str,
        arguments: Dict,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Add a tool execution record."""
        if not self.postgres.pool:
            return None
        
        try:
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO agent_tool_executions (
                        trace_id, step_id, tool_name, tool_id, task_id,
                        arguments, status, file_path, file_type,
                        started_at, retry_count
                    )
                    VALUES (
                        $1, $2::uuid, $3, $4, $5,
                        $6::jsonb, $7, $8, $9,
                        $10::timestamptz, $11
                    )
                    RETURNING id, trace_id, tool_name, tool_id, task_id, status,
                              started_at, created_at
                    """,
                    trace_id,
                    kwargs.get("step_id"),
                    tool_name,
                    kwargs.get("tool_id"),
                    kwargs.get("task_id"),
                    json.dumps(arguments),
                    kwargs.get("status", "pending"),
                    kwargs.get("file_path"),
                    kwargs.get("file_type"),
                    _parse_dt(kwargs.get("started_at")) or datetime.utcnow(),
                    kwargs.get("retry_count", 0)
                )
                
                if row:
                    return dict(row)
                    
        except Exception as e:
            log.error(f"Error adding tool execution: {e}")
        
        return None
    
    async def update_tool_execution(
        self,
        trace_id: str,
        tool_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update a tool execution."""
        if not self.postgres.pool:
            return None
        
        allowed_fields = {
            'status', 'result', 'result_summary', 'result_sent_to_llm',
            'error', 'completed_at', 'duration_ms', 'retry_count'
        }
        
        valid_updates = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
        
        if not valid_updates:
            return None
        
        try:
            set_clauses = []
            values = []
            param_idx = 1
            
            for field, value in valid_updates.items():
                if field in ['result', 'result_sent_to_llm']:
                    set_clauses.append(f"{field} = ${param_idx}::jsonb")
                    values.append(json.dumps(value) if isinstance(value, dict) else value)
                elif field == 'completed_at':
                    set_clauses.append(f"{field} = ${param_idx}::timestamptz")
                    values.append(_parse_dt(value))
                else:
                    set_clauses.append(f"{field} = ${param_idx}")
                    values.append(value)
                param_idx += 1
            
            values.extend([trace_id, tool_id])
            
            query = f"""
                UPDATE agent_tool_executions
                SET {', '.join(set_clauses)}
                WHERE trace_id = ${param_idx} AND tool_id = ${param_idx + 1}
                RETURNING id, trace_id, tool_id, status, completed_at, duration_ms
            """
            
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)
                if row:
                    return dict(row)
                    
        except Exception as e:
            log.error(f"Error updating tool execution: {e}")
        
        return None
    
    async def list_tool_executions(
        self,
        trace_id: str,
        task_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List tool executions for a trace."""
        if not self.postgres.pool:
            return []
        
        try:
            query = """
                SELECT id, trace_id, step_id, tool_name, tool_id, task_id,
                       arguments, status, result, result_summary, result_sent_to_llm,
                       error, file_path, file_type,
                       started_at, completed_at, duration_ms, retry_count, created_at
                FROM agent_tool_executions
                WHERE trace_id = $1
            """
            values = [trace_id]
            
            if task_id:
                query += " AND task_id = $2"
                values.append(task_id)
            
            query += " ORDER BY created_at ASC"
            
            async with self.postgres.pool.acquire() as conn:
                rows = await conn.fetch(query, *values)
                return [self._row_to_tool_exec(row) for row in rows]
                
        except Exception as e:
            log.error(f"Error listing tool executions: {e}")
        
        return []
    
    # =========================================================================
    # STREAM EVENTS
    # =========================================================================
    
    async def add_stream_event(
        self,
        trace_id: str,
        event_type: str,
        event_data: Dict,
        sequence_number: int,
        timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Add a stream event for replay."""
        if not self.postgres.pool:
            return None
        
        try:
            async with self.postgres.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO agent_stream_events (
                        trace_id, event_type, event_data, sequence_number, timestamp
                    )
                    VALUES ($1, $2, $3::jsonb, $4, $5::timestamptz)
                    RETURNING id, trace_id, event_type, sequence_number, timestamp
                    """,
                    trace_id, event_type, json.dumps(event_data),
                    sequence_number, _parse_dt(timestamp) or datetime.utcnow()
                )
                
                if row:
                    return dict(row)
                    
        except Exception as e:
            log.error(f"Error adding stream event: {e}")
        
        return None
    
    async def list_stream_events(
        self,
        trace_id: str,
        after_sequence: int = 0,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """List stream events for replay."""
        if not self.postgres.pool:
            return []
        
        try:
            async with self.postgres.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, trace_id, event_type, event_data, sequence_number, timestamp
                    FROM agent_stream_events
                    WHERE trace_id = $1 AND sequence_number > $2
                    ORDER BY sequence_number ASC
                    LIMIT $3
                    """,
                    trace_id, after_sequence, limit
                )
                
                return [self._row_to_event(row) for row in rows]
                
        except Exception as e:
            log.error(f"Error listing stream events: {e}")
        
        return []
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    async def batch_write(self, items: List[Dict]) -> Dict[str, int]:
        """Write multiple items in a batch."""
        if not self.postgres.pool:
            return {"success": 0, "failed": len(items)}
        
        success = 0
        failed = 0
        
        for item in items:
            item_type = item.get("type")
            data = item.get("data", {})
            
            try:
                if item_type == "trace_update":
                    await self.update_trace(data.pop("trace_id"), **data)
                    success += 1
                elif item_type == "step":
                    await self.add_step(
                        data.pop("trace_id"),
                        data.pop("step_number"),
                        data.pop("node_name"),
                        **data
                    )
                    success += 1
                elif item_type == "step_update":
                    await self.update_step(
                        data.pop("trace_id"),
                        data.pop("step_number"),
                        **data
                    )
                    success += 1
                elif item_type == "tool_execution":
                    await self.add_tool_execution(
                        data.pop("trace_id"),
                        data.pop("tool_name"),
                        data.pop("arguments"),
                        **data
                    )
                    success += 1
                elif item_type == "tool_execution_update":
                    await self.update_tool_execution(
                        data.pop("trace_id"),
                        data.pop("tool_id"),
                        **data
                    )
                    success += 1
                elif item_type == "stream_event":
                    await self.add_stream_event(
                        data.get("trace_id"),
                        data.get("event_type"),
                        data.get("event_data"),
                        data.get("sequence_number"),
                        data.get("timestamp")
                    )
                    success += 1
                else:
                    log.warning(f"Unknown batch item type: {item_type}")
                    failed += 1
                    
            except Exception as e:
                log.error(f"Batch item failed: {e}")
                failed += 1
        
        return {"success": success, "failed": failed}
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _row_to_trace(self, row) -> Dict[str, Any]:
        """Convert DB row to trace dict."""
        result = dict(row)
        
        # Parse JSON fields
        for field in ['execution_plan', 'error_details', 'metadata']:
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except:
                    pass
        
        # Format datetimes (including user_feedback_at)
        for field in ['started_at', 'completed_at', 'created_at', 'user_feedback_at']:
            if result.get(field):
                result[field] = result[field].isoformat() if hasattr(result[field], 'isoformat') else str(result[field])
        
        return result
    
    def _row_to_step(self, row) -> Dict[str, Any]:
        """Convert DB row to step dict."""
        result = dict(row)
        
        # Parse JSON fields
        json_fields = [
            'input_data', 'output_data', 'plan_generated', 'tool_calls_requested',
            'evaluation', 'llm_messages_sent', 'llm_response', 'error_details'
        ]
        for field in json_fields:
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except:
                    pass
        
        # Format datetimes
        for field in ['started_at', 'completed_at', 'created_at']:
            if result.get(field):
                result[field] = result[field].isoformat() if hasattr(result[field], 'isoformat') else str(result[field])
        
        return result
    
    def _row_to_tool_exec(self, row) -> Dict[str, Any]:
        """Convert DB row to tool execution dict."""
        result = dict(row)
        
        # Parse JSON fields
        for field in ['arguments', 'result', 'result_sent_to_llm']:
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except:
                    pass
        
        # Format datetimes
        for field in ['started_at', 'completed_at', 'created_at']:
            if result.get(field):
                result[field] = result[field].isoformat() if hasattr(result[field], 'isoformat') else str(result[field])
        
        return result
    
    def _row_to_event(self, row) -> Dict[str, Any]:
        """Convert DB row to stream event dict."""
        result = dict(row)
        
        if result.get('event_data') and isinstance(result['event_data'], str):
            try:
                result['event_data'] = json.loads(result['event_data'])
            except:
                pass
        
        if result.get('timestamp'):
            result['timestamp'] = result['timestamp'].isoformat() if hasattr(result['timestamp'], 'isoformat') else str(result['timestamp'])
        
        return result
