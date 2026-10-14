{domain_brief}

Grade the ASSISTANT ANSWER to the request against EACH rubric criterion.
A criterion passes only when the answer clearly satisfies it; an empty,
truncated or error answer fails every criterion. Tool runs are listed for
context: an answer may rely on them only if their content supports it.
An answer that breaks an operator instruction below fails every criterion
about that instruction, however good it is otherwise.

Return STRICT JSON:
{{"verdicts": [{{"id": "c1", "pass": true | false, "why": "short reason"}}, ...]}}

=== OPERATOR INSTRUCTIONS FOR THIS CHAT (may be none) ===
{instructions}

=== REQUEST ===
{request}

=== RUBRIC ===
{rubric}

=== TOOL RUNS ===
{tool_runs}

=== ASSISTANT ANSWER ===
{answer}
