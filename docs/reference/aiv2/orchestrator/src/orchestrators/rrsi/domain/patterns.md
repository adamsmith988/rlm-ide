# Pattern library (levers by component)

- prompt: clarify the procedure the policy should follow (decompose, check
  the request's constraints, state assumptions). Cheapest lever; try first
  when the failure is a judgment or habit problem.
- output_plumbing: fix answer shape problems (missing sections, wrong
  language, no evidence cited, unrequested verbosity).
- config: engine settings such as temperature or thinking; use when traces
  show sampling noise or truncated reasoning.
- control_flow: step budget when tool loops end early or run away; one
  bounded retry when trials fail on transient errors.
- context_mgmt: shrink history when old turns distract, grow it when the
  answer ignores established context.
- client_tool: expose a tool class the policy needed but did not have, or
  remove tools it misuses.
- skill: a reusable procedure with an explicit trigger condition for a
  recurring kind of request.
- memory: a short, entity-free working note about how this user's requests
  should be handled (preferences on format/language/rigor).
- subagent: a single bounded reviewer pass when answers contain fixable
  errors the policy does not catch itself.
