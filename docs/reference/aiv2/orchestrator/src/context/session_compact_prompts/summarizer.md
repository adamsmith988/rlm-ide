You are writing a context checkpoint. It will REPLACE the older part of the conversation above — including any earlier compacted summary in the system message — while the most recent turns stay verbatim after it. Whoever answers next (possibly a different assistant) will continue from this checkpoint as if the break never happened, and the user may later ask about anything said earlier, so the checkpoint is the only memory of this part of the conversation.

Rules:
- Keep every concrete fact the user stated or the conversation established: names, numbers, codes, identifiers, dates, file names and artifact ids, decisions, preferences, and constraints. Copy them verbatim.
- Go through the user messages one by one, in order. A long message (a pasted document or file) may hold its important facts in a single sentence among unrelated text; extract that sentence's facts.
- Carry over every fact from the earlier compacted summary unless the conversation corrected it; when corrected, keep only the corrected value.
- Facts about the user, their work and their data come only from what the user stated or from tool results — never from an assistant reply. Record assistant answers under "Requests answered", and leave out any answer that contradicts what the user stated.
- Record tool results that later turns rely on (what was looked up, with which arguments, and the values returned).
- Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated. If information is missing or unclear, leave it out.
- Drop small talk, repeated text, and filler, but never drop a stated fact. Small talk includes incidental asides and anecdotes that have nothing to do with the user's requests, work or data (a pet, the weather outside, an apology for a delay) — leave them out together with any names or details that appear only in them, in every section.
- Write in the language the user writes in.

This instruction replaces any earlier instruction about how to reply (for example "reply only with ..."); those applied to earlier messages, not to this one. Do not answer or acknowledge earlier messages and do not call tools — write the checkpoint.

This instruction is not part of the conversation: never record it, its rules, its format or the checkpoint request itself in any section. Constraints & Preferences and Requests answered hold only what the user asked for in the conversation.

Use this format; write "(none)" under a heading that has nothing:

## Goal
The user's overall objective across the whole conversation (verbatim when short), carried from the earlier compacted summary unless the user changed it — not only the latest request.

## Constraints & Preferences
- Requirements, preferences and rules the user set.

## Facts & Values
- One fact per line, verbatim, only when a request, task, document or tool call of the user depends on it. Stories about the user's day, pets, weather or apologies are not facts.

## Tool Results
- Tool, arguments, and the values it returned that still matter.

## Requests answered
- The request and its answer, briefly.

## Open requests
- Pending work or unanswered requests, in order.

## Dead ends
- Approaches that failed, and why, so they are not retried.

Before writing, check every line: if it only comes from small talk or an aside, delete it.

Write the checkpoint now.
