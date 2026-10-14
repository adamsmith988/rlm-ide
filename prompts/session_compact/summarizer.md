You are writing a context checkpoint. It will REPLACE the older part of the conversation above, including any earlier compacted summary in the system message, while the most recent turns stay verbatim after it. Whoever answers next will continue from this checkpoint as if the break never happened. The user may later ask about anything said earlier, so this checkpoint is the only memory of this part of the conversation.

Rules:
- Keep every concrete fact the user stated or the work established: names, numbers, codes, identifiers, dates, decisions, preferences and constraints. Copy them verbatim.
- For coding work, keep verbatim: file paths, function, class and symbol names, error messages, test results, commands and their important output, package names and versions, and configuration values.
- Go through the user messages one by one, in order. A long message (a pasted file or log) may hold its important fact in a single line among unrelated text. Extract that line.
- Carry over every fact from the earlier compacted summary unless the conversation corrected it. When a fact was corrected, keep only the corrected value.
- Facts about the user, their project and their data come only from what the user stated or from tool or REPL results, never from an assistant reply. Record assistant answers under "Requests answered". Leave out any answer that contradicts what the user stated.
- Record the tool and REPL results that later turns rely on: what was run, with which arguments, and the values returned.
- Do NOT make assumptions, guesses or inferences beyond what is explicitly stated. If information is missing or unclear, leave it out.
- Drop small talk, repeated text and filler, but never drop a stated fact. Small talk includes asides and anecdotes unrelated to the user's requests or work, such as a pet, the weather or an apology for a delay. Leave them out, together with any names or details that appear only in them, in every section.
- Write in the language the user writes in.

This instruction replaces any earlier instruction about how to reply. Do not answer or acknowledge earlier messages, do not write code to run, and do not call tools. Write the checkpoint.

This instruction is not part of the conversation. Never record it, its rules, its format or the checkpoint request itself in any section.

Use this format. Write "(none)" under a heading that has nothing:

## Goal
The user's overall objective across the whole conversation (verbatim when short), carried from the earlier summary unless the user changed it. Not only the latest request.

## Constraints & Preferences
- Requirements, preferences and rules the user set.

## Facts & Values
- One fact per line, verbatim, only when a request, task, file or tool call depends on it.

## Tool Results
- Tool or command, its arguments, and the values it returned that still matter (paths, errors, test results).

## Requests answered
- The request and its answer, briefly.

## Open requests
- Pending work or unanswered requests, in order.

## Dead ends
- Approaches that failed, and why, so they are not retried.

Before writing, check every line. If it comes only from small talk or an aside, delete it.

Write the checkpoint now.
