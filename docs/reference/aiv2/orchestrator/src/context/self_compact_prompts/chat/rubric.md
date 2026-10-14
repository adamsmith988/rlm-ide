You are about to decide whether to compress your conversation history into a summary that REPLACES the full history above. After compression, you continue from only [system, latest_user_message, assistant_summary, user_continue]. Compression is irreversible: anything not preserved in the summary is gone.

Compression is safe ONLY when ALL FOUR of the following hold:
(C1) the trajectory has reached a closed unit (not mid-thought),
(C2) the essential information is reducible to 3–5 cite-able facts without loss,
(C3) something has progressed since the last compression,
(N1) you are NOT currently stuck in a way summarization would mask.

Answer C1, C2, C3, N1 honestly. Each Y answer requires verbatim evidence quoted from the trajectory above; answers without evidence default to N.

C1 CLOSED-UNIT: The most recent assistant message is a closed unit — a finished answer to the user, or a completed tool call whose result is now visible. It is NOT mid-sentence reasoning ("Let me now check...", "I should next look at..."), and not a half-formulated tool call. If Y, quote the closing fragment of the last assistant message. If N, quote the open fragment that shows the trajectory is mid-thought.

C2 SUMMARIZABLE: You can write 3–5 essential facts (with verbatim citations from the trajectory) that future-you needs to continue the conversation after compression. Each fact must be a single concrete statement: a name, number, identifier, decision, user preference, or resolved request. Answer N if the conversation's value is dispersed across many small details (e.g., a long list of values the user gave, constraints, or ruled-out options) that would be lost without the dispersal. If Y, list the 3–5 facts numbered, each with a verbatim citation in quotes, separated by " | ". If N, name in one sentence the class of information that would be lost.

C3 PROGRESS: Since the most recent compression (or since the start of the conversation if none), a request has been answered OR a new concrete fact (name, number, identifier, decision, or quoted claim) has been established. If Y, name the answered request or the new fact. If N, state that you are returning the same state you compressed from.

N1 STUCK: At least 3 of your last 4 tool calls or answer attempts produced no new fact (i.e., were duplicates, errors, or returned already-known content). If you have made fewer than 4 such attempts, answer N. If Y, name 1 distinct strategy you have NOT yet tried. If N, name one new fact obtained recently.

This instruction replaces any earlier instruction about how to reply; do not answer earlier messages and do not call tools.

Output: exactly 4 lines, no preamble or trailing text.
C1: Y/N -- <evidence>
C2: Y/N -- <if Y: 1. fact "citation" | 2. fact "citation" | 3. fact "citation"; if N: <class of info lost>>
C3: Y/N -- <evidence>
N1: Y/N -- <evidence>
