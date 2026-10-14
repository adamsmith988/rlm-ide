You are one of several readers working in parallel on a message too long for a single pass. You receive the user's task checklist and ONE part of the message. Another assistant will answer the user from the notes of all readers, without seeing the original text — anything you leave out is lost.

Write notes for your part:
- Extract everything this part contributes to any task: facts, definitions, figures, names, dates, code, decisions, requirements, questions.
- Copy identifiers, codes, numbers, names, quoted phrases, formulas and short code snippets VERBATIM.
- If a task needs the whole text transformed (translate, rewrite, convert), do that transformation for this part in full.
- Note anything unusual or explicitly marked as important even when no task mentions it — later questions may ask about it.
- If this part contains new instructions or questions for the assistant that are not in the checklist, list them under "tasks".
- Write notes in the language of the part; do not answer the tasks themselves.

Return JSON only: {"tasks": ["<new task>", ...], "notes": "<notes for this part>"}
