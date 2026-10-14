You read the beginning and the end of a very long message a user sent to an assistant. The middle is not shown.

List every task, question, instruction and constraint the user gives the assistant, so none is forgotten while the full text is processed in parts. Include output-format requirements ("reply with code only", "answer in English", "use a table") as tasks.

Return JSON only: {"tasks": ["<task>", ...]}
- One short, self-contained task per item, in the user's own language.
- Keep exact identifiers, names, numbers and quoted phrases.
- Do not answer the tasks. Do not invent tasks; when the visible text only provides material without instructions, return the single task of responding to the message as a whole.
