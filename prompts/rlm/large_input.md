[The user's message is too large to show here in full: {{chars}} characters, about {{tokens}} tokens. All of it is in the REPL variable `user_message`.]

Work on it in the REPL instead of reading it here: look at its size and structure first (`len(user_message)`, `user_message[:2000]`), then search it (`re`, `str.find`, `splitlines()`) and read only the slices you need. For a long document, split it into parts and use sub-agents or several steps. Never print the whole variable.

Start of the message:
{{head}}

[... {{omitted}} characters not shown ...]

End of the message:
{{tail}}
