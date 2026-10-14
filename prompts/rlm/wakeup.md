[Scheduled wake-up #{{count}}] This message comes from the IDE, not from the user. You scheduled it at {{scheduled}}; it fired {{timing}}. Now: {{now}}.

Your note for this wake-up: {{reason}}

Your task board in the system message holds the goal, the original request, the steps with their status and your progress log. Continue the job from there:
1. Check the jobs you started: `shell.processes()`, their log files and the files they write. `shell.processes()` only lists processes started since the IDE last started; if a job is missing there, check its log and outputs and restart it (resumably) if it did not finish.
2. Update the board (`task.step`, `task.log`) with what you find.
3. Do the next step. If a job still needs time, schedule another wake-up with `task.wake_me` and end the turn with a short status.
4. When every step is done, give the user the final answer.
