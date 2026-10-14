/** Rubric-based LLM judge over an OpenAI-compatible endpoint (EVAL_JUDGE_URL / EVAL_JUDGE_MODEL / EVAL_JUDGE_KEY). */
export async function judge({ task, rubric, files, render, answer }) {
  const base = (process.env.EVAL_JUDGE_URL ?? "http://localhost:8100/v1").replace(/\/+$/, "");
  const model = process.env.EVAL_JUDGE_MODEL ?? "Qwen3.5-4B";
  const fileText = files
    .map((f) => `--- ${f.path} (${f.content.length} chars)\n${f.content.slice(0, 6000)}${f.content.length > 6000 ? "\n[truncated]" : ""}`)
    .join("\n\n")
    .slice(0, 40000);
  const renderText = render
    ? `Rendered page: loaded=${render.loaded}, console errors=${JSON.stringify(render.console ?? []).slice(0, 800)}, buttons=${render.counts?.buttons}, fonts=${JSON.stringify(render.fonts ?? [])}, background=${render.bodyBg}\nVisible text:\n${(render.text ?? "").slice(0, 1500)}\nInteraction probe: ${JSON.stringify(render.probe ?? null).slice(0, 1500)}`
    : "No rendered page.";
  const prompt = `You are a strict senior reviewer grading work produced by an AI coding agent.

TASK GIVEN TO THE AGENT:
${task}

RUBRIC (score each 1-5, 5 = excellent, 3 = acceptable with clear gaps, 1 = missing/broken):
${rubric.map((r, i) => `${i + 1}. ${r}`).join("\n")}

AGENT'S FINAL MESSAGE:
${answer.slice(0, 2000)}

${renderText}

FILES PRODUCED:
${fileText}

Reply with ONLY a JSON object: {"scores": [<one integer per rubric item>], "issues": ["<concrete problem>", ...], "summary": "<one sentence>"}`;
  const headers = { "Content-Type": "application/json" };
  if (process.env.EVAL_JUDGE_KEY) headers.Authorization = `Bearer ${process.env.EVAL_JUDGE_KEY}`;
  try {
    const res = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        temperature: 0,
        max_tokens: 1200,
        chat_template_kwargs: { enable_thinking: false },
      }),
    });
    const j = await res.json();
    const text = j.choices?.[0]?.message?.content ?? "";
    const m = /\{[\s\S]*\}/.exec(text);
    const parsed = JSON.parse(m ? m[0] : text);
    const scores = rubric.map((_, i) => Math.max(1, Math.min(5, Number(parsed.scores?.[i] ?? 1))));
    return { scores, mean: scores.reduce((a, b) => a + b, 0) / scores.length, issues: parsed.issues ?? [], summary: parsed.summary ?? "", model };
  } catch (e) {
    return { scores: [], mean: null, issues: [`judge failed: ${e}`], summary: "", model };
  }
}
