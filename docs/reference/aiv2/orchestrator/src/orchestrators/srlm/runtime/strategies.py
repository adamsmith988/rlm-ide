#
# Prompt texts and post-processing below are taken from
# https://github.com/apple-aiml-research/ml-selfreflect
# (src/selfreflect/generate_summaries.py), adapted from tokenized vLLM prompts
# to chat messages for an OpenAI-compatible gateway.
#
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Use, reproduction, modification and redistribution are subject to the Apple
# license terms reproduced in ../NOTICE.
#
"""SelfReflect summary strategies (paper Sec. 4 / repo ``--summary_strats``).

Each strategy turns a question (plus, for the sampling strategies, K answers
sampled from the same model) into the user prompt of one chat completion. The
controller decides sampling parameters; this module owns only prompt text and
the strategy-specific post-processing.
"""

from __future__ import annotations

import re
from typing import List, Optional

GREEDY = "greedy"
BASIC = "basic"
SAMPLE_AND_SUMMARIZE = "sample_and_summarize"
CONCATENATED_ANSWERS = "concatenated_answers"
COT = "cot"
TWOSTEP_COT = "twostep_cot"

STRATEGIES = (
    GREEDY,
    BASIC,
    SAMPLE_AND_SUMMARIZE,
    CONCATENATED_ANSWERS,
    COT,
    TWOSTEP_COT,
)
# Strategies conditioned on the sampled answer distribution.
NEEDS_SAMPLES = frozenset({SAMPLE_AND_SUMMARIZE, CONCATENATED_ANSWERS})
# Strategies whose output is produced without a summary LLM call.
NO_LLM = frozenset({CONCATENATED_ANSWERS})
# ``--enable_thinking_for_summaries`` applies only to these (generate_summaries.py).
THINKING_ELIGIBLE = frozenset({BASIC, SAMPLE_AND_SUMMARIZE})

_SUMMARY_LINE = re.compile(r"(?i)summary:\s*(.*)", re.S)


def numbered_answers(answers: List[str]) -> str:
    return "\n".join(f"x_{i + 1} = '{a}'" for i, a in enumerate(answers))


def greedy_prompt(question: str) -> str:
    return f"{question}"


def basic_prompt(question: str) -> str:
    return f"""Please respond to the following question '{question}'.

Your goal is to summarize all possible answers to this question:
* If there are multiple possible answers, the summarized answer should mention the main possible answers. However, you do not have to list possibilities that are too unlikely.
* If some possibilities are more likely than others, delineate which possibilities are more more likely by using words like "most likely" and "could also be".
* The format of the summarized answer should be the same as a normal answer.
* If there is only clear answer to the question, just provide that answer, without hedging across possibilities.

Please provide the summarized answer."""


def sample_and_summarize_prompt(question: str, answers: List[str]) -> str:
    n_answers = len(answers)
    listed = numbered_answers(answers)
    return f"""Below, you are given {n_answers} individual answers to the question '{question}' that were given by different people.

Your goal is to summarize the {n_answers} answers into one concise summarized answer.
* The summarized answer should mention the main possibility mentioned in the {n_answers} answers, i.e. the one that is mentioned most often.
* The summarized answer should also mention further possibilites mentioned in the {n_answers}.
* If some possibilities are mentioned more often than others, delineate which possibilities are more often found in the others by using words like "most likely" and "could also be".
* If individual answers differ in numeric values, aggregate them into ranges. For example, if two answers say '20', one answer says '22' and another one '24', a good summary would be to give a range from 20 to 24.
* The format of the summarized answer should be the same as each individual answer. Provide only the answer, as if it were part of the {n_answers} answers, without statements like "The answers include...".
* The summarized answer should not use synonyms for the original wording used in individual answers.
* The summarized answer should reflect what the {n_answers} answers deem possible. They can contain factually wrong options. Do not correct those, just report the possibilities as they are given in the answers.

Here are the {n_answers} answers:
{listed}

Please provide the summarized answer."""


def concatenated_answers(answers: List[str]) -> str:
    return numbered_answers(answers)


def cot_prompt(question: str) -> str:
    return f"""
Please respond to the following question: '{question}'.

Your goal is to **reason about multiple plausible answers** to this question, not just provide a single answer (unless there is only one possible answer):
* First, reflect explicitly on whether there are multiple reasonable answers to this question. Consider ambiguity, interpretation, and context.
* Then, explore at least 2–3 **distinct answer possibilities**, especially if the question allows it.
* Clearly indicate if some possibilities are more likely than others using words like "most likely", "could also be", or "another possible interpretation is".
* You do not need to include very unlikely answers.
* After reasoning, **summarize the best answer(s)** in a way that reflects the uncertainty or multiplicity, while still sounding natural.
* If there is only one clearly correct answer, explain why it is the only plausible one, and just give that.

Use the following format:
Reasoning: [Detailed reasoning about the different possible answers and their likelihoods.]
Summary: [Stand-alone answer reflecting the reasoning.]

Make sure to show multiple possibilities if relevant before committing to the final summary.
"""


def cot_analysis_prompt(question: str) -> str:
    """Two-step CoT, step 1."""
    return f"{question}\nBrainstorm all plausible answers and list each potential answer.\n"


def twostep_summary_prompt() -> str:
    """Two-step CoT, step 2 (asked after the step-1 analysis as an assistant turn)."""
    return (
        "Now distill these insights into a summary. Mention all details and possibilities, but make it one coherent answer.\n"
        "Return only that summary. Write it like a response to the question, do not start with 'The analysis includes...' but start it like you would start a response to the question."
    )


def cot_chain_remover(text: str) -> str:
    """Keep only what follows ``Summary:`` (repo ``cot_chain_remover``)."""
    match = _SUMMARY_LINE.search(str(text or ""))
    return match.group(1).strip() if match else str(text or "").strip()


def reasoning_remover(text: str) -> str:
    """Drop a ``…</think>\\n\\n`` prefix (repo ``reasoning_remover``)."""
    parts = str(text or "").split("</think>\n\n")
    if len(parts) == 2:
        return parts[1]
    return str(text or "")


def validate(strategy: str) -> str:
    s = str(strategy or "").strip()
    if s not in STRATEGIES:
        raise ValueError(f"unknown summary_strategy {s!r}; one of {list(STRATEGIES)}")
    return s


def postprocess(strategy: str, text: str, *, thinking_enabled: bool) -> str:
    out = str(text or "")
    if strategy == COT:
        out = cot_chain_remover(out)
    if thinking_enabled and strategy in THINKING_ELIGIBLE:
        out = reasoning_remover(out)
    return out.strip()


def summary_user_prompt(strategy: str, question: str, answers: Optional[List[str]]) -> str:
    if strategy == GREEDY:
        return greedy_prompt(question)
    if strategy == BASIC:
        return basic_prompt(question)
    if strategy == SAMPLE_AND_SUMMARIZE:
        if not answers:
            raise ValueError("sample_and_summarize needs sampled answers")
        return sample_and_summarize_prompt(question, answers)
    if strategy == COT:
        return cot_prompt(question)
    raise ValueError(f"strategy {strategy!r} has no single-prompt form")
