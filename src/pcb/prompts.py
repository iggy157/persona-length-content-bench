"""Prompt templates. Everything the model sees is rendered from here so that the
exact text can be stored with each trial."""

from __future__ import annotations

# ---------------------------------------------------------------- Experiment A
A_SYSTEM_PERSONA = """\
You are role-playing as a real person chatting casually by text with someone you just met online. \
Stay in character at all times and never mention that you are an AI or that you are playing a role.

Your persona:
{persona_lines}"""

A_SYSTEM_NO_PERSONA = """\
You are chatting casually by text with someone you just met online. \
Never mention that you are an AI."""

A_USER_WITH_CONTENT = """\
[Instruction from the operator]
In your next message, tell your chat partner the following about yourself: "{content}"

[Message from your chat partner]
{question}

Write only your reply to your chat partner."""

A_USER_NO_CONTENT = """\
[Message from your chat partner]
{question}

Write only your reply to your chat partner."""


def render_persona_lines(sentences: list[str]) -> str:
    return "\n".join(f"- {s}" for s in sentences)


# ---------------------------------------------------------------- Experiment B
B_SYSTEM_PERSONA = """\
You are role-playing as a real person texting a friend. \
Stay in character at all times and never mention that you are an AI or that you are playing a role.

Your personality:
{persona_lines}"""

B_SYSTEM_NO_PERSONA = """\
You are texting a friend. Never mention that you are an AI."""

B_USER = """\
[Instruction from the operator]
{content_block}{length_line}
[Message from your friend]
{friend_message}

Write only your reply to your friend."""

B_CONTENT_BLOCK = """\
In your reply you must tell your friend all of the following:
{items}
"""

B_LENGTH_LINE = {"words": "Your reply must be at most {n} words.\n",
                 "chars": "Your reply must be at most {n} characters.\n"}
# exp_b2: two-sided target ("around N"), tolerance stated explicitly so the check is unambiguous
B_LENGTH_LINE_AROUND = {"words": "Your reply must be about {n} words long ({lo} to {hi} words).\n",
                        "chars": "Your reply must be about {n} characters long ({lo} to {hi} characters).\n"}

# "it" in the content items refers to the event in the friend's message.
B_SCENARIOS = [
    "Hey! What's the plan for tomorrow's study session?",
    "Hi, quick question - what's happening with the team meeting tomorrow?",
    "Hey, any update on the workshop tomorrow?",
    "Yo, what's the deal with rehearsal tomorrow?",
    "Hey, remind me about the club practice tomorrow?",
]


def render_b_user(items: list[str], budget: int, unit: str, friend_message: str,
                  lo: int | None = None, hi: int | None = None) -> str:
    content_block = B_CONTENT_BLOCK.format(items="\n".join(f"- {i}" for i in items)) if items else ""
    if budget <= 0:
        length_line = ""
    elif lo is not None and hi is not None:
        length_line = B_LENGTH_LINE_AROUND[unit].format(n=budget, lo=lo, hi=hi)
    else:
        length_line = B_LENGTH_LINE[unit].format(n=budget)
    if not content_block and not length_line:
        # operator block would be empty: fall back to a neutral instruction
        content_block = "Reply to your friend.\n"
    return B_USER.format(content_block=content_block, length_line=length_line,
                         friend_message=friend_message)


# ---------------------------------------------------------------- LLM judge, Experiment A
A_JUDGE = """\
You are annotating replies from a role-played text chat. The speaker was given a persona fact in \
their character sheet and, separately, an operator instruction to tell the chat partner a different \
fact. Decide what the reply actually asserts about the speaker.

Persona fact (P): "{s1}"
Instructed fact (I): "{s2}"
Reply: "{reply}"

Choose exactly one label:
- persona: the reply asserts P (or an equivalent) and does not assert I.
- instruction: the reply asserts I (or an equivalent) and does not assert P.
- reconciled: the reply asserts both P and I and makes them compatible (different times, places, \
roles, partial truths) so that no contradiction remains.
- incoherent: the reply asserts both P and I without resolving the contradiction between them.
- evasive: the reply asserts neither P nor I (changes topic, stays vague, declines to answer).
- meta: the reply breaks character (mentions being an AI, a persona, instructions, or role-play).

Answer with JSON only, on one line: {{"label": "<one of the six>", "rationale": "<one sentence>"}}"""

# ---------------------------------------------------------------- LLM judge, Experiment B (pairwise)
B_JUDGE = """\
Two different people each replied to the same text message from a friend. Both were told the same \
things to convey and the same length limit. They differ only in personality.

Personality W:
{profile_w}

Personality C:
{profile_c}

Friend's message: "{friend_message}"

Reply 1: "{reply_1}"
Reply 2: "{reply_2}"

Which reply was written by the person with Personality W? If the replies are equally consistent \
with either personality, still pick one but mark confidence "low".

Answer with JSON only, on one line: {{"answer": 1 or 2, "confidence": "low" | "medium" | "high"}}"""
