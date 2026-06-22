"""
Procedural Memory — dynamic skill loading.

A skill registry holds multiple skills. At startup the agent only sees
names + one-line descriptions (Discovery). When a task arrives, the agent
picks the relevant skill and loads its full instructions (Activation).
Only the selected skill's instructions enter the context window.

Compare:
  naive approach:  all 3 skill instruction sets in system prompt = ~500 tokens always
  skill registry:  20 tokens at discovery, ~150 tokens after activation
"""

from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b")

# ── skill registry ─────────────────────────────────────────────────────
# Each skill: short description (for discovery) + full instructions (for execution).
# Full instructions only enter the context when the skill is activated.

SKILLS = {
    "code_review": {
        "description": "Review code for bugs, security issues, and best practices.",
        "instructions": """
You are doing a code review. Follow these steps in order:

1. SECURITY: Check for injection vulnerabilities, hardcoded secrets, unsafe deserialization.
2. ERROR HANDLING: Identify unhandled exceptions, missing input validation, silent failures.
3. READABILITY: Flag unclear variable names, missing docstrings, overly complex logic.
4. PERFORMANCE: Spot N+1 queries, unnecessary loops, missing indexes.
5. SUMMARY: List findings as CRITICAL / WARNING / SUGGESTION with one line each.

Be concise. No praise. Only actionable findings.
""",
    },

    "data_report": {
        "description": "Analyse data and produce a structured summary report.",
        "instructions": """
You are writing a data analysis report. Structure it as follows:

1. HEADLINE METRIC: The single most important number in one sentence.
2. TREND: Is the metric going up, down, or flat? Over what period?
3. BREAKDOWN: Top 3 contributing factors or segments.
4. ANOMALY: Any outlier or unexpected data point worth flagging.
5. RECOMMENDATION: One concrete action based on the data.

Use plain language. Avoid jargon. Keep each section to 1-2 sentences.
""",
    },

    "email_draft": {
        "description": "Draft a professional email based on a brief.",
        "instructions": """
You are drafting a professional email. Follow this structure:

1. SUBJECT LINE: Clear, specific, under 60 characters.
2. OPENING: One sentence establishing context (no "I hope this email finds you well").
3. BODY: The key message in 2-3 sentences. State what you need or are sharing.
4. CALL TO ACTION: One specific ask with a deadline if relevant.
5. CLOSING: Professional sign-off. No fluff.

Tone: direct and respectful. Avoid passive voice.
""",
    },
}

# ── discovery context (tiny — just names + descriptions) ───────────────

DISCOVERY_PROMPT = "Available skills:\n" + "\n".join(
    f"- {name}: {skill['description']}"
    for name, skill in SKILLS.items()
)

# ── skill selection ────────────────────────────────────────────────────

class SkillSelection(BaseModel):
    skill_name: Literal["code_review", "data_report", "email_draft"]
    reason: str

selector = model.with_structured_output(SkillSelection)

def select_skill(task: str) -> str:
    result = selector.invoke([
        SystemMessage(f"Given a user task, select the most relevant skill.\n\n{DISCOVERY_PROMPT}"),
        HumanMessage(task),
    ])
    print(f"  [discovery] selected '{result.skill_name}' — {result.reason}")
    return result.skill_name

# ── activation + execution ─────────────────────────────────────────────

def run(task: str):
    print(f"\ntask: {task}")

    # Step 1: Discovery — agent sees only names + descriptions
    skill_name = select_skill(task)

    # Step 2: Activation — load full instructions for selected skill only
    instructions = SKILLS[skill_name]["instructions"]
    print(f"  [activation] loaded '{skill_name}' instructions ({len(instructions.split())} words)")

    # Step 3: Execution — agent follows the skill's steps
    response = model.invoke([
        SystemMessage(instructions.strip()),
        HumanMessage(task),
    ])

    print(f"\n--- output ---")
    print(response.content[:400])
    print()

# ── demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    context_naive = sum(len(s["instructions"].split()) for s in SKILLS.values())
    context_skill = max(len(s["instructions"].split()) for s in SKILLS.values())

    print("=== context window comparison ===")
    print(f"  naive (all skills in prompt): ~{context_naive} instruction words always loaded")
    print(f"  skill registry:               ~{context_skill} instruction words after activation")
    print(f"  discovery overhead:           {len(DISCOVERY_PROMPT.split())} words\n")

    run("Please review this Python function for issues:\n\ndef get_user(id):\n    return db.execute(f'SELECT * FROM users WHERE id={id}')")

    run("Our Q3 revenue was $2.1M, up from $1.8M in Q2 and $1.6M in Q1. Write a short report.")

    run("Write an email to the engineering team announcing that deployments are frozen this Friday.")
