"""
LLM integration for turning a structured learning path into a natural-language
explanation for each step.

Design: the *ordering and course selection* is done by deterministic code
(similarity ranking + prerequisite graph, see similarity.py / planner.py) so it
is reproducible and explainable without an API call. The LLM's job is narrower
and more suited to a language model: writing a clear, personalized rationale
for each step given the student's profile and the already-computed path.

If no API key is configured, we fall back to a template-based rationale so the
agent still produces useful output end-to-end without any external dependency.
"""
import os
from typing import Dict, List, Optional

SYSTEM_PROMPT = """You are a course advisor for an ed-tech platform. You will be given:
- A student profile (background, goal, known skills)
- An ordered list of courses already selected for them by a prerequisite-aware planner

Your job is ONLY to explain, in 1-2 concise sentences per course, WHY each course
is the right next step for this specific student -- referencing their goal and
what skill it builds on or unlocks next. Do not reorder or add/remove courses.
Do not invent courses that are not in the list you were given.

Respond ONLY as a JSON object of the form:
{"rationale": {"<course_id>": "<1-2 sentence explanation>", ...}}
No other text, no markdown fences.
"""


def _fallback_rationale(
    profile: dict, ordered_courses: List[dict], target_ids: List[str]
) -> Dict[str, str]:
    """Deterministic, template-based rationale used when no LLM API key is set."""
    rationale = {}
    for i, course in enumerate(ordered_courses):
        cid = course["id"]
        skills = ", ".join(course["skills_taught"])
        if course["prerequisites"]:
            prereq_titles = [c["title"] for c in ordered_courses if c["id"] in course["prerequisites"]]
            basis = f" building on {', '.join(prereq_titles)}" if prereq_titles else ""
        else:
            basis = ""
        if cid in target_ids:
            purpose = f"directly supports the goal: \"{profile['goal']}\""
        else:
            purpose = "is a prerequisite needed before later courses in this path"
        rationale[cid] = (
            f"Teaches {skills}{basis}; this course {purpose}."
        )
    return rationale


def generate_rationale(
    profile: dict, ordered_courses: List[dict], target_ids: List[str]
) -> Dict[str, str]:
    """
    Returns a dict mapping course_id -> rationale string.
    Uses Anthropic's Claude API if ANTHROPIC_API_KEY is set, otherwise falls
    back to a deterministic template so the agent always produces output.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_rationale(profile, ordered_courses, target_ids)

    try:
        import anthropic
        import json

        client = anthropic.Anthropic(api_key=api_key)

        course_summary = [
            {
                "id": c["id"],
                "title": c["title"],
                "skills_taught": c["skills_taught"],
                "prerequisites": c["prerequisites"],
                "is_goal_course": c["id"] in target_ids,
            }
            for c in ordered_courses
        ]

        user_message = (
            f"Student profile:\n"
            f"- Background: {profile['background']}\n"
            f"- Goal: {profile['goal']}\n"
            f"- Known skills: {profile.get('known_skills', [])}\n\n"
            f"Ordered course path (already sequenced, do not reorder):\n"
            f"{json.dumps(course_summary, indent=2)}"
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        rationale = parsed.get("rationale", {})

        # Safety net: if the model skipped a course, fill it in with the template.
        fallback = _fallback_rationale(profile, ordered_courses, target_ids)
        for course in ordered_courses:
            rationale.setdefault(course["id"], fallback[course["id"]])
        return rationale

    except Exception as exc:  # noqa: BLE001 - we want the agent to degrade gracefully
        print(f"[warn] LLM call failed ({exc}); using template-based rationale instead.")
        return _fallback_rationale(profile, ordered_courses, target_ids)
