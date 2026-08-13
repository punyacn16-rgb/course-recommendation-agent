"""Core orchestration: profile -> ranked courses -> ordered path -> rationale."""
from typing import Dict

from .catalogue import load_catalogue
from .similarity import rank_courses_by_relevance
from .planner import build_learning_path
from .llm import generate_rationale


def recommend_for_profile(profile: dict, catalogue_path: str, top_n: int = 4) -> Dict:
    """
    Run the full pipeline for one student profile and return a structured result:
    {
      "profile": {...},
      "path": [ {course fields..., "reason": "...", "is_goal_course": bool}, ... ]
    }
    """
    catalogue = load_catalogue(catalogue_path)

    query_text = f"{profile['goal']} {profile['background']}"
    ranked = rank_courses_by_relevance(catalogue, query_text)

    ordered_ids, target_ids = build_learning_path(
        catalogue, ranked, profile.get("known_skills", []), top_n=top_n
    )
    ordered_courses = [catalogue[cid] for cid in ordered_ids]

    rationale = generate_rationale(profile, ordered_courses, target_ids)

    path = []
    for course in ordered_courses:
        path.append({
            **course,
            "reason": rationale.get(course["id"], ""),
            "is_goal_course": course["id"] in target_ids,
        })

    return {"profile": profile, "path": path}
