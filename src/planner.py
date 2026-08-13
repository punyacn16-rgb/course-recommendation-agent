"""
Builds an ordered learning path.

Steps:
1. Use similarity scores to pick the top-N most goal-relevant courses ("target courses").
2. Walk backwards through prerequisites to find every course needed to reach those targets.
3. Drop any course whose skills are already fully covered by the student's known_skills.
4. Topologically sort the remaining courses so prerequisites always come before
   the courses that depend on them.
"""
from typing import Dict, List, Set, Tuple

from .catalogue import all_prereqs_recursive


def _is_already_known(course: dict, known_skills: Set[str]) -> bool:
    return set(course["skills_taught"]).issubset(known_skills)


def select_target_courses(
    ranked: List[Tuple[str, float]],
    catalogue: Dict[str, dict],
    known_skills: Set[str],
    top_n: int = 4,
    min_score: float = 0.05,
) -> List[str]:
    """Pick the highest-relevance courses the student doesn't already know."""
    targets = []
    for course_id, score in ranked:
        if score < min_score:
            continue
        if _is_already_known(catalogue[course_id], known_skills):
            continue
        targets.append(course_id)
        if len(targets) >= top_n:
            break
    return targets


def build_required_set(
    catalogue: Dict[str, dict], target_ids: List[str], known_skills: Set[str]
) -> Set[str]:
    """Collect target courses plus every prerequisite they transitively need,
    excluding anything the student already knows."""
    required = set()
    for target_id in target_ids:
        required.add(target_id)
        required.update(all_prereqs_recursive(catalogue, target_id))

    required = {
        cid for cid in required
        if not _is_already_known(catalogue[cid], known_skills)
    }
    return required


def topological_order(catalogue: Dict[str, dict], course_ids: Set[str]) -> List[str]:
    """Kahn's algorithm restricted to the given subset of courses."""
    course_ids = set(course_ids)
    in_degree = {cid: 0 for cid in course_ids}
    edges = {cid: [] for cid in course_ids}

    for cid in course_ids:
        for prereq_id in catalogue[cid]["prerequisites"]:
            if prereq_id in course_ids:
                edges[prereq_id].append(cid)
                in_degree[cid] += 1

    queue = sorted([cid for cid in course_ids if in_degree[cid] == 0])
    ordered = []

    while queue:
        # stable, deterministic ordering: prefer lower estimated_hours, then id
        queue.sort(key=lambda cid: (catalogue[cid]["estimated_hours"], cid))
        current = queue.pop(0)
        ordered.append(current)
        for neighbor in edges[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(course_ids):
        # Cycle guard (shouldn't happen with this catalogue, but fail safe).
        remaining = course_ids - set(ordered)
        ordered.extend(sorted(remaining))

    return ordered


def build_learning_path(
    catalogue: Dict[str, dict],
    ranked: List[Tuple[str, float]],
    known_skills: List[str],
    top_n: int = 4,
) -> Tuple[List[str], List[str]]:
    """
    Returns (ordered_course_ids, target_course_ids).
    target_course_ids are the "goal" courses; the rest of ordered_course_ids
    are prerequisites needed to reach them.
    """
    known = set(known_skills)
    targets = select_target_courses(ranked, catalogue, known, top_n=top_n)
    required = build_required_set(catalogue, targets, known)
    ordered = topological_order(catalogue, required)
    return ordered, targets
