"""Loading and basic graph utilities for the course catalogue."""
import json
from pathlib import Path
from typing import Dict, List


def load_catalogue(path: str) -> Dict[str, dict]:
    """Load catalogue.json and return a dict keyed by course id."""
    data = json.loads(Path(path).read_text())
    return {course["id"]: course for course in data["courses"]}


def load_profiles(path: str) -> List[dict]:
    data = json.loads(Path(path).read_text())
    return data["profiles"]


def course_text(course: dict) -> str:
    """Flatten a course into a single text blob for similarity matching."""
    return " ".join([
        course["title"],
        course["track"],
        course["description"],
        " ".join(course["skills_taught"]),
    ])


def all_prereqs_recursive(catalogue: Dict[str, dict], course_id: str, seen=None) -> List[str]:
    """Return every ancestor prerequisite course id (transitively) for a course."""
    if seen is None:
        seen = set()
    course = catalogue[course_id]
    for prereq_id in course["prerequisites"]:
        if prereq_id not in seen:
            seen.add(prereq_id)
            all_prereqs_recursive(catalogue, prereq_id, seen)
    return list(seen)
