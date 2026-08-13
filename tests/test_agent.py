"""
Lightweight sanity tests - no pytest required, just run:
    python tests/test_agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalogue import load_catalogue, load_profiles, all_prereqs_recursive
from src.similarity import rank_courses_by_relevance
from src.planner import build_learning_path

CATALOGUE_PATH = "data/catalogue.json"
PROFILES_PATH = "data/profiles.json"


def test_catalogue_loads():
    catalogue = load_catalogue(CATALOGUE_PATH)
    assert len(catalogue) >= 15, "expected at least 15 courses in catalogue"
    for cid, course in catalogue.items():
        assert course["id"] == cid
        for prereq in course["prerequisites"]:
            assert prereq in catalogue, f"unknown prerequisite '{prereq}' on {cid}"
    print("test_catalogue_loads: PASS")


def test_prereq_chain_resolves():
    catalogue = load_catalogue(CATALOGUE_PATH)
    ancestors = all_prereqs_recursive(catalogue, "deep-learning-intro")
    assert "ml-foundations" in ancestors
    assert "python-basics" in ancestors  # transitively required
    print("test_prereq_chain_resolves: PASS")


def test_path_respects_prerequisite_order():
    catalogue = load_catalogue(CATALOGUE_PATH)
    profiles = {p["id"]: p for p in load_profiles(PROFILES_PATH)}
    arjun = profiles["arjun"]

    query = f"{arjun['goal']} {arjun['background']}"
    ranked = rank_courses_by_relevance(catalogue, query)
    ordered_ids, target_ids = build_learning_path(catalogue, ranked, arjun["known_skills"])

    position = {cid: i for i, cid in enumerate(ordered_ids)}
    for cid in ordered_ids:
        for prereq in catalogue[cid]["prerequisites"]:
            if prereq in position:
                assert position[prereq] < position[cid], (
                    f"{prereq} should come before {cid} but doesn't"
                )
    assert len(target_ids) > 0, "expected at least one goal course for Arjun"
    print("test_path_respects_prerequisite_order: PASS")


def test_known_skills_are_excluded():
    catalogue = load_catalogue(CATALOGUE_PATH)
    profiles = {p["id"]: p for p in load_profiles(PROFILES_PATH)}
    karan = profiles["karan"]  # already knows ml-foundations-level skills

    query = f"{karan['goal']} {karan['background']}"
    ranked = rank_courses_by_relevance(catalogue, query)
    ordered_ids, _ = build_learning_path(catalogue, ranked, karan["known_skills"])

    assert "ml-foundations" not in ordered_ids, "should skip courses Karan already knows"
    print("test_known_skills_are_excluded: PASS")


if __name__ == "__main__":
    test_catalogue_loads()
    test_prereq_chain_resolves()
    test_path_respects_prerequisite_order()
    test_known_skills_are_excluded()
    print("\nAll tests passed.")
