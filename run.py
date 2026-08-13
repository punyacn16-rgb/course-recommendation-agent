#!/usr/bin/env python3
"""
Course Recommendation Agent - CLI entry point.

Usage:
    python run.py --sample                 Run all sample profiles from data/profiles.json
    python run.py --sample priya           Run a single named sample profile
    python run.py --interactive            Answer a few prompts to build a custom profile

Output is printed to the console and saved as Markdown files under outputs/.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.catalogue import load_profiles
from src.agent import recommend_for_profile

CATALOGUE_PATH = "data/catalogue.json"
PROFILES_PATH = "data/profiles.json"
OUTPUT_DIR = Path("outputs")


def format_result_markdown(result: dict) -> str:
    profile = result["profile"]
    lines = [
        f"# Learning Path for {profile['name']}",
        "",
        f"**Background:** {profile['background']}",
        f"**Goal:** {profile['goal']}",
        f"**Known skills:** {', '.join(profile.get('known_skills', [])) or 'None'}",
        "",
        "## Recommended Path",
        "",
    ]
    for i, course in enumerate(result["path"], start=1):
        tag = "🎯 goal course" if course["is_goal_course"] else "prerequisite"
        lines.append(f"### {i}. {course['title']}  _( {tag}, ~{course['estimated_hours']}h )_")
        lines.append(f"- **Skills taught:** {', '.join(course['skills_taught'])}")
        lines.append(f"- **Why:** {course['reason']}")
        lines.append("")
    if not result["path"]:
        lines.append("_No courses recommended — the student's known skills already cover "
                      "everything relevant to this goal in the catalogue._")
    return "\n".join(lines)


def print_result(result: dict) -> None:
    print(format_result_markdown(result))
    print("\n" + "=" * 70 + "\n")


def run_for_profile(profile: dict) -> dict:
    result = recommend_for_profile(profile, CATALOGUE_PATH)
    print_result(result)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{profile['id']}.md"
    out_path.write_text(format_result_markdown(result))
    print(f"[saved] {out_path}")
    return result


def interactive_profile() -> dict:
    print("Let's build your profile.\n")
    name = input("Your name: ").strip() or "Student"
    background = input("Your background (1-2 sentences): ").strip()
    goal = input("Your learning goal: ").strip()
    known = input("Skills you already know (comma-separated, or leave blank): ").strip()
    known_skills = [s.strip() for s in known.split(",") if s.strip()]
    return {
        "id": name.lower().replace(" ", "-"),
        "name": name,
        "background": background,
        "goal": goal,
        "known_skills": known_skills,
    }


def main():
    load_dotenv()  # pulls ANTHROPIC_API_KEY from .env if present

    parser = argparse.ArgumentParser(description="Course Recommendation Agent")
    parser.add_argument("--sample", nargs="?", const="ALL",
                         help="Run sample profile(s). Optionally give a profile id (e.g. priya).")
    parser.add_argument("--interactive", action="store_true",
                         help="Build a custom profile via prompts.")
    args = parser.parse_args()

    if args.interactive:
        profile = interactive_profile()
        run_for_profile(profile)
        return

    if args.sample:
        profiles = load_profiles(PROFILES_PATH)
        if args.sample == "ALL":
            for profile in profiles:
                run_for_profile(profile)
        else:
            match = next((p for p in profiles if p["id"] == args.sample), None)
            if not match:
                available = ", ".join(p["id"] for p in profiles)
                print(f"No sample profile named '{args.sample}'. Available: {available}")
                return
            run_for_profile(match)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
