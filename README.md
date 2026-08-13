# Course Recommendation Agent

An AI agent that takes a student's background, goal, and known skills, and
produces a **personalised, prerequisite-ordered learning path** with a
plain-English reason for every course.

Built for the Rooman 24-Hour AI Agent Challenge.

---

## What it does

**Input:** a student profile — background, learning goal, known skills.
**Output:** an ordered list of courses to take, each tagged as either a
*prerequisite* or a *goal course*, with a one-line explanation of why it's
in the path and what it builds on.

```
"My agent takes a student profile (background, goals, known skills)
 and produces an ordered learning path with reasons for each step."
```

---

## How it works (architecture)

The pipeline has four deterministic stages plus one LLM stage:

```
Student profile
      │
      ▼
1. similarity.py   →  TF-IDF + cosine similarity ranks every course in the
                       catalogue against the student's goal + background text
      │
      ▼
2. planner.py       → picks the top-N most relevant courses the student
                       doesn't already know ("goal courses"), then walks
                       backwards through the prerequisite graph to pull in
                       every course needed to reach them
      │
      ▼
3. planner.py        → topologically sorts the required courses (Kahn's
   (topo sort)          algorithm) so prerequisites always appear before the
                         courses that depend on them
      │
      ▼
4. llm.py            → given the already-ordered path, an LLM (Claude)
                         writes a short, personalized rationale for each
                         step. If no API key is set, a deterministic
                         template fills in instead.
      │
      ▼
Ordered path + reasons (printed + saved to outputs/*.md)
```

**Why split it this way?** Course *selection and ordering* is a graph/ranking
problem — deterministic code does it faster, cheaper, and more
reproducibly than an LLM, and prerequisite correctness is easy to verify
with a topological sort. The LLM is used only where it's actually the right
tool: turning structured data into a clear, personalized sentence. This also
means the agent **still works with zero API calls** — useful for reviewers
without a key, and for keeping the "core" logic testable and deterministic.

### NLP / similarity method

Goal-to-course matching uses **TF-IDF vectorization + cosine similarity**
(scikit-learn) over each course's title, track, description, and taught
skills, compared against the student's `goal + background` text. This is
lightweight, needs no API call or embeddings model, and is fully
explainable — you can trace a match back to the overlapping terms. If
scikit-learn isn't installed, `similarity.py` falls back to a simple
keyword-overlap (Jaccard) score so the agent still runs.

---

## Project structure

```
course-recommendation-agent/
├── data/
│   ├── catalogue.json     # 18 courses across 4 tracks, with prerequisites
│   └── profiles.json      # 4 sample student profiles
├── src/
│   ├── catalogue.py        # load data, graph helpers
│   ├── similarity.py       # TF-IDF relevance ranking
│   ├── planner.py          # target selection + prerequisite resolution + topo sort
│   ├── llm.py               # Claude call + template fallback for rationale
│   └── agent.py             # orchestrates the full pipeline
├── tests/
│   └── test_agent.py        # sanity tests (no pytest needed)
├── outputs/                 # sample runs, saved as Markdown
├── run.py                   # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
git clone <your-repo-url>
cd course-recommendation-agent
pip install -r requirements.txt
```

**(Optional) enable LLM-generated rationale:**

```bash
cp .env.example .env
# then edit .env and paste your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, the agent still runs end-to-end — it just uses a
deterministic template instead of an LLM-written explanation. See
[Tradeoffs](#tradeoffs--design-notes) below for why this fallback exists.

---

## Running it

**Run all 4 sample profiles:**
```bash
python run.py --sample
```

**Run a single sample profile:**
```bash
python run.py --sample priya
# other ids: arjun, meera, karan
```

**Build your own profile interactively:**
```bash
python run.py --interactive
```

Every run prints the path to the console **and** saves it as
`outputs/<profile-id>.md`.

**Run the test suite:**
```bash
python tests/test_agent.py
```

---

## Sample input/output

**Input** (`data/profiles.json`, Arjun):
```json
{
  "id": "arjun",
  "name": "Arjun",
  "background": "Final-year CS student who has completed an intro Python course and knows basic SQL.",
  "goal": "Break into machine learning and eventually specialize in natural language processing.",
  "known_skills": ["python", "programming-basics", "sql"]
}
```

**Output** (`outputs/arjun.md`, abridged):
```
1. Data Analysis with Pandas       (prerequisite, ~15h)
2. Statistics & Probability        (prerequisite, ~16h)
3. Intermediate Python (OOP)       (goal course, ~18h)
4. Machine Learning Foundations    (goal course, ~30h)
5. Deep Learning with PyTorch      (goal course, ~35h)
```

Note Arjun's already-known `python`/`sql` skills correctly **excluded**
`python-basics` and `sql-basics` from the path — the planner only pulls in
what's actually missing. Full outputs for all 4 sample profiles (Priya,
Arjun, Meera, Karan) are committed in `outputs/`.

---

## Tradeoffs & design notes

- **TF-IDF over embeddings:** chosen for zero extra dependencies/API cost and
  full explainability. The tradeoff is real: TF-IDF is bag-of-words, so it
  can occasionally rank a loosely-related course above a more relevant one
  when the goal text is short (e.g. in testing, Karan's "deploy ML models on
  the cloud" goal pulled in NLP/deep-learning courses alongside the correct
  MLOps/cloud/Docker ones, because "ML" overlaps lexically). With more time
  I'd swap in sentence embeddings (e.g. `sentence-transformers`) for
  semantic rather than lexical matching, or add a small curated
  goal → skill-tag mapping as a rule-based layer.
- **Deterministic planner + LLM-for-explanation split:** keeps the "what
  order are courses in" question testable and reproducible (see
  `tests/test_agent.py`), while still using an LLM where it adds real value.
  A pure "ask the LLM for the whole path" approach would be less reliable at
  respecting prerequisites and harder to unit test.
- **Fallback rationale without an API key:** ensures reviewers can run and
  fully evaluate the agent with zero setup friction, per the challenge's
  "make setup foolproof" guidance. The fallback is template-based (not an
  LLM), so its writing quality is intentionally more mechanical — I've left
  the actual template in `src/llm.py::_fallback_rationale`, and the
  `outputs/*.md` files in this repo were generated with the fallback since
  no key was available in the build environment.
- **Small hand-built catalogue (18 courses):** enough to demonstrate
  multi-step prerequisite chains across 4 tracks (data, web, cloud/DevOps,
  programming foundations) without spending the day on data collection. A
  real product would pull this from a live course database.
- **`top_n=4` goal courses per path:** kept paths readable for a demo.
  Configurable via `recommend_for_profile(..., top_n=...)`.
- **What I'd add with more time:** semantic (embedding-based) matching,
  a larger/real catalogue, a difficulty/time budget the student can specify
  ("I only have 5 hours/week"), and a simple web UI instead of CLI-only.

---

## What's honestly not done

- No persistence/database — profiles are static JSON files, not a saved
  user history across runs.
- No UI beyond the CLI.
- Rationale quality depends on whether an LLM key is configured; the
  committed sample outputs use the template fallback (see above).
