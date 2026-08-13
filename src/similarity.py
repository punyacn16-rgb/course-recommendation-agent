"""
Similarity scoring between a student's goal/background text and course descriptions.

Approach: TF-IDF vectorization + cosine similarity (scikit-learn). This is a
lightweight, dependency-light NLP method that works well for short catalogue-style
text without needing embeddings or an API call, and it's fully explainable
(we can point to the exact overlapping terms if needed).

Falls back to a simple keyword-overlap score if scikit-learn is unavailable,
so the agent still runs in a minimal environment.
"""
from typing import Dict, List, Tuple

from .catalogue import course_text

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _keyword_overlap_score(query: str, doc: str) -> float:
    q = set(query.lower().split())
    d = set(doc.lower().split())
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def rank_courses_by_relevance(
    catalogue: Dict[str, dict], query_text: str
) -> List[Tuple[str, float]]:
    """
    Rank every course in the catalogue by relevance to query_text (typically the
    student's goal + background). Returns a list of (course_id, score) sorted
    descending by score.
    """
    course_ids = list(catalogue.keys())
    docs = [course_text(catalogue[cid]) for cid in course_ids]

    if _HAS_SKLEARN:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(docs + [query_text])
        query_vec = matrix[-1]
        course_vecs = matrix[:-1]
        scores = cosine_similarity(query_vec, course_vecs)[0]
    else:
        scores = [_keyword_overlap_score(query_text, doc) for doc in docs]

    ranked = sorted(zip(course_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked
