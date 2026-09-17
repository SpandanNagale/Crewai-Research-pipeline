"""
Empirically calibrate settings.KNOWLEDGE_REUSE_THRESHOLD against the real
embedding model in use (settings.EMBEDDING_MODEL), rather than carrying a
value tuned for a different embedding function across a storage migration.

The 0.87 threshold that shipped in earlier phases was set against Chroma's
DefaultEmbeddingFunction (all-MiniLM-L6-v2). Gemini embeddings have a
different cosine similarity distribution -- scores between unrelated texts
sit much higher -- so carrying 0.87 forward unchanged would cause silent
false reuse: the pipeline returns a cached report for a topic that merely
looks adjacent, and it does so silently, because a wrong cache hit still
produces a plausible-looking report.

Run: python -m eval.calibrate_reuse
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config.settings import settings
from memory.embedder import embed

GOLDEN_SET_PATH = Path(__file__).parent / "golden" / "golden_set.json"

# Paraphrases of the same underlying question, drawn from / adjacent to
# eval/golden/golden_set.json's topics -- a real cache hit should always
# clear the chosen threshold on these.
TRUE_MATCH_PAIRS: List[Tuple[str, str]] = [
    ("The impact of remote work on urban commercial real estate since 2020",
     "How has remote work affected commercial real estate in cities since 2020?"),
    ("How electric vehicle adoption is changing the global automotive supply chain",
     "The way growing EV adoption is reshaping the global car parts supply chain"),
    ("The role of mRNA vaccine technology beyond COVID-19",
     "How mRNA vaccine technology is being applied to diseases other than COVID-19"),
    ("The current state of renewable energy storage technology",
     "Where renewable energy storage technology stands today"),
    ("How generative AI is being adopted in enterprise software development",
     "The adoption of generative AI tools within enterprise software engineering teams"),
    ("Whether social media use causes or merely correlates with teenage mental health decline",
     "Does social media usage cause teen mental health problems, or is it merely correlated with it?"),
    ("The economic impact of the gig economy on traditional employment",
     "How the gig economy is economically affecting traditional jobs"),
    ("Recent advances in reusable rocket technology and their effect on launch costs",
     "How recent progress in reusable rockets has changed the cost of launches"),
]

# Same domain, genuinely different question -- a reuse hit here would be
# wrong: these must NOT be treated as the same cached research question.
NEAR_MISS_PAIRS: List[Tuple[str, str]] = [
    ("The impact of remote work on urban commercial real estate since 2020",
     "Whether remote work increases or decreases long-term employee productivity"),
    ("How electric vehicle adoption is changing the global automotive supply chain",
     "What is the global market share of electric vehicles in new car sales as of 2024?"),
    ("How generative AI is being adopted in enterprise software development",
     "Whether AI coding assistants improve or degrade long-term code quality"),
    ("The current state of renewable energy storage technology",
     "Whether nuclear power is a net positive or negative for meeting climate goals"),
    ("The economic impact of the gig economy on traditional employment",
     "Whether a four-day work week improves or harms overall economic output"),
    ("The role of mRNA vaccine technology beyond COVID-19",
     "Does intermittent fasting provide a metabolic benefit beyond simple calorie restriction?"),
    ("Whether current AI safety regulation proposals help or hinder innovation",
     "How streaming services have changed the film and television industry business model"),
]


def _cosine(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def _pairwise_similarities(pairs: List[Tuple[str, str]]) -> List[float]:
    flat = [text for pair in pairs for text in pair]
    vectors = embed(flat)
    if vectors is None:
        raise RuntimeError("Embedding request failed -- check GEMINI_API_KEY and connectivity")
    return [_cosine(vectors[i * 2], vectors[i * 2 + 1]) for i in range(len(pairs))]


def calibrate() -> dict:
    true_match_scores = _pairwise_similarities(TRUE_MATCH_PAIRS)
    near_miss_scores = _pairwise_similarities(NEAR_MISS_PAIRS)

    min_true_match = min(true_match_scores)
    max_near_miss = max(near_miss_scores)

    # Zero false-reuse on the near-miss set is non-negotiable: a silent
    # wrong cache hit is worse than an occasional unnecessary re-run.
    # Recommend the midpoint of the gap rather than hugging either edge --
    # it gives equal margin against a near-miss that scores a bit higher
    # next time and a genuine paraphrase that scores a bit lower. Report
    # failure if the populations overlap at all (no gap to sit in).
    separated = max_near_miss < min_true_match
    recommended = round((max_near_miss + min_true_match) / 2, 4) if separated else None

    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
        "true_match_scores": [round(s, 4) for s in true_match_scores],
        "near_miss_scores": [round(s, 4) for s in near_miss_scores],
        "min_true_match": round(min_true_match, 4),
        "max_near_miss": round(max_near_miss, 4),
        "recommended_threshold": recommended,
        "separated": separated,
    }


def main() -> int:
    result = calibrate()
    print(json.dumps(result, indent=2))
    if not result["separated"]:
        print(
            "\nWARNING: true-match and near-miss populations overlap -- no single "
            "threshold gives zero false-reuse on this set. Add more pairs or "
            "reconsider the embedding model before trusting KNOWLEDGE_REUSE_THRESHOLD.",
            file=sys.stderr,
        )
        return 1
    print(f"\nRecommended KNOWLEDGE_REUSE_THRESHOLD: {result['recommended_threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
