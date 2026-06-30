"""
jd_parser.py — Parse a user-uploaded JD JSON and produce a ScoringConfig
that the scoring engine can use instead of its hardcoded defaults.
"""

from typing import Any, Dict, List, Optional, Set


# ─────────────────────────────────────────────────────────────────────────────
# Default config (current Redrob hackathon JD) — used when no JD is uploaded
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_JD_CONFIG: Dict[str, Any] = {
    "title": "Senior AI Engineer — Founding Team",
    "company_name": "Redrob",
    "required_skills": {
        # Core retrieval/search
        "retrieval", "search system", "ranking system", "recommendation system",
        "recommender system", "relevance", "candidate ranking", "search infrastructure",
        "search relevance", "search quality",
        # Vector search
        "embedding", "vector search", "dense retrieval", "semantic search", "vector index",
        "faiss", "annoy", "hnsw", "vector database", "vector db", "vector store",
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "chroma",
        "elasticsearch", "solr", "typesense",
        # Architectures
        "hybrid search", "hybrid retrieval", "bm25", "sparse retrieval",
        "bi-encoder", "cross-encoder", "dense passage retrieval", "dpr", "colbert",
        "sentence-transformer", "sentence transformer", "all-minilm", "bge", "e5 model",
        # LLM/RAG
        "rag", "retrieval augmented", "llm ranker", "llm reranker", "reranker",
        "cross-encoder reranking", "monobert", "monot5",
        # Evaluation
        "ndcg", "mrr", "mean average precision", "map@", "precision@", "recall@",
        "offline evaluation", "online evaluation", "a/b test", "ab test",
        "eval framework", "evaluation framework", "quality regression",
        # ML tooling
        "python", "pytorch", "tensorflow", "scikit-learn", "keras", "jax",
        "huggingface", "transformers", "langchain", "llamaindex",
    },
    "positive_title_tokens": {
        "ml", "machine learning", "ai engineer", "artificial intelligence",
        "data scientist", "nlp", "natural language", "search engineer",
        "ranking engineer", "applied scientist", "applied ml", "applied ai",
        "research engineer", "recommendation", "retrieval engineer",
    },
    "general_engineer_tokens": {
        "software engineer", "senior engineer", "staff engineer", "principal engineer",
        "backend engineer", "full stack", "platform engineer", "systems engineer",
        "founding engineer", "tech lead",
    },
    "hard_negative_title_tokens": {
        "marketing", "hr ", "human resource", "graphic design",
        "civil engineer", "mechanical engineer", "electrical engineer",
        "structural engineer", "chemical engineer",
        "accountant", "finance manager", "tax",
        "content writer", "copywriter", "seo",
        "customer support", "customer success", "call center",
        "sales executive", "business development",
        "supply chain", "procurement", "logistics",
        "legal", "compliance officer",
        "frontend", "front-end", "front end", "ui/ux", "designer",
        "java developer", "android", "ios", "mobile", "react", "angular",
        "cloud engineer", "devops", "sre", "system admin", "network engineer",
        "qa engineer", "tester", "quality assurance",
    },
    "preferred_locations": {"pune", "noida"},
    "acceptable_locations": {
        "delhi", "new delhi", "ncr", "gurgaon", "gurugram",
        "hyderabad", "bengaluru", "bangalore", "mumbai",
        "navi mumbai", "thane", "delhi ncr",
    },
    "experience_years": {"min": 4, "max": 12, "ideal_min": 5, "ideal_max": 9},
    "salary_range_inr_lpa": {"min": 0, "max": 130},
    "avoid_consulting_only": True,
    "cv_specialty_penalty_keywords": [
        "computer vision", "cv only", "speech recognition", "asr", "tts",
        "robotics", "ros ", "slam", "lidar", "embedded ml",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_jd(jd: Dict[str, Any]) -> List[str]:
    """
    Validate a user-uploaded JD dict.
    Returns a list of error strings. Empty list = valid.
    """
    errors = []

    if not isinstance(jd, dict):
        return ["JD must be a JSON object (dict), not a list or string."]

    # Required fields
    if not jd.get("title") or not isinstance(jd["title"], str):
        errors.append("'title' is required and must be a string (e.g. 'Senior Data Engineer').")

    req_skills = jd.get("required_skills")
    if not req_skills or not isinstance(req_skills, list) or len(req_skills) < 3:
        errors.append("'required_skills' is required — list of at least 3 skill/keyword strings.")

    pref_locs = jd.get("preferred_locations")
    if not pref_locs or not isinstance(pref_locs, list) or len(pref_locs) < 1:
        errors.append("'preferred_locations' is required — list of at least 1 city name.")

    exp = jd.get("experience_years")
    if not exp or not isinstance(exp, dict):
        errors.append("'experience_years' is required — object with 'min' and 'max' fields.")
    else:
        if "min" not in exp or "max" not in exp:
            errors.append("'experience_years' must have both 'min' and 'max' numeric fields.")
        elif exp["max"] < exp["min"]:
            errors.append("'experience_years.max' must be >= 'experience_years.min'.")

    # Optional field type checks
    if "positive_title_tokens" in jd and not isinstance(jd["positive_title_tokens"], list):
        errors.append("'positive_title_tokens' must be a list of strings.")
    if "hard_negative_title_tokens" in jd and not isinstance(jd["hard_negative_title_tokens"], list):
        errors.append("'hard_negative_title_tokens' must be a list of strings.")
    if "acceptable_locations" in jd and not isinstance(jd["acceptable_locations"], list):
        errors.append("'acceptable_locations' must be a list of strings.")
    if "salary_range_inr_lpa" in jd:
        sal = jd["salary_range_inr_lpa"]
        if not isinstance(sal, dict) or "max" not in sal:
            errors.append("'salary_range_inr_lpa' must be an object with at least a 'max' field.")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Parser — converts raw JD dict into a normalized ScoringConfig
# ─────────────────────────────────────────────────────────────────────────────

def parse_jd(jd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a validated JD dict and return a ScoringConfig dict
    ready to be passed into compute_composite_score().

    Falls back to DEFAULT_JD_CONFIG for any missing optional field.
    """
    D = DEFAULT_JD_CONFIG  # shorthand for defaults

    # Helper: list → lowercase set
    def to_set(lst, default: set) -> set:
        if lst and isinstance(lst, list):
            return {str(s).lower().strip() for s in lst if s}
        return default

    # Experience range
    exp = jd.get("experience_years", {})
    exp_min     = float(exp.get("min", D["experience_years"]["min"]))
    exp_max     = float(exp.get("max", D["experience_years"]["max"]))
    exp_ideal_min = float(exp.get("ideal_min", exp_min))
    exp_ideal_max = float(exp.get("ideal_max", exp_max))

    # Salary
    sal = jd.get("salary_range_inr_lpa", D["salary_range_inr_lpa"])
    sal_min = float(sal.get("min", 0))
    sal_max = float(sal.get("max", D["salary_range_inr_lpa"]["max"]))

    config = {
        "title":          str(jd.get("title", D["title"])),
        "company_name":   str(jd.get("company_name", D.get("company_name", ""))),

        # Core domain keywords — searched in career text + skills
        "required_skills": to_set(
            jd.get("required_skills"), D["required_skills"]
        ),

        # Title signals
        "positive_title_tokens": to_set(
            jd.get("positive_title_tokens"), D["positive_title_tokens"]
        ),
        "general_engineer_tokens": D["general_engineer_tokens"],  # always keep generic
        "hard_negative_title_tokens": to_set(
            jd.get("hard_negative_title_tokens"), D["hard_negative_title_tokens"]
        ),

        # Location
        "preferred_locations":  to_set(
            jd.get("preferred_locations"), D["preferred_locations"]
        ),
        "acceptable_locations": to_set(
            jd.get("acceptable_locations"), D["acceptable_locations"]
        ),

        # Experience
        "exp_min": exp_min,
        "exp_max": exp_max,
        "exp_ideal_min": exp_ideal_min,
        "exp_ideal_max": exp_ideal_max,

        # Salary
        "sal_min": sal_min,
        "sal_max": sal_max,

        # Flags
        "avoid_consulting_only": bool(jd.get("avoid_consulting_only",
                                             D["avoid_consulting_only"])),
        "cv_specialty_penalty_keywords": [
            str(k).lower() for k in jd.get(
                "cv_specialty_penalty_keywords",
                D["cv_specialty_penalty_keywords"]
            )
        ],
    }

    return config


def get_default_config() -> Dict[str, Any]:
    """Return the parsed default config (Redrob hackathon JD)."""
    return parse_jd(DEFAULT_JD_CONFIG)
