"""
scoring.py — Composite scoring formula for Redrob candidate ranker.

Design philosophy:
- A Marketing Manager with 'Pinecone' in skills should score ~0.05
- A Recommendation Systems Engineer with actual vector search work should score ~0.75+
- Behavioral signals modulate availability, not core fit
- Honeypots get zeroed before the formula runs

All computation is pure Python stdlib — no dependencies, no network.
"""

import math
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

# Import JD parser for dynamic config support
try:
    from rank.jd_parser import get_default_config as _get_default_jd
except ImportError:
    try:
        from jd_parser import get_default_config as _get_default_jd
    except ImportError:
        _get_default_jd = None

_DEFAULT_JD_CONFIG: Optional[Dict[str, Any]] = None

def _default_jd() -> Dict[str, Any]:
    """Lazy-load the default JD config."""
    global _DEFAULT_JD_CONFIG
    if _DEFAULT_JD_CONFIG is None and _get_default_jd is not None:
        _DEFAULT_JD_CONFIG = _get_default_jd()
    return _DEFAULT_JD_CONFIG or {}

# ─────────────────────────────────────────────────────────────────────────────
# JD REQUIREMENTS — what the role MEANS, not just what it SAYS
# ─────────────────────────────────────────────────────────────────────────────

# Career-history keywords that indicate PRODUCTION retrieval/ranking/ML work.
# These are looked for in job descriptions, not in skills lists.
# The distinction: skills list is self-reported; career description is harder to fake.
PRODUCTION_RETRIEVAL_SIGNALS = {
    # Core system types
    "retrieval", "search system", "ranking system", "recommendation system",
    "recommender system", "relevance", "candidate ranking", "search infrastructure",
    "search relevance", "search quality",
    # Vector / dense retrieval
    "embedding", "vector search", "dense retrieval", "semantic search", "vector index",
    "faiss", "annoy", "hnsw", "vector database", "vector db", "vector store",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "chroma",
    "elasticsearch", "solr", "typesense",
    # Retrieval architectures
    "hybrid search", "hybrid retrieval", "bm25", "sparse retrieval",
    "bi-encoder", "cross-encoder", "dense passage retrieval", "dpr", "colbert",
    "sentence-transformer", "sentence transformer", "all-minilm", "bge", "e5 model",
    # LLMs applied to IR
    "rag", "retrieval augmented", "llm ranker", "llm reranker", "reranker",
    "cross-encoder reranking", "monobert", "monot5",
    # Evaluation (the JD explicitly asks for this)
    "ndcg", "mrr", "mean average precision", "map@", "precision@", "recall@",
    "offline evaluation", "online evaluation", "a/b test", "ab test",
    "eval framework", "evaluation framework", "quality regression",
    "embedding drift", "index refresh", "retrieval quality",
    # Core ML tooling (improves Python evidence detection)
    "python", "pytorch", "tensorflow", "scikit-learn", "keras", "jax",
    "huggingface", "transformers", "langchain", "llamaindex",
}

# Title keywords that indicate the person IS an AI/ML/Search engineer
# Weight: career history title, not just current title
POSITIVE_TITLE_TOKENS = {
    "ml", "machine learning", "ai engineer", "artificial intelligence",
    "data scientist", "nlp", "natural language", "search engineer",
    "ranking engineer", "applied scientist", "applied ml", "applied ai",
    "research engineer",  # different from "researcher" — engineers ship code
    "recommendation", "retrieval engineer",
}

# Title tokens that DESCRIBE the role WITHOUT requiring ML keyword
GENERAL_ENGINEER_TOKENS = {
    "software engineer", "senior engineer", "staff engineer", "principal engineer",
    "backend engineer", "full stack", "platform engineer", "systems engineer",
    "founding engineer", "tech lead",  # tech lead who writes code
}

# Hard negative title tokens — these candidates are NOT the right role
# Note: we check CAREER HISTORY titles, not just current title
HARD_NEGATIVE_TITLE_TOKENS = {
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
}

# Full-career consulting penalty (if ALL companies are these → consulting-only flag)
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl technologies", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "persistent systems", "mindtree",
    "kpit", "mastech", "igate", "niit technologies",
}

# Target locations (Pune/Noida preferred + Tier-1 cities)
PREFERRED_LOCATIONS = {"pune", "noida"}
ACCEPTABLE_LOCATIONS = {
    "delhi", "new delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "bengaluru", "bangalore", "mumbai",
    "navi mumbai", "thane", "delhi ncr",
}

# Industries that signal product company / applied ML context
PRODUCT_INDUSTRIES = {
    "software", "saas", "technology", "tech", "fintech", "edtech",
    "hrtech", "hr tech", "adtech", "healthtech", "ecommerce",
    "internet", "mobile", "gaming", "media tech",
    "ai", "machine learning", "data", "analytics",
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    """Days elapsed since a YYYY-MM-DD date string. Returns 9999 on error."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 9999


def _normalize(value: float, lo: float, hi: float) -> float:
    """Clamp and linearly normalize value to [0, 1]."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _text_contains_any(text: str, token_set: set) -> bool:
    """Check if lowercased text contains any token from the set."""
    t = text.lower()
    return any(tok in t for tok in token_set)


def _count_matches(text: str, token_set: set) -> int:
    """Count how many unique tokens from token_set appear in text."""
    t = text.lower()
    return sum(1 for tok in token_set if tok in t)


def _safe_list(val) -> list:
    """Always returns a list — handles None, strings, or other wrong types."""
    if isinstance(val, list):
        return val
    return []  # None, strings, ints, dicts all become empty list


def _safe_dict(val) -> dict:
    """Always returns a dict — handles None or wrong types."""
    if isinstance(val, dict):
        return val
    return {}


def _safe_num(val, default=0) -> float:
    """Always returns a float — handles None, strings, or invalid types."""
    try:
        result = float(val)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_str(val, default="") -> str:
    """Always returns a str — handles None or non-string types."""
    if val is None:
        return default
    try:
        return str(val)
    except Exception:
        return default


def _build_text_blob(candidate: dict) -> str:
    """Build a single lowercase blob of all text in a candidate profile."""
    parts: list = []
    p = _safe_dict(candidate.get("profile"))
    parts.append(_safe_str(p.get("headline")))
    parts.append(_safe_str(p.get("summary")))
    parts.append(_safe_str(p.get("current_title")))
    for ch in _safe_list(candidate.get("career_history")):
        parts.append(_safe_str(ch.get("title")))
        parts.append(_safe_str(ch.get("description")))
        parts.append(_safe_str(ch.get("company")))
    for sk in _safe_list(candidate.get("skills")):
        parts.append(_safe_str(sk.get("name")))
    return " ".join(parts).lower()


def _build_career_text(candidate: dict) -> str:
    """Career descriptions + titles only — harder to keyword-stuff than skills."""
    parts: list = []
    for ch in _safe_list(candidate.get("career_history")):
        parts.append(_safe_str(ch.get("title")))
        parts.append(_safe_str(ch.get("description")))
    return " ".join(parts).lower()


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION  (run FIRST — zeros everything)
# ─────────────────────────────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> Tuple[bool, str]:
    """
    Hard rules for impossible profiles.
    Returns (is_honeypot, reason).
    
    These rules are intentionally strict — ~80 honeypots in 100K candidates means
    we expect <0.1% false positives.
    """
    profile = _safe_dict(candidate.get("profile"))
    career = candidate.get("career_history", [])
    skills = _safe_list(candidate.get("skills"))
    yoe = _safe_num(profile.get("years_of_experience"), 0)
    yoe_months = yoe * 12

    # Rule 1: Total consecutive career duration massively exceeds stated YoE.
    # Allow 18 months of legitimate overlap (e.g., part-time + full-time).
    total_months = sum(int(j.get("duration_months", 0) or 0) for j in career)
    if yoe > 0 and total_months > yoe_months + 18:
        return True, (
            f"Total job durations ({total_months}mo) exceed stated experience "
            f"({yoe:.0f}yrs = {yoe_months:.0f}mo) by >{18}mo"
        )

    # Rule 2: ≥3 expert-proficiency skills with 0 duration_months.
    zero_dur_experts = [
        s.get("name", "unknown") for s in skills
        if s.get("proficiency") == "expert" and int(s.get("duration_months", 0) or 0) == 0
    ]
    if len(zero_dur_experts) >= 3:
        return True, (
            f"{len(zero_dur_experts)} expert skills with 0 months usage: "
            f"{', '.join(zero_dur_experts[:3])}"
        )

    # Rule 3: A single skill duration > YoE (you can't use a skill longer than you've worked).
    max_skill_months = max(
        (int(s.get("duration_months", 0) or 0) for s in skills), default=0
    )
    if yoe > 0 and max_skill_months > yoe_months * 1.4:
        return True, (
            f"Single skill duration {max_skill_months}mo exceeds YoE "
            f"{yoe:.0f}yrs × 1.4 = {yoe_months*1.4:.0f}mo"
        )

    # Rule 4: Implausibly high endorsements for very junior candidate.
    total_endorsements = sum(int(s.get("endorsements", 0) or 0) for s in skills)
    if total_endorsements > 5000 and yoe < 3:
        return True, (
            f"{total_endorsements} endorsements with only {yoe:.0f} years experience"
        )

    # Rule 5: Computed job duration differs drastically from stated duration_months.
    for ch in career:
        start_str = ch.get("start_date")
        end_str = ch.get("end_date")
        stated_dur = int(ch.get("duration_months", 0) or 0)
        if not start_str or not end_str or stated_dur == 0:
            continue
        try:
            s = datetime.strptime(start_str, "%Y-%m-%d").date()
            e = datetime.strptime(end_str, "%Y-%m-%d").date()
            computed = (e.year - s.year) * 12 + (e.month - s.month)
            if abs(computed - stated_dur) > 24:
                return True, (
                    f"Job at {ch.get('company','?')}: stated {stated_dur}mo "
                    f"but dates compute to {computed}mo (diff={abs(computed-stated_dur)}mo)"
                )
        except Exception:
            pass

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION A: CAREER SUBSTANCE  (weight 0.40)
# The most important signal — did this person actually build ML/AI/search systems?
# ─────────────────────────────────────────────────────────────────────────────

def score_career_substance(candidate: dict, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Evaluates the CAREER HISTORY and PROFILE for AI/ML/search substance.
    Uses jd_config for domain keywords and hard-negative title tokens if provided.
    """
    # Resolve JD-specific constants (fall back to module-level if no config)
    hard_neg = (cfg or {}).get("hard_negative_title_tokens") or HARD_NEGATIVE_TITLE_TOKENS
    prod_signals = (cfg or {}).get("required_skills") or PRODUCTION_RETRIEVAL_SIGNALS
    consulting_flag = (cfg or {}).get("avoid_consulting_only", True)

    career = candidate.get("career_history", [])
    profile = _safe_dict(candidate.get("profile"))
    if not career:
        return 0.0

    current_title = _safe_str(profile.get("current_title")).lower()
    if _text_contains_any(current_title, hard_neg):
        return 0.04

    # ── Signal 1: Profile headline and summary (unique per candidate) ────────
    headline = (profile.get("headline", "") or "").lower()
    summary = (profile.get("summary", "") or "").lower()
    current_title = (profile.get("current_title", "") or "").lower()
    profile_text = current_title + " " + headline + " " + summary

    # Strong AI/ML title/headline signals
    STRONG_TITLE_SIGNALS = {
        "ml engineer", "machine learning engineer", "ai engineer", "applied ml",
        "applied ai", "nlp engineer", "search engineer", "ranking engineer",
        "recommendation", "data scientist", "research engineer",
        "retrieval", "applied scientist", "founding ml", "founding ai",
    }
    MODERATE_TITLE_SIGNALS = {
        "data engineer", "backend engineer", "software engineer", "full stack",
        "platform engineer", "senior engineer", "staff engineer",
    }

    profile_title_score = 0.0
    if _text_contains_any(profile_text, STRONG_TITLE_SIGNALS):
        profile_title_score = 1.0
    elif _text_contains_any(profile_text, MODERATE_TITLE_SIGNALS):
        profile_title_score = 0.4

    # Profile summary mentions production AI/ML work
    profile_prod_hits = _count_matches(summary + " " + headline, PRODUCTION_RETRIEVAL_SIGNALS)
    profile_prod_score = min(1.0, profile_prod_hits / 3.0)

    # ── Signal 2: Career job titles ─────────────────────────────────────────
    ai_ml_title_months = 0.0
    hard_negative_roles = 0
    product_company_score = 0.0
    consulting_only = True
    career_desc_prod_hits_total = 0.0

    for ch in career:
        title = (ch.get("title", "") or "").lower()
        desc = (ch.get("description", "") or "").lower()
        company = (ch.get("company", "") or "").lower()
        industry = (ch.get("industry", "") or "").lower()
        company_size = ch.get("company_size", "")
        duration = int(ch.get("duration_months", 0) or 0)

        # Hard negative: clearly non-AI/ML role
        if _text_contains_any(title, hard_neg):
            hard_negative_roles += 1
            continue

        # Score based on JOB TITLE (most reliable signal)
        if _text_contains_any(title, STRONG_TITLE_SIGNALS):
            ai_ml_title_months += duration * 1.0
        elif _text_contains_any(title, MODERATE_TITLE_SIGNALS):
            ai_ml_title_months += duration * 0.3
        elif any(kw in title for kw in ["engineer", "developer", "scientist"]):
            ai_ml_title_months += duration * 0.15

        # Secondary: career description hits
        prod_hits = _count_matches(desc, prod_signals)
        career_desc_prod_hits_total += min(1.0, prod_hits / 3.0)

        # Company type
        is_consulting = consulting_flag and any(cf in company for cf in CONSULTING_FIRMS)
        if not is_consulting:
            consulting_only = False
            size_scores = {
                "1-10": 1.2, "11-50": 1.3, "51-200": 1.2,
                "201-500": 1.0, "501-1000": 0.9, "1001-5000": 0.8,
                "5001-10000": 0.7, "10001+": 0.6,
            }
            product_company_score += size_scores.get(company_size, 0.7)
        else:
            product_company_score += 0.2  # consulting firms get minimum credit

        # Industry bonus
        if _text_contains_any(industry, PRODUCT_INDUSTRIES):
            product_company_score += 0.25

    # ── Signal 3: High-quality skills (endorsements + duration) ─────────────
    skills = _safe_list(candidate.get("skills"))
    rs = _safe_dict(candidate.get("redrob_signals"))
    assessments = rs.get("skill_assessment_scores", {}) or {}

    high_quality_ai_skills = 0.0
    for s in skills:
        name = (s.get("name", "") or "").lower()
        endorsements = int(s.get("endorsements", 0) or 0)
        duration = int(s.get("duration_months", 0) or 0)
        proficiency = s.get("proficiency", "beginner")

        if not _text_contains_any(name, prod_signals):
            continue

        # Trust: endorsements + duration are hard to fake
        trust = 0.0
        if endorsements >= 30:
            trust += 0.50
        elif endorsements >= 15:
            trust += 0.35
        elif endorsements >= 5:
            trust += 0.20
        elif endorsements >= 1:
            trust += 0.10

        if duration >= 24:
            trust += 0.40
        elif duration >= 12:
            trust += 0.25
        elif duration >= 6:
            trust += 0.10

        # Objective assessment score from platform (best signal of all)
        for assess_name, assess_val in assessments.items():
            if any(kw in assess_name.lower() for kw in name.split()):
                av = float(assess_val or 0)
                if av >= 70:
                    trust += 0.30
                elif av >= 50:
                    trust += 0.15
                break

        prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9, "expert": 1.0}.get(proficiency, 0.5)
        high_quality_ai_skills += trust * prof_w

    # ── Compute sub-scores ─────────────────────────────────────────────────
    # Hard negative career penalty
    if career and (hard_negative_roles / len(career)) > 0.6:
        return 0.04  # Keyword stuffer with wrong career

    consulting_penalty = 0.55 if (consulting_flag and consulting_only and len(career) >= 2) else 1.0

    ai_months_score = min(1.0, ai_ml_title_months / 60.0)  # 5 yrs in AI/ML titles = max
    product_score = min(1.0, product_company_score / (len(career) * 1.5)) if career else 0.0
    desc_score = min(1.0, career_desc_prod_hits_total / len(career)) if career else 0.0
    skills_bonus = min(0.3, high_quality_ai_skills * 0.1)

    # Combine
    career_score = (
        profile_title_score * 0.30 +   # current title/headline — who are you?
        profile_prod_score  * 0.15 +   # summary mentions production AI work
        ai_months_score     * 0.30 +   # time spent in AI/ML titled roles
        product_score       * 0.15 +   # product company vs consulting
        desc_score          * 0.10     # career description matches (secondary)
    ) + skills_bonus

    career_score *= consulting_penalty
    return max(0.0, min(1.0, career_score))




# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION B: SKILL CREDIBILITY  (weight 0.22)
# Skills are self-reported — we trust them only when corroborated
# ─────────────────────────────────────────────────────────────────────────────

def score_skill_credibility(candidate: dict, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Scores skills with a CORROBORATION multiplier.
    A skill gets full credit only when:
    1. It appears in the skills list with real duration/endorsements, AND
    2. The career descriptions actually reference the same domain
    
    This prevents keyword stuffers from winning.
    """
    skills = _safe_list(candidate.get("skills"))
    career_text = _build_career_text(candidate)
    signals = candidate.get("redrob_signals", {})

    # The key must-have skill families for this JD
    SKILL_FAMILIES = {
        "embeddings_retrieval": {
            "keywords": {"embedding", "sentence-transformer", "bge", "e5", "openai embeddings",
                         "semantic search", "dense retrieval", "vector search", "dpr"},
            "career_corroborators": {"embedding", "retrieval", "semantic search", "vector",
                                     "sentence-transformer", "dense"},
        },
        "vector_db": {
            "keywords": {"pinecone", "weaviate", "qdrant", "milvus", "faiss",
                         "opensearch", "elasticsearch", "chroma", "pgvector"},
            "career_corroborators": {"pinecone", "weaviate", "qdrant", "milvus", "faiss",
                                     "opensearch", "elasticsearch", "vector store", "index"},
        },
        "ranking_ir": {
            "keywords": {"ranking", "bm25", "learning to rank", "ltr", "information retrieval",
                         "reranking", "ndcg", "mrr", "map"},
            "career_corroborators": {"rank", "bm25", "ltr", "ndcg", "mrr", "relevance",
                                     "search quality", "information retrieval"},
        },
        "llm_nlp": {
            "keywords": {"llm", "large language model", "transformer", "bert", "nlp",
                         "natural language processing", "fine-tuning", "lora", "qlora",
                         "rag", "retrieval augmented"},
            "career_corroborators": {"llm", "bert", "transformer", "nlp", "fine-tun",
                                     "language model", "rag", "retrieval augmented"},
        },
        "python_mlops": {
            "keywords": {"python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
                         "hugging face", "huggingface"},
            "career_corroborators": {"python", "pytorch", "tensorflow", "sklearn",
                                     "hugging face", "huggingface"},
        },
        "eval_frameworks": {
            "keywords": {"a/b testing", "ab testing", "offline evaluation", "online evaluation",
                         "eval framework", "evaluation", "ndcg", "mrr"},
            "career_corroborators": {"a/b test", "evaluation", "ndcg", "mrr",
                                     "offline eval", "online eval"},
        },
    }

    family_scores: Dict[str, float] = {}

    # Score each skill family
    for family_name, family_def in SKILL_FAMILIES.items():
        skill_credit = 0.0
        career_corroborated = any(
            corr in career_text for corr in family_def["career_corroborators"]
        )

        for skill in skills:
            sname = _safe_str(skill.get("name")).lower()
            # Does this skill match the family?
            if not any(kw in sname or sname in kw for kw in family_def["keywords"]):
                continue

            proficiency = skill.get("proficiency", "beginner")
            endorsements = int(skill.get("endorsements", 0) or 0)
            duration = int(skill.get("duration_months", 0) or 0)

            # Base from proficiency
            prof_base = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}.get(
                proficiency, 0.4
            )

            # Trust multiplier: endorsements + duration + career corroboration
            trust = 0.2  # base
            if endorsements >= 20:
                trust += 0.35
            elif endorsements >= 10:
                trust += 0.25
            elif endorsements >= 3:
                trust += 0.15
            elif endorsements >= 1:
                trust += 0.05

            if duration >= 24:
                trust += 0.30
            elif duration >= 12:
                trust += 0.20
            elif duration >= 6:
                trust += 0.10

            if career_corroborated:
                trust += 0.20  # Career description backs this up

            trust = min(1.0, trust)

            # Skill assessment bonus (objective score from platform)
            assessments = signals.get("skill_assessment_scores", {}) or {}
            for assess_name, assess_val in assessments.items():
                if any(kw in assess_name.lower() for kw in family_def["keywords"]):
                    if float(assess_val or 0) >= 70:
                        trust = min(1.0, trust + 0.10)
                    break

            skill_credit = max(skill_credit, prof_base * trust)

        # If career doesn't corroborate this family at all,
        # reduce the maximum possible credit (keyword stuffing mitigation)
        if not career_corroborated and skill_credit > 0:
            skill_credit *= 0.35  # steep penalty for uncorroborated claims

        family_scores[family_name] = skill_credit

    # Weights per family (must_haves weighted highest)
    weights = {
        "embeddings_retrieval": 0.28,
        "vector_db":            0.20,
        "ranking_ir":           0.20,
        "llm_nlp":              0.15,
        "python_mlops":         0.12,
        "eval_frameworks":      0.05,
    }

    total = sum(family_scores.get(f, 0.0) * w for f, w in weights.items())
    return min(1.0, total)


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION C: EXPERIENCE QUALITY  (weight 0.18)
# Right kind of experience at the right stage of career
# ─────────────────────────────────────────────────────────────────────────────

def score_experience_quality(candidate: dict, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Evaluates whether the AMOUNT and TYPE of experience fits the role.
    Uses jd_config for YoE range if provided.
    """
    exp_ideal_min = (cfg or {}).get("exp_ideal_min", 5)
    exp_ideal_max = (cfg or {}).get("exp_ideal_max", 9)
    exp_min       = (cfg or {}).get("exp_min", 4)
    exp_max       = (cfg or {}).get("exp_max", 12)
    profile = _safe_dict(candidate.get("profile"))
    career = candidate.get("career_history", [])
    yoe = _safe_num(profile.get("years_of_experience"), 0)

    # YoE fit — use JD-configured range
    if exp_ideal_min <= yoe <= exp_ideal_max:
        yoe_score = 1.0
    elif exp_min <= yoe < exp_ideal_min:
        yoe_score = 0.80
    elif exp_ideal_max < yoe <= exp_max:
        yoe_score = 0.80
    elif (exp_min - 1) <= yoe < exp_min:
        yoe_score = 0.55
    elif yoe > exp_max:
        yoe_score = 0.60
    elif (exp_min - 2) <= yoe < (exp_min - 1):
        yoe_score = 0.30
    else:
        yoe_score = 0.10

    # Career stability check (JD explicitly penalizes title-chasers)
    # Check: companies per year of experience
    unique_companies = len(set(ch.get("company", "").lower() for ch in career))
    if yoe > 0:
        job_density = unique_companies / yoe  # jobs per year
        if job_density <= 0.5:  # very stable
            stability_score = 1.0
        elif job_density <= 0.8:
            stability_score = 0.85
        elif job_density <= 1.2:
            stability_score = 0.70
        elif job_density <= 1.8:
            stability_score = 0.50
        else:
            stability_score = 0.30  # job-hopper
    else:
        stability_score = 0.5

    # Tenure at current role (are they settled vs just started?)
    current_tenure_score = 0.5
    for ch in career:
        if ch.get("is_current"):
            months = int(ch.get("duration_months", 0) or 0)
            if months >= 30:
                current_tenure_score = 1.0
            elif months >= 18:
                current_tenure_score = 0.85
            elif months >= 12:
                current_tenure_score = 0.70
            elif months >= 6:
                current_tenure_score = 0.55
            else:
                current_tenure_score = 0.40
            break

    # Research vs applied penalty
    # The JD: "pure research backgrounds — academic labs or research-only roles → reject"
    research_penalty = 0.0
    for ch in career:
        title = ch.get("title", "").lower()
        industry = ch.get("industry", "").lower()
        if ("research" in title or "phd" in title or "postdoc" in title) and "engineer" not in title:
            research_penalty += 0.15
        if "research" in industry or "academia" in industry or "university" in industry:
            research_penalty += 0.10
    research_penalty = min(0.5, research_penalty)

    score = (
        yoe_score * 0.50 +
        stability_score * 0.25 +
        current_tenure_score * 0.25
    ) - research_penalty

    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION D: BEHAVIORAL AVAILABILITY  (weight 0.15)
# The JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
# and has 5% recruiter response rate is not actually available"
# ─────────────────────────────────────────────────────────────────────────────

def score_behavioral_availability(candidate: dict) -> float:
    """
    Computes an availability score from ALL platform behavioral signals.
    Uses all 23 redrob_signals fields — none left on the table.
    Range: 0.15 (completely unreachable) to 1.0 (actively looking, responsive).
    """
    rs = _safe_dict(candidate.get("redrob_signals"))

    # ── Signal 1: Last active date (recency) ─────────────────────────────────
    days_inactive = _days_since(str(rs.get("last_active_date", "2020-01-01") or "2020-01-01"))
    if days_inactive <= 7:    recency = 1.0
    elif days_inactive <= 30: recency = 0.90
    elif days_inactive <= 60: recency = 0.80
    elif days_inactive <= 90: recency = 0.65
    elif days_inactive <= 180: recency = 0.40
    elif days_inactive <= 365: recency = 0.20
    else:                     recency = 0.10

    # ── Signal 2: Open-to-work flag + applications submitted ─────────────────
    open_to_work = 1.0 if rs.get("open_to_work_flag") else 0.45
    apps_submitted = min(1.0, (float(rs.get("applications_submitted_30d", 0) or 0) / 5.0))
    activity = open_to_work * 0.65 + apps_submitted * 0.35

    # ── Signal 3: Recruiter responsiveness ───────────────────────────────────
    rrr = _safe_num(rs.get("recruiter_response_rate"), 0.3)
    resp_hours = _safe_num(rs.get("avg_response_time_hours"), 72)
    if rrr >= 0.70 and resp_hours <= 24:  responsiveness = 1.0
    elif rrr >= 0.50:                     responsiveness = 0.80
    elif rrr >= 0.30:                     responsiveness = 0.60
    elif rrr >= 0.15:                     responsiveness = 0.40
    else:                                 responsiveness = 0.20

    # ── Signal 4: Interview completion rate ──────────────────────────────────
    icr = float(rs.get("interview_completion_rate", 0.5) or 0.5)
    interview_score = 0.3 + icr * 0.7

    # ── Signal 5: Notice period ───────────────────────────────────────────────
    notice = int(_safe_num(rs.get("notice_period_days"), 90))
    if notice <= 15:    notice_score = 1.0
    elif notice <= 30:  notice_score = 0.95
    elif notice <= 60:  notice_score = 0.80
    elif notice <= 90:  notice_score = 0.60
    elif notice <= 120: notice_score = 0.40
    else:               notice_score = 0.25

    # ── Signal 6: Profile completeness ───────────────────────────────────────
    completeness = float(rs.get("profile_completeness_score", 50) or 50)
    completeness_score = completeness / 100.0

    # ── Signal 7: External recruiter demand (saved + profile views + search appearances)
    saved = float(rs.get("saved_by_recruiters_30d", 0) or 0)
    views = float(rs.get("profile_views_received_30d", 0) or 0)
    appearances = float(rs.get("search_appearance_30d", 0) or 0)
    # Log-scale demand: each signal independently contributes
    demand_saved = min(1.0, math.log1p(max(0, saved)) / math.log1p(15))
    demand_views = min(1.0, math.log1p(max(0, views)) / math.log1p(50))
    demand_appearances = min(1.0, math.log1p(max(0, appearances)) / math.log1p(100))
    demand = demand_saved * 0.50 + demand_views * 0.30 + demand_appearances * 0.20

    # ── Signal 8: Network strength (connection_count) ────────────────────────
    # A well-networked candidate is more visible and credible
    connections = float(rs.get("connection_count", 0) or 0)
    network_score = min(1.0, math.log1p(max(0, connections)) / math.log1p(500))

    # ── Signal 9: Total endorsements received (platform-wide trust signal) ───
    total_endorsements = float(rs.get("endorsements_received", 0) or 0)
    endorsement_score = min(1.0, math.log1p(max(0, total_endorsements)) / math.log1p(200))

    # ── Signal 10: Platform tenure (signup_date) ──────────────────────────────
    # Longer tenure = committed platform user (not a new account created to spam)
    days_on_platform = _days_since(str(rs.get("signup_date", "2024-01-01") or "2024-01-01"))
    # Negative = older signup (more days since they joined = longer tenure)
    if days_on_platform >= 730:   tenure_score = 1.0   # 2+ years on platform
    elif days_on_platform >= 365: tenure_score = 0.80  # 1+ year
    elif days_on_platform >= 180: tenure_score = 0.65  # 6+ months
    elif days_on_platform >= 90:  tenure_score = 0.50  # 3+ months
    else:                         tenure_score = 0.30  # brand new account

    # ── Signal 11: Preferred work mode alignment with JD ─────────────────────
    # JD says "hybrid — flexible cadence" so hybrid/flexible = full score
    work_mode = str(rs.get("preferred_work_mode", "") or "").lower()
    if work_mode in ("hybrid", "flexible"):
        work_mode_score = 1.0
    elif work_mode == "onsite":
        work_mode_score = 0.85  # fine, JD is mostly in-office
    elif work_mode == "remote":
        work_mode_score = 0.60  # JD doesn't support full remote
    else:
        work_mode_score = 0.75  # unknown — neutral

    # ── Composite behavioral score ────────────────────────────────────────────
    score = (
        recency            * 0.22 +
        activity           * 0.18 +
        responsiveness     * 0.16 +
        notice_score       * 0.10 +
        interview_score    * 0.08 +
        demand             * 0.08 +
        completeness_score * 0.06 +
        network_score      * 0.04 +
        endorsement_score  * 0.03 +
        tenure_score       * 0.03 +
        work_mode_score    * 0.02
    )

    return max(0.15, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION E: LOCATION & LOGISTICS  (weight 0.05)
# ─────────────────────────────────────────────────────────────────────────────

def score_location(candidate: dict, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Location scoring — uses JD-configured preferred/acceptable cities if provided."""
    preferred = (cfg or {}).get("preferred_locations") or PREFERRED_LOCATIONS
    acceptable = (cfg or {}).get("acceptable_locations") or ACCEPTABLE_LOCATIONS
    profile = _safe_dict(candidate.get("profile"))
    rs = _safe_dict(candidate.get("redrob_signals"))

    location = str(profile.get("location", "") or "").lower()
    country = str(profile.get("country", "") or "").lower()
    willing_to_relocate = bool(rs.get("willing_to_relocate", False))

    if any(city in location for city in preferred):
        return 1.0
    elif any(city in location for city in acceptable):
        return 0.85
    elif country == "india":
        return 0.65 if willing_to_relocate else 0.55
    elif willing_to_relocate:
        return 0.35
    else:
        return 0.20


# ─────────────────────────────────────────────────────────────────────────────
# CURRENT COMPANY QUALITY BONUS
# Uses: current_company, current_company_size, current_industry
# ─────────────────────────────────────────────────────────────────────────────

# Top-tier product companies (Indian & global) relevant to this role
_PRESTIGE_COMPANIES = {
    "google", "meta", "amazon", "microsoft", "apple", "netflix", "openai",
    "deepmind", "anthropic", "stripe", "airbnb", "uber", "linkedin",
    "flipkart", "paytm", "swiggy", "zomato", "razorpay", "cred",
    "meesho", "freshworks", "zoho", "ola", "nykaa", "phonepe",
    "observe.ai", "mad street den", "sarvam", "krutrim", "saarthi",
    "aganitha", "leadsquared", "darwinbox", "zepto", "blinkit",
    "dream11", "mpl", "games24x7", "juspay", "smallcase", "groww",
}
_CONSULTING_COMPANIES = CONSULTING_FIRMS  # reuse existing set

def compute_current_company_bonus(candidate: dict) -> float:
    """
    Bonus based on current employer quality.
    Uses profile.current_company, profile.current_company_size,
    profile.current_industry — all unused until now.
    Max: +0.04
    """
    profile = _safe_dict(candidate.get("profile"))
    company = (profile.get("current_company") or "").lower()
    size = (profile.get("current_company_size") or "").lower()
    industry = (profile.get("current_industry") or "").lower()

    # Prestige company bonus
    if any(p in company for p in _PRESTIGE_COMPANIES):
        company_score = 0.04
    elif any(cf in company for cf in _CONSULTING_COMPANIES):
        company_score = 0.00   # consulting firm — no bonus (penalty already in career_substance)
    else:
        company_score = 0.02   # generic product company

    # Size signal: we want Series A–C style companies (51-1000 range)
    size_bonus = {
        "1-10": 0.005, "11-50": 0.010, "51-200": 0.015,
        "201-500": 0.012, "501-1000": 0.010, "1001-5000": 0.008,
        "5001-10000": 0.005, "10001+": 0.003,
    }.get(profile.get("current_company_size", ""), 0.005)

    # Industry signal: tech/AI/ML/SaaS = good fit
    industry_bonus = 0.005 if _text_contains_any(industry, PRODUCT_INDUSTRIES) else 0.0

    return min(0.04, company_score + size_bonus + industry_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# CERTIFICATIONS BONUS
# Uses the certifications[] array — entirely unused until now
# ─────────────────────────────────────────────────────────────────────────────

# ML/AI certifications that indicate genuine investment in the domain
_ML_CERT_KEYWORDS = {
    "machine learning", "deep learning", "tensorflow", "pytorch", "aws",
    "google cloud", "gcp", "azure", "data science", "nlp", "ai",
    "coursera", "deeplearning.ai", "fast.ai", "mlops", "sagemaker",
    "vertex ai", "databricks", "spark", "hugging face", "transformers",
    "llm", "generative ai", "rag", "langchain", "vector", "embedding",
}

def compute_certifications_bonus(candidate: dict) -> float:
    """
    Bonus for relevant ML/AI certifications.
    A candidate who invested time in certifications shows domain commitment.
    Max: +0.04
    """
    certifications = _safe_list(candidate.get("certifications"))
    if not certifications:
        return 0.0

    bonus = 0.0
    for cert in certifications:
        name = (cert.get("name") or "").lower()
        issuer = (cert.get("issuer") or "").lower()
        year = int(cert.get("year") or 0)
        combined = name + " " + issuer

        if not _text_contains_any(combined, _ML_CERT_KEYWORDS):
            continue   # unrelated cert (PMP, Six Sigma, etc.) — no bonus

        # Recency bonus: certs from last 4 years score higher
        recency_mult = 1.0 if year >= 2021 else (0.7 if year >= 2018 else 0.4)

        # Issuer prestige
        if any(kw in issuer for kw in ["google", "aws", "amazon", "microsoft", "coursera", "deeplearning"]):
            cert_val = 0.015
        else:
            cert_val = 0.008

        bonus += cert_val * recency_mult

    return min(0.04, bonus)


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE BONUS
# Uses the languages[] array — entirely unused until now
# ─────────────────────────────────────────────────────────────────────────────

def compute_language_bonus(candidate: dict) -> float:
    """
    Small bonus for professional English proficiency.
    This role requires strong written communication (JD: "we write a lot").
    Also, multi-lingual candidates show broader capability.
    Max: +0.02
    """
    languages = _safe_list(candidate.get("languages"))
    if not languages:
        return 0.0

    bonus = 0.0
    for lang in languages:
        language = (lang.get("language") or "").lower()
        proficiency = (lang.get("proficiency") or "").lower()

        if language == "english":
            if proficiency in ("native", "professional"):
                bonus += 0.015  # strong English = can write well
            elif proficiency == "conversational":
                bonus += 0.005
        else:
            # Additional language = small diversity/communication bonus
            if proficiency in ("native", "professional"):
                bonus += 0.003

    return min(0.02, bonus)


# ─────────────────────────────────────────────────────────────────────────────
# SALARY ALIGNMENT CHECK
# Uses expected_salary_range_inr_lpa — avoids extremely misaligned candidates
# ─────────────────────────────────────────────────────────────────────────────

def compute_salary_alignment(candidate: dict, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Checks if salary expectations are within the JD budget range.
    Returns a multiplier: 0.88 (misaligned) to 1.0 (aligned).
    Uses jd_config salary_range_inr_lpa if provided.
    """
    sal_budget_max = (cfg or {}).get("sal_max", 130)
    sal_budget_min = (cfg or {}).get("sal_min", 0)
    rs = _safe_dict(candidate.get("redrob_signals"))
    salary_range = rs.get("expected_salary_range_inr_lpa") or {}
    if not salary_range:
        return 1.0  # unknown — no penalty

    sal_min = float(salary_range.get("min") or 0)
    sal_max = float(salary_range.get("max") or 0)

    if sal_min == 0 and sal_max == 0:
        return 1.0  # not specified

    sal_mid = (sal_min + sal_max) / 2.0
    budget_mid = (sal_budget_min + sal_budget_max) / 2.0
    budget_lo = sal_budget_min * 0.8 if sal_budget_min > 0 else 0
    budget_hi = sal_budget_max

    if budget_lo <= sal_mid <= budget_hi:
        return 1.0   # within budget
    elif sal_mid < budget_lo:
        return 0.97  # below minimum — may be under-experienced
    elif sal_mid <= budget_hi * 1.15:
        return 0.95  # slightly over budget
    elif sal_mid <= budget_hi * 1.4:
        return 0.88  # significantly over budget
    else:
        return 0.82  # way out of range


# ─────────────────────────────────────────────────────────────────────────────
# HIDDEN GEM DETECTION
# The hardest trap: plain-language engineers who never say "RAG" or "Pinecone"
# ─────────────────────────────────────────────────────────────────────────────

def detect_hidden_gem(candidate: dict, career_score: float, skill_score: float) -> bool:
    """
    Flags candidates who would be missed by a keyword ranker but have real fit.
    
    A hidden gem is someone who:
    1. Has strong career substance score (real work evidence) but
    2. Low/medium skill score (didn't keyword-stuff their skills list), AND
    3. Career descriptions show production system ownership
    """
    career = candidate.get("career_history", [])
    profile = _safe_dict(candidate.get("profile"))
    yoe = _safe_num(profile.get("years_of_experience"), 0)

    # Gap between career substance and skill listing — sign of plain-language engineer
    career_skill_gap = career_score - skill_score

    # Check for ownership language in career descriptions
    ownership_indicators = [
        "built", "designed", "architected", "owned", "led", "shipped",
        "deployed", "developed", "created", "launched",
    ]
    
    system_indicators = [
        "search", "recommend", "rank", "retrieval", "pipeline", "platform",
        "system", "infrastructure", "service", "engine",
    ]

    ownership_hits = 0
    system_hits = 0
    for ch in career:
        desc = ch.get("description", "").lower()
        ownership_hits += sum(1 for w in ownership_indicators if w in desc)
        system_hits += sum(1 for w in system_indicators if w in desc)

    # Anti-false-positive: exclude pure CV/Speech specialists from being gems
    # for an NLP/retrieval role. Counts how many wrong-domain vs right-domain
    # signals appear across ALL career descriptions.
    wrong_domain_signals = (
        "computer vision", "image recognition", "object detection",
        "image classification", "image segmentation", "resnet", "yolo",
        "speech recognition", "voice recognition", "text to speech",
        "robotics", "slam", "lidar",
    )
    right_domain_signals = (
        "retrieval", "search", "recommend", "nlp",
        "natural language", "embedding", "vector",
        "rag", "language model", "semantic", "ranking",
        "relevance", "query", "text classification",
    )
    full_text = " ".join(
        (ch.get("description", "") or "").lower() for ch in career
    ) + " " + (profile.get("headline", "") or "").lower()

    wrong_hits = sum(1 for s in wrong_domain_signals if s in full_text)
    right_hits  = sum(1 for s in right_domain_signals if s in full_text)

    # Only disqualify when wrong-domain STRICTLY dominates right-domain
    # (a candidate who mentions CV briefly but has more NLP/retrieval signals is still a gem)
    if wrong_hits > 0 and wrong_hits > right_hits:
        return False

    # Criteria: meaningful career score, career-skill gap, ownership evidence, right YoE
    is_gem = (
        career_score >= 0.50 and       # actual strong production evidence in career
        skill_score < 0.35 and         # didn't keyword stuff their skills
        career_skill_gap >= 0.20 and   # career >> skills list (plain-language engineer)
        ownership_hits >= 4 and        # owns real systems
        system_hits >= 3 and           # builds real systems
        yoe >= 4.0                     # enough experience to be the right hire
    )

    return is_gem


# ─────────────────────────────────────────────────────────────────────────────
# DECEPTION SIGNAL ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def assess_deception_signals(candidate: dict) -> Dict[str, Any]:
    """
    Assesses multiple deception patterns.
    Returns a dict with flags and an overall deception_risk level.
    """
    profile = _safe_dict(candidate.get("profile"))
    career = candidate.get("career_history", [])
    skills = _safe_list(candidate.get("skills"))
    yoe = _safe_num(profile.get("years_of_experience"), 0)

    flags = []

    # 1. Vague ownership language in career descriptions
    vague_phrases = [
        "contributed to", "part of the team", "involved in", "supported",
        "assisted with", "helped with", "participated in",
    ]
    vague_count = 0
    for ch in career:
        desc = ch.get("description", "").lower()
        vague_count += sum(1 for p in vague_phrases if p in desc)
    if vague_count >= 4:
        flags.append("vague_ownership: high use of 'contributed to / part of team' language")

    # 2. Expert skills without any career/duration evidence
    skills_without_evidence = []
    career_text = _build_career_text(candidate)
    for s in skills:
        if s.get("proficiency") not in ("advanced", "expert"):
            continue
        name = s.get("name", "").lower()
        duration = int(s.get("duration_months", 0) or 0)
        endorsements = int(s.get("endorsements", 0) or 0)
        in_career = any(part in career_text for part in name.split())
        if duration < 6 and endorsements < 3 and not in_career:
            skills_without_evidence.append(s.get("name"))
    if len(skills_without_evidence) >= 5:
        flags.append(f"skill_without_evidence: {len(skills_without_evidence)} advanced/expert skills not supported by career or duration")

    # 3. Title inflation: senior title but junior description language
    for ch in career:
        title = ch.get("title", "").lower()
        desc = ch.get("description", "").lower()
        if any(kw in title for kw in ["senior", "lead", "staff", "principal", "head"]):
            ownership_words = sum(1 for w in ["designed", "architected", "led", "owned", "built"] if w in desc)
            junior_words = sum(1 for w in ["assist", "support", "learn", "shadow", "junior"] if w in desc)
            if junior_words > ownership_words and len(desc) > 100:
                flags.append(f"title_inflation: '{ch.get('title')}' title but description lacks ownership language")
                break

    # 4. YoE inconsistency
    career_months_total = sum(int(ch.get("duration_months", 0) or 0) for ch in career)
    if career_months_total > yoe * 12 + 12:
        flags.append(f"timeline_anomaly: career months ({career_months_total}) > stated YoE ({yoe:.0f}y)")

    # 5. Keyword-stuffing pattern: current title is clearly non-AI but many AI skills listed
    current_title = _safe_str(profile.get("current_title")).lower()
    is_non_ai_title = _text_contains_any(current_title, HARD_NEGATIVE_TITLE_TOKENS)
    ai_skill_count = sum(
        1 for s in skills
        if _text_contains_any(_safe_str(s.get("name")), PRODUCTION_RETRIEVAL_SIGNALS)
    )
    if is_non_ai_title and ai_skill_count >= 5:
        flags.append(f"keyword_stuffing: non-AI title ('{current_title}') with {ai_skill_count} AI/ML skills listed")

    # Overall risk level
    if len(flags) >= 3:
        risk = "high"
    elif len(flags) >= 1:
        risk = "medium"
    else:
        risk = "low"

    return {
        "deception_risk": risk,
        "flags": flags,
        "vague_ownership": vague_count >= 4,
        "skill_without_evidence": len(skills_without_evidence) >= 5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB & EXTERNAL VALIDATION BONUS
# ─────────────────────────────────────────────────────────────────────────────

def compute_github_bonus(candidate: dict) -> float:
    """GitHub activity bonus — real code evidence."""
    rs = _safe_dict(candidate.get("redrob_signals"))
    github = float(rs.get("github_activity_score", -1) or -1)
    if github < 0:
        return 0.0  # No GitHub — no penalty, no bonus
    elif github >= 75:
        return 0.06
    elif github >= 50:
        return 0.04
    elif github >= 25:
        return 0.02
    else:
        return 0.01


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION TIER BONUS
# Uses the schema's built-in `tier` field (tier_1 / tier_2 / tier_3 / tier_4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_education_bonus(candidate: dict) -> float:
    """
    Bonus for elite institution tier.
    Schema provides tier_1/tier_2/tier_3/tier_4 directly — no string matching needed.
    Also considers field_of_study (CS/ML/Stats = good) and degree level.
    Max bonus: +0.06
    """
    education = _safe_list(candidate.get("education"))
    if not education:
        return 0.0

    TIER_BONUS = {"tier_1": 0.06, "tier_2": 0.03, "tier_3": 0.01, "tier_4": 0.0, "unknown": 0.0}
    CS_FIELDS = {
        "computer science", "computer engineering", "information technology",
        "software engineering", "machine learning", "artificial intelligence",
        "data science", "statistics", "mathematics", "electrical engineering",
        "electronics", "information systems",
    }
    ADVANCED_DEGREES = {"m.tech", "mtech", "m.s", "ms ", "ms.", "phd", "ph.d", "m.e", "mba"}

    best_bonus = 0.0
    for edu in education:
        tier = (edu.get("tier") or "unknown").lower()
        field = (edu.get("field_of_study") or "").lower()
        degree = (edu.get("degree") or "").lower()

        tier_val = TIER_BONUS.get(tier, 0.0)
        # CS/ML field multiplier
        field_mult = 1.2 if any(f in field for f in CS_FIELDS) else 0.8
        # Advanced degree small boost
        degree_mult = 1.1 if any(d in degree for d in ADVANCED_DEGREES) else 1.0

        bonus = tier_val * field_mult * degree_mult
        best_bonus = max(best_bonus, bonus)

    return min(0.06, best_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE AUTHENTICITY & OFFER SERIOUSNESS BONUS
# Uses: offer_acceptance_rate, verified_email, verified_phone, linkedin_connected
# ─────────────────────────────────────────────────────────────────────────────

def compute_authenticity_bonus(candidate: dict) -> float:
    """
    Small bonus for verifiable, authentic, serious candidates.
    Max: +0.03 (keeps the signal modest — authenticity is table stakes, not a differentiator)
    """
    rs = _safe_dict(candidate.get("redrob_signals"))

    # Offer acceptance rate: high rate = serious candidate who follows through
    oar = float(rs.get("offer_acceptance_rate", -1) if rs.get("offer_acceptance_rate") is not None else -1)
    if oar >= 0.80:
        offer_score = 0.015
    elif oar >= 0.60:
        offer_score = 0.010
    elif oar == -1:  # no history — neutral
        offer_score = 0.007
    elif oar >= 0.30:
        offer_score = 0.004
    else:
        offer_score = 0.0  # consistently declines offers

    # Profile verification (each is a small authenticity signal)
    verified_email = bool(rs.get("verified_email", False))
    verified_phone = bool(rs.get("verified_phone", False))
    linkedin_connected = bool(rs.get("linkedin_connected", False))
    verification_score = (
        (0.005 if verified_email else 0.0) +
        (0.005 if verified_phone else 0.0) +
        (0.005 if linkedin_connected else 0.0)
    )

    return min(0.03, offer_score + verification_score)


# ─────────────────────────────────────────────────────────────────────────────
# CV / SPEECH / ROBOTICS SPECIALIST DETECTION
# JD explicitly says: "primary expertise in CV/speech/robotics without NLP/IR = disqualifier"
# ─────────────────────────────────────────────────────────────────────────────

CV_SPECIALTY_SIGNALS = {
    "computer vision", "object detection", "image segmentation", "image classification",
    "object recognition", "convolutional", "cnn", "yolo", "opencv", "image processing",
    "medical imaging", "satellite imagery", "video analysis",
}
SPEECH_SPECIALTY_SIGNALS = {
    "speech recognition", "asr", "text-to-speech", "tts", "speaker recognition",
    "voice recognition", "automatic speech", "kaldi", "wav2vec",
}
ROBOTICS_SPECIALTY_SIGNALS = {
    "robotics", "ros", "robot operating", "autonomous vehicle", "slam", "lidar",
    "sensor fusion", "motion planning", "actuator", "drone",
}
NLP_IR_COUNTER_SIGNALS = {
    "nlp", "natural language", "text", "retrieval", "search", "ranking",
    "recommendation", "language model", "llm", "bert", "transformer",
    "embedding", "sentiment", "ner", "named entity",
}


def compute_cv_speech_penalty(candidate: dict, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Penalty for candidates whose expertise is PRIMARILY in an unwanted specialty
    WITHOUT crossover into the role's required domain.
    Returns a negative adjustment (0.0 = no penalty, -0.12 = max penalty).
    Uses jd_config cv_specialty_penalty_keywords if provided.
    """
    text = _build_text_blob(candidate)
    # Use custom penalty keywords from JD if provided
    custom_penalty_kws = (cfg or {}).get("cv_specialty_penalty_keywords") or []
    if custom_penalty_kws:
        prod_signals = (cfg or {}).get("required_skills") or PRODUCTION_RETRIEVAL_SIGNALS
        wrong_domain_hits = sum(1 for s in custom_penalty_kws if s in text)
        nlp_hits = sum(1 for s in prod_signals if s in text)
    else:
        wrong_domain_hits = sum(1 for s in CV_SPECIALTY_SIGNALS if s in text) + \
                            sum(1 for s in SPEECH_SPECIALTY_SIGNALS if s in text) + \
                            sum(1 for s in ROBOTICS_SPECIALTY_SIGNALS if s in text)
        nlp_hits = sum(1 for s in NLP_IR_COUNTER_SIGNALS if s in text)

    # Only penalize if strong wrong-domain signal AND weak required-domain signal
    if wrong_domain_hits >= 4 and nlp_hits <= 2:
        return -0.12
    elif wrong_domain_hits >= 2 and nlp_hits == 0:
        return -0.06
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION F: STAR PREDICTOR  (weight 0.07)
# Learning velocity and scope expansion across the career arc.
# This is NOT the same as years_of_experience.
# Two engineers with 7 years each can have wildly different growth trajectories.
# ─────────────────────────────────────────────────────────────────────────────

# Seniority levels for progression scoring
_SENIORITY_MAP = {
    "intern": 0, "trainee": 0, "graduate": 1, "associate": 1, "junior": 1,
    "engineer": 2, "developer": 2, "analyst": 2, "scientist": 2, "specialist": 2,
    "senior": 3, "lead": 3, "sr.": 3, "sr ": 3,
    "staff": 4, "principal": 4, "architect": 4,
    "head": 5, "director": 5, "vp ": 5, "vp,": 5,
    "chief": 6, "cto": 6, "cpo": 6,
}

def _title_seniority(title: str) -> int:
    """Extract numeric seniority level from a job title."""
    t = title.lower()
    for token, level in sorted(_SENIORITY_MAP.items(), key=lambda x: -x[1]):
        if token in t:
            return level
    return 2  # default: IC engineer


def score_star_predictor(candidate: dict) -> float:
    """
    Measures career arc quality — learning velocity and scope expansion over time.

    Signals:
      1. Title progression       — did seniority grow chronologically?
      2. Company quality arc     — did employer quality/prestige grow?
      3. Responsibility scope    — do later descriptions show more ownership language?
      4. Domain depth            — sustained focus in AI/ML vs shallow exposure
      5. Lateral breadth         — added adjacent skills at each role (versatility)

    Returns 0.0 (stagnated) → 1.0 (strong upward arc across all dimensions).
    """
    career = candidate.get("career_history", [])
    profile = _safe_dict(candidate.get("profile"))

    if len(career) < 2:
        # Only one role: use YoE + current title level as proxy
        yoe = _safe_num(profile.get("years_of_experience"), 0)
        title = profile.get("current_title", "") or ""
        level = _title_seniority(title)
        return min(0.5, (level / 6.0) * 0.5 + min(0.25, yoe / 20.0))

    # Career is most-recent-first → reverse to get chronological order
    chronological = list(reversed(career))

    # ── Signal 1: Title progression ─────────────────────────────────────────
    levels = [_title_seniority(ch.get("title", "") or "") for ch in chronological]
    first_level = levels[0]
    last_level = levels[-1]
    peak_level = max(levels)

    if last_level > first_level:
        title_progression = min(1.0, (last_level - first_level) / 3.0)
    elif last_level == first_level and peak_level > first_level:
        title_progression = 0.3  # peaked then leveled — lateral move
    else:
        title_progression = 0.0  # flat or downward

    # ── Signal 2: Company quality arc ────────────────────────────────────────
    # Heuristic: product company > consulting firm; later companies > earlier
    CONSULTING_SET = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "tech mahindra", "hcl", "mphasis", "hexaware",
    }
    PRESTIGE_SET = {
        "google", "meta", "amazon", "microsoft", "apple", "netflix", "openai",
        "deepmind", "anthropic", "stripe", "airbnb", "uber", "linkedin",
        "flipkart", "paytm", "swiggy", "zomato", "razorpay", "cred",
        "meesho", "freshworks", "zoho", "ola", "nykaa", "phonepe",
    }

    def company_tier(company: str) -> int:
        c = (company or "").lower()
        if any(p in c for p in PRESTIGE_SET):
            return 3
        if any(cf in c for cf in CONSULTING_SET):
            return 1
        return 2  # generic product company

    company_tiers = [company_tier(ch.get("company", "")) for ch in chronological]
    first_tier = company_tiers[0]
    last_tier = company_tiers[-1]
    if last_tier > first_tier:
        company_arc = 1.0
    elif last_tier == first_tier:
        company_arc = 0.5
    else:
        company_arc = 0.1  # moved down in company quality

    # ── Signal 3: Ownership language growth ──────────────────────────────────
    # Count strong ownership words in early vs late career descriptions
    OWNERSHIP_WORDS = [
        "owned", "led", "built", "designed", "architected", "launched",
        "shipped", "created", "drove", "spearheaded", "founded",
    ]
    JUNIOR_WORDS = [
        "assisted", "supported", "contributed", "helped", "participated",
        "shadowed", "involved",
    ]

    def ownership_score_for(ch: dict) -> float:
        desc = (ch.get("description", "") or "").lower()
        if not desc:
            return 0.5
        own = sum(1 for w in OWNERSHIP_WORDS if w in desc)
        jun = sum(1 for w in JUNIOR_WORDS if w in desc)
        total = own + jun
        return (own / total) if total > 0 else 0.5

    early_ownership = sum(ownership_score_for(ch) for ch in chronological[:2]) / 2
    late_ownership = sum(ownership_score_for(ch) for ch in chronological[-2:]) / 2
    ownership_growth = max(0.0, late_ownership - early_ownership)
    # Even if flat, absolute ownership level matters
    ownership_score = late_ownership * 0.6 + ownership_growth * 0.4

    # ── Signal 4: Domain depth — sustained AI/ML title thread ────────────────
    AI_TITLE_KWS = {
        "ml", "machine learning", "ai ", "nlp", "search", "retrieval",
        "recommendation", "ranking", "data scientist", "applied scientist",
        "research engineer", "applied ml",
    }
    ai_roles = sum(
        1 for ch in chronological
        if any(kw in (ch.get("title", "") or "").lower() for kw in AI_TITLE_KWS)
    )
    domain_depth = min(1.0, ai_roles / max(1, len(chronological)))

    # ── Signal 5: Tenure quality — are they staying long enough to ship? ─────
    # Short stints = learning sprints OR instability. Long stints at good companies = depth.
    durations = [int(ch.get("duration_months", 0) or 0) for ch in chronological]
    if durations:
        avg_tenure = sum(durations) / len(durations)
        # Sweet spot: 18-42 months per role
        if 18 <= avg_tenure <= 42:
            tenure_score = 1.0
        elif 12 <= avg_tenure < 18:
            tenure_score = 0.75
        elif 42 < avg_tenure <= 60:
            tenure_score = 0.85  # long but not stuck
        elif avg_tenure > 60:
            tenure_score = 0.60  # may be stuck
        else:
            tenure_score = 0.40  # too short
    else:
        tenure_score = 0.5

    # ── Composite star score ──────────────────────────────────────────────────
    star_score = (
        title_progression * 0.30 +
        company_arc       * 0.20 +
        ownership_score   * 0.25 +
        domain_depth      * 0.15 +
        tenure_score      * 0.10
    )
    return round(max(0.0, min(1.0, star_score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

# Dimension weights — must sum to 1.0
# experience_quality drops 0.18 → 0.11 to make room for star_predictor 0.07
WEIGHTS = {
    "career_substance":        0.40,  # Primary — titles, headline, product companies
    "skill_credibility":       0.22,  # Corroborated skills (endorsements + duration)
    "experience_quality":      0.11,  # Right YoE band + stability
    "behavioral_availability": 0.15,  # Are they actually reachable?
    "star_predictor":          0.07,  # Career arc — growing scope over time
    "location":                0.05,  # Pune/Noida preferred
}

# Deception penalties
DECEPTION_PENALTIES = {"low": 0.00, "medium": 0.06, "high": 0.18}


def compute_composite_score(candidate: dict, jd_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entry point. Returns a dict with all sub-scores and the final composite.

    Args:
        candidate: A single candidate dict (from candidates.jsonl).
        jd_config:  Optional parsed JD config from jd_parser.parse_jd().
                    If None, the default Redrob JD config is used.

    Flow:
    1. Honeypot check → zero everything
    2. Compute 5 dimension scores (JD-aware)
    3. Deception signals assessment
    4. Hidden gem detection
    5. Apply bonuses and penalties
    6. Produce final composite in [0, 1]
    """
    cfg = jd_config if jd_config is not None else _default_jd()
    cid = candidate.get("candidate_id", "UNKNOWN")

    # ── Step 1: Honeypot gate ──────────────────────────────────────────────
    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        return {
            "candidate_id": cid,
            "composite": 0.0,
            "honeypot": True,
            "honeypot_reason": honeypot_reason,
            "career_substance": 0.0,
            "skill_credibility": 0.0,
            "experience_quality": 0.0,
            "behavioral_availability": 0.0,
            "location": 0.0,
            "star_predictor": 0.0,
            "deception_risk": "n/a",
            "hidden_gem": False,
            "github_bonus": 0.0,
            "edu_bonus": 0.0,
            "auth_bonus": 0.0,
            "cv_penalty": 0.0,
            "company_bonus": 0.0,
            "cert_bonus": 0.0,
            "lang_bonus": 0.0,
            "salary_mult": 1.0,
        }

    # ── Step 2: Dimension scores (JD-aware) ───────────────────────────────
    career_score = score_career_substance(candidate, cfg)
    skill_score = score_skill_credibility(candidate, cfg)
    exp_score = score_experience_quality(candidate, cfg)
    avail_score = score_behavioral_availability(candidate)
    loc_score = score_location(candidate, cfg)
    star_score = score_star_predictor(candidate)

    # Weighted base score
    base = (
        career_score  * WEIGHTS["career_substance"] +
        skill_score   * WEIGHTS["skill_credibility"] +
        exp_score     * WEIGHTS["experience_quality"] +
        avail_score   * WEIGHTS["behavioral_availability"] +
        star_score    * WEIGHTS["star_predictor"] +
        loc_score     * WEIGHTS["location"]
    )

    # ── Step 3: Deception assessment ──────────────────────────────────────
    deception = assess_deception_signals(candidate)
    deception_risk_level = deception.get("deception_risk", "low")
    deception_penalty = DECEPTION_PENALTIES.get(deception_risk_level, 0.0)

    # ── Step 4: Hidden gem detection ──────────────────────────────────────
    is_hidden_gem = detect_hidden_gem(candidate, career_score, skill_score)
    hidden_gem_bonus = 0.05 if is_hidden_gem else 0.0

    # ── Step 5: GitHub bonus ───────────────────────────────────────────────
    github_bonus = compute_github_bonus(candidate)

    # ── Step 5b: Education tier bonus ─────────────────────────────────────
    edu_bonus = compute_education_bonus(candidate)

    # ── Step 5c: Profile authenticity + offer seriousness bonus ───────────
    auth_bonus = compute_authenticity_bonus(candidate)

    # ── Step 5d: CV/speech/robotics specialist penalty ─────────────────────
    cv_penalty = compute_cv_speech_penalty(candidate, cfg)

    # ── Step 5e: Current company quality bonus ──────────────────────────
    company_bonus = compute_current_company_bonus(candidate)

    # ── Step 5f: Certifications bonus ────────────────────────────────────
    cert_bonus = compute_certifications_bonus(candidate)

    # ── Step 5g: Language proficiency bonus ─────────────────────────────
    lang_bonus = compute_language_bonus(candidate)

    # ── Step 5h: Salary alignment multiplier (JD-aware) ──────────────────
    salary_mult = compute_salary_alignment(candidate, cfg)

    # ── Step 6: Final composite ───────────────────────────────────────────
    # GATE: career_substance is the most important signal.
    # If a candidate has no production AI/ML/search work in their career history,
    # they should NOT reach top-100 regardless of how good their YoE or behavioral scores are.
    # This is the core anti-trap mechanism against keyword stuffers and wrong-domain candidates.
    #
    # Gate levels:
    #   career_substance < 0.08 → cap at 0.15 (wrong domain, e.g. Marketing Manager)
    #   career_substance < 0.18 → cap at 0.28 (weak signal, e.g. backend with no AI)
    #   career_substance < 0.30 → cap at 0.45 (some signal but not a core fit)
    #   career_substance >= 0.30 → no cap (let the full formula apply)
    if career_score < 0.08:
        score_cap = 0.15
    elif career_score < 0.18:
        score_cap = 0.28
    elif career_score < 0.30:
        score_cap = 0.45
    else:
        score_cap = 0.999  # no cap

    # Supplementary bonus pool — hard cap at 0.08 to prevent score compression.
    # Without this cap, 61/100 candidates hit 0.999 ceiling and NDCG can't differentiate.
    # All signals still matter and influence ranking, just bounded so top-tier spreads properly.
    supplementary_bonus = min(0.08,
        github_bonus + edu_bonus + auth_bonus + company_bonus + cert_bonus + lang_bonus
    )

    composite = (
        base + hidden_gem_bonus + supplementary_bonus + cv_penalty - deception_penalty
    ) * salary_mult
    composite = round(max(0.0, min(score_cap, composite)), 6)

    return {
        "candidate_id": cid,
        "composite": composite,
        "honeypot": False,
        "honeypot_reason": "",
        "career_substance": round(career_score, 4),
        "skill_credibility": round(skill_score, 4),
        "experience_quality": round(exp_score, 4),
        "behavioral_availability": round(avail_score, 4),
        "star_predictor": round(star_score, 4),
        "location": round(loc_score, 4),
        "deception_risk": deception.get("deception_risk", "low"),
        "deception_flags": deception.get("flags", []),
        "hidden_gem": is_hidden_gem,
        "github_bonus": round(github_bonus, 4),
        "edu_bonus": round(edu_bonus, 4),
        "auth_bonus": round(auth_bonus, 4),
        "cv_penalty": round(cv_penalty, 4),
        "company_bonus": round(company_bonus, 4),
        "cert_bonus": round(cert_bonus, 4),
        "lang_bonus": round(lang_bonus, 4),
        "salary_mult": round(salary_mult, 4),
    }
