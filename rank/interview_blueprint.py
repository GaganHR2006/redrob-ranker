"""
interview_blueprint.py — Generate 3 candidate-specific interview questions.

Each question targets a different dimension:
  1. Probe the strongest AI/ML claim
  2. Probe the biggest JD gap
  3. Culture / stage fit
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AI_ML_SKILL_KEYWORDS = {
    "embedding", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "bm25", "rag", "retrieval", "nlp", "bert",
    "transformer", "llm", "lora", "qlora", "recommendation", "ranking",
    "sentence transformer",
}

_AI_ML_TITLE_KEYWORDS = {
    "ai", "ml", "machine learning", "data scientist", "data science",
    "nlp", "deep learning", "research", "applied scientist",
    "recommendation", "ranking", "search", "retrieval",
}

_JD_MUST_HAVES = [
    {
        "label": "Production embedding-based retrieval",
        "keywords": {
            "sentence-transformers", "sentence transformer", "bge", "e5",
            "embedding", "retrieval", "rag",
        },
    },
    {
        "label": "Vector database operational experience",
        "keywords": {
            "pinecone", "qdrant", "weaviate", "milvus", "faiss",
            "elasticsearch", "vector database", "vector db", "vectordb",
        },
    },
    {
        "label": "Evaluation framework design",
        "keywords": {
            "ndcg", "mrr", "evaluation", "offline evaluation",
            "online evaluation", "a/b test", "ab test", "metrics",
            "precision", "recall", "map",
        },
    },
    {
        "label": "Strong Python",
        "keywords": {"python"},
    },
]

_LARGE_COMPANIES = {
    "google", "amazon", "microsoft", "flipkart", "meta", "facebook",
    "apple", "netflix", "uber", "linkedin", "salesforce", "oracle",
    "ibm", "intel", "adobe", "samsung", "walmart", "target",
    "twitter", "x", "snap", "spotify", "tiktok", "bytedance",
    "tcs", "infosys", "wipro", "cognizant", "accenture", "deloitte",
    "mckinsey", "jpmorgan", "goldman sachs", "morgan stanley",
    "cisco", "qualcomm", "nvidia", "paypal", "stripe", "airbnb",
    "doordash", "instacart", "lyft", "pinterest", "dropbox",
}

_LEAD_TITLE_KEYWORDS = {
    "lead", "principal", "staff", "manager", "director", "head",
    "vp", "chief", "architect", "team lead", "tech lead",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """Return a stripped lowercase string; empty string for None."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce *value* to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _skill_matches_ai_ml(skill_name_lower: str) -> bool:
    """Return True if the skill name matches any AI/ML keyword."""
    for kw in _AI_ML_SKILL_KEYWORDS:
        if kw in skill_name_lower:
            return True
    return False


def _title_matches_ai_ml(title_lower: str) -> bool:
    """Return True if the title contains an AI/ML keyword."""
    for kw in _AI_ML_TITLE_KEYWORDS:
        if kw in title_lower:
            return True
    return False


def _most_recent_relevant_company(career_history: list) -> Optional[Dict]:
    """Return the most recent career entry whose title is AI/ML-related."""
    if not career_history:
        return None
    for entry in career_history:
        title = _safe_str(entry.get("title"))
        if _title_matches_ai_ml(title):
            return entry
    return None


def _candidate_skill_names_lower(candidate: dict) -> List[str]:
    """Return list of lowercase skill names the candidate has."""
    skills = candidate.get("skills") or []
    return [_safe_str(s.get("name")) for s in skills if s and s.get("name")]


def _all_companies_lower(career_history: list) -> List[str]:
    """Return lowercase company names from career history."""
    if not career_history:
        return []
    return [
        _safe_str(e.get("company"))
        for e in career_history
        if e and e.get("company")
    ]


def _all_titles_lower(career_history: list) -> List[str]:
    """Return lowercase titles from career history."""
    if not career_history:
        return []
    return [
        _safe_str(e.get("title"))
        for e in career_history
        if e and e.get("title")
    ]


def _challenge_for_skill(skill_name_lower: str) -> str:
    """Map a skill keyword to a realistic challenge phrase."""
    challenges = {
        "embedding": "scaling embedding generation and serving latency",
        "faiss": "index selection and memory trade-offs in FAISS",
        "pinecone": "namespace partitioning and metadata filtering in Pinecone",
        "weaviate": "schema design and hybrid search tuning in Weaviate",
        "qdrant": "collection sharding and filtering performance in Qdrant",
        "milvus": "index build times and consistency guarantees in Milvus",
        "elasticsearch": "relevance tuning and query performance in Elasticsearch",
        "bm25": "BM25 parameter tuning and its limitations for semantic queries",
        "rag": "retrieval-augmented generation pipeline reliability and chunk strategies",
        "retrieval": "retrieval quality and latency in production",
        "nlp": "NLP pipeline accuracy and edge-case handling",
        "bert": "fine-tuning BERT for domain-specific tasks",
        "transformer": "transformer model optimization for inference",
        "llm": "LLM integration, prompt management, and cost control",
        "lora": "LoRA adapter training and merging strategies",
        "qlora": "QLoRA quantisation trade-offs during fine-tuning",
        "recommendation": "recommendation model iteration and feedback loops",
        "ranking": "ranking model feature engineering and online/offline alignment",
        "sentence transformer": "sentence-transformer model selection and fine-tuning",
    }
    for kw, challenge in challenges.items():
        if kw in skill_name_lower:
            return challenge
    return "scaling and reliability challenges"


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------

def _question_strongest_claim(candidate: dict) -> dict:
    """Q1 — Probe the strongest AI/ML claim."""
    skills = candidate.get("skills") or []
    career_history = candidate.get("career_history") or []

    # Score each AI/ML-relevant skill by endorsements * duration_months
    best_skill = None
    best_score = -1
    for s in skills:
        name_lower = _safe_str(s.get("name"))
        if not _skill_matches_ai_ml(name_lower):
            continue
        endorsements = _safe_int(s.get("endorsements"), 0)
        duration = _safe_int(s.get("duration_months"), 0)
        credibility = endorsements * duration
        if credibility > best_score:
            best_score = credibility
            best_skill = s

    # Fallback: pick the first AI/ML skill if none scored > 0
    if best_skill is None:
        for s in skills:
            if _skill_matches_ai_ml(_safe_str(s.get("name"))):
                best_skill = s
                break

    # If still nothing, produce a generic question
    if best_skill is None:
        return {
            "question": (
                "Which AI or ML technology have you used most extensively "
                "in production, and what was the hardest operational problem "
                "you solved with it?"
            ),
            "why_this_question": (
                "The candidate lists no recognisable AI/ML skills, so this "
                "open-ended question lets them self-select while revealing depth."
            ),
            "strong_answer": (
                "Names a specific system, describes the failure mode or "
                "scaling bottleneck, quantifies the improvement."
            ),
            "weak_answer": (
                "Mentions a technology only at tutorial level, cannot "
                "describe production constraints or measurable outcomes."
            ),
        }

    skill_name = (best_skill.get("name") or "AI/ML").strip()
    skill_lower = _safe_str(skill_name)
    challenge = _challenge_for_skill(skill_lower)

    # Find most recent relevant company
    relevant_entry = _most_recent_relevant_company(career_history)
    if relevant_entry:
        company = (relevant_entry.get("company") or "your company").strip()
        title = (relevant_entry.get("title") or "your role").strip()
    else:
        # Fallback to most recent entry
        if career_history:
            company = (career_history[0].get("company") or "your company").strip()
            title = (career_history[0].get("title") or "your role").strip()
        else:
            company = "your company"
            title = "your role"

    question = (
        "Walk me through how you handled {} "
        "with {} at {} while you were {}."
    ).format(challenge, skill_name, company, title)

    why = (
        "The candidate's strongest credibility signal is {} "
        "(endorsements x duration = {}). This question forces them to "
        "demonstrate hands-on depth rather than surface familiarity."
    ).format(skill_name, max(best_score, 0))

    strong = (
        "A real practitioner would cite specific metrics (latency p99, "
        "recall@k, QPS), describe operational incidents or trade-offs, "
        "and explain why they chose {} over alternatives."
    ).format(skill_name)

    weak = (
        "A surface-level claimant would recite textbook definitions, "
        "struggle to name concrete numbers, or be unable to describe "
        "what went wrong and how they debugged it."
    )

    return {
        "question": question,
        "why_this_question": why,
        "strong_answer": strong,
        "weak_answer": weak,
    }


def _question_biggest_jd_gap(candidate: dict) -> dict:
    """Q2 — Probe the biggest JD must-have gap."""
    skill_names = _candidate_skill_names_lower(candidate)
    # Also pull keywords from career descriptions
    career_history = candidate.get("career_history") or []
    career_text_parts = []
    for entry in career_history:
        desc = _safe_str(entry.get("description"))
        title = _safe_str(entry.get("title"))
        career_text_parts.append(desc)
        career_text_parts.append(title)
    career_blob = " ".join(career_text_parts)

    def _has_evidence(must_have: dict) -> bool:
        """Return True if any keyword appears in skills or career text."""
        for kw in must_have["keywords"]:
            for sn in skill_names:
                if kw in sn:
                    return True
            if kw in career_blob:
                return True
        return False

    # Walk must-haves in priority order; first missing one is the gap
    gap = None
    for mh in _JD_MUST_HAVES:
        if not _has_evidence(mh):
            gap = mh
            break

    # Fallback if everything is covered
    if gap is None:
        return {
            "question": (
                "You have evidence across all our must-have areas. "
                "Tell me about a time you had to integrate multiple "
                "components — embeddings, a vector store, and an "
                "evaluation framework — into one coherent system. "
                "What broke first?"
            ),
            "why_this_question": (
                "No single gap was identified, so we probe integration "
                "ability and real-world failure handling across the stack."
            ),
            "strong_answer": (
                "Describes a concrete integration, names the failure mode "
                "(e.g., stale embeddings, index drift, metric divergence), "
                "and explains the fix with numbers."
            ),
            "weak_answer": (
                "Gives a theoretical architecture diagram but cannot "
                "describe a real failure or how they diagnosed it."
            ),
        }

    gap_label = gap["label"]

    # Produce gap-specific questions
    gap_questions = {
        "Production embedding-based retrieval": {
            "question": (
                "We rely heavily on embedding-based retrieval using models "
                "like sentence-transformers, BGE, and E5. How would you "
                "choose between these models for a new retrieval use case, "
                "and how would you move the chosen model from prototype to "
                "production serving at scale?"
            ),
            "why_this_question": (
                "The candidate shows no evidence of production embedding "
                "retrieval experience. This surfaces whether the gap is "
                "cosmetic (unlisted skill) or substantive."
            ),
            "strong_answer": (
                "Discusses benchmark evaluation (MTEB, domain-specific "
                "test sets), quantisation, ONNX export, batching strategy, "
                "and monitoring embedding drift."
            ),
            "weak_answer": (
                "Only names one model, cannot explain selection criteria, "
                "and skips production concerns like latency or versioning."
            ),
        },
        "Vector database operational experience": {
            "question": (
                "Our stack includes a managed vector database for "
                "real-time retrieval. Walk me through how you would "
                "design the indexing pipeline, choose an index type, "
                "and handle failure modes like stale vectors or "
                "partial index rebuilds."
            ),
            "why_this_question": (
                "No vector database operational experience is evident. "
                "This probes whether the candidate can operate — not just "
                "query — a vector store."
            ),
            "strong_answer": (
                "Names specific index types (HNSW, IVF-PQ), discusses "
                "trade-offs (recall vs latency vs memory), and describes "
                "a real incident or migration."
            ),
            "weak_answer": (
                "Treats the vector DB as a black box, cannot discuss "
                "index internals or operational failure scenarios."
            ),
        },
        "Evaluation framework design": {
            "question": (
                "How would you design an offline evaluation pipeline "
                "for a ranking system? Walk me through what metrics "
                "you would track and how you would validate they "
                "correlate with online performance."
            ),
            "why_this_question": (
                "The candidate has no evidence of evaluation framework "
                "experience (NDCG, MRR, offline-to-online). This is "
                "critical for iterating on ranking quality."
            ),
            "strong_answer": (
                "Defines metric hierarchy (NDCG@k, MRR, coverage), "
                "describes how to build golden sets, explains "
                "offline-online correlation analysis, and mentions "
                "guard-rail metrics."
            ),
            "weak_answer": (
                "Can name metrics but not explain how to collect "
                "labels, or has never compared offline lifts to "
                "online A/B results."
            ),
        },
        "Strong Python": {
            "question": (
                "Our codebase is Python-heavy with strict performance "
                "requirements. Describe a time you profiled and "
                "optimised a Python service in production. What tools "
                "did you use, and what was the outcome?"
            ),
            "why_this_question": (
                "Python is not prominently listed in the candidate's "
                "skill set. We need confidence they can own "
                "production Python code."
            ),
            "strong_answer": (
                "Names profiling tools (cProfile, py-spy, memory_profiler), "
                "describes a concrete bottleneck, quantifies the speed-up "
                "or memory reduction."
            ),
            "weak_answer": (
                "Claims Python expertise but cannot describe profiling "
                "methodology or production-grade coding practices."
            ),
        },
    }

    return gap_questions.get(gap_label, {
        "question": (
            "Tell me about your experience with {}. "
            "What have you built, and what did you learn?"
        ).format(gap_label),
        "why_this_question": (
            "This probes the gap in: {}.".format(gap_label)
        ),
        "strong_answer": "Describes hands-on work with concrete outcomes.",
        "weak_answer": "Gives only theoretical or tutorial-level answers.",
    })


def _question_culture_fit(candidate: dict) -> dict:
    """Q3 — Culture / stage fit based on career signals."""
    career_history = candidate.get("career_history") or []
    profile = candidate.get("profile") or {}

    companies = _all_companies_lower(career_history)
    titles = _all_titles_lower(career_history)

    # --- Signal: all large companies ---
    all_large = False
    if companies:
        all_large = all(
            any(lc in c for lc in _LARGE_COMPANIES)
            for c in companies
        )

    # --- Signal: multiple short tenures (< 14 months) ---
    short_tenure_count = 0
    for entry in career_history:
        dur = _safe_int(entry.get("duration_months"), 0)
        if 0 < dur < 14:
            short_tenure_count += 1
    many_short = short_tenure_count >= 2

    # --- Signal: only IC, never a lead ---
    has_lead = any(
        any(lk in t for lk in _LEAD_TITLE_KEYWORDS)
        for t in titles
    )

    # Pick the first company name and title for personalisation
    current_company = (profile.get("current_company") or "").strip()
    current_title = (profile.get("current_title") or "").strip()
    if not current_company and career_history:
        current_company = (career_history[0].get("company") or "").strip()
    if not current_title and career_history:
        current_title = (career_history[0].get("title") or "").strip()

    # Build readable company list for question text (max 3)
    named_companies = [
        (e.get("company") or "").strip()
        for e in career_history
        if e and e.get("company")
    ][:3]
    company_str = ", ".join(named_companies) if named_companies else "your previous companies"

    # --- Decision tree ---
    if all_large and companies:
        return {
            "question": (
                "Your career so far has been at established organisations "
                "like {}. In a startup or small team, there is no "
                "existing ML platform, no feature store, and no on-call "
                "rotation — you would build it all. How would you "
                "prioritise what to build first, and what would you "
                "deliberately skip?"
            ).format(company_str),
            "why_this_question": (
                "All listed companies ({}) are large orgs. We need to "
                "know if the candidate can operate without mature "
                "infrastructure and support teams."
            ).format(company_str),
            "strong_answer": (
                "Demonstrates scrappy prioritisation: start with the "
                "simplest end-to-end pipeline, defer tooling that does "
                "not block iteration, and names specific shortcuts "
                "(e.g., use managed services, skip custom monitoring "
                "early)."
            ),
            "weak_answer": (
                "Proposes a large-company playbook (build a feature "
                "store first, set up a model registry) without "
                "acknowledging resource constraints."
            ),
        }

    if many_short:
        # Collect short-tenure entries for specificity
        short_entries = [
            e for e in career_history
            if 0 < _safe_int(e.get("duration_months"), 0) < 14
        ]
        short_examples = [
            "{} at {} ({} months)".format(
                (e.get("title") or "role").strip(),
                (e.get("company") or "company").strip(),
                _safe_int(e.get("duration_months"), 0),
            )
            for e in short_entries[:2]
        ]
        short_str = "; ".join(short_examples) if short_examples else "several short stints"

        return {
            "question": (
                "I notice some shorter tenures in your history — {}. "
                "Tell me about a time you stayed with a genuinely hard "
                "technical problem for six months or longer. What kept "
                "you engaged, and what was the arc from confusion to "
                "resolution?"
            ).format(short_str),
            "why_this_question": (
                "Multiple roles under 14 months ({}) may signal "
                "difficulty with sustained engagement. This question "
                "looks for evidence of persistence."
            ).format(short_str),
            "strong_answer": (
                "Describes a specific multi-month arc: the initial "
                "confusion, iterative experiments, dead-ends, and the "
                "eventual breakthrough with measurable impact."
            ),
            "weak_answer": (
                "Cannot name a single extended deep-dive, or only "
                "describes tasks that were completed in days."
            ),
        }

    if not has_lead and titles:
        return {
            "question": (
                "Your roles — such as {} — have been individual "
                "contributor positions. If you needed to drive a "
                "technical direction that required other engineers to "
                "change their approach, how would you bring them along "
                "without formal authority?"
            ).format(
                ", ".join(
                    (e.get("title") or "").strip()
                    for e in career_history[:2]
                    if e and e.get("title")
                ) or current_title or "your current role"
            ),
            "why_this_question": (
                "No lead or management titles detected. The role "
                "requires influencing others; this checks for informal "
                "leadership skills."
            ),
            "strong_answer": (
                "Gives a concrete example: wrote an RFC or design doc, "
                "held office hours, paired with sceptics, and tracked "
                "adoption metrics."
            ),
            "weak_answer": (
                "Says 'I would just present the idea' without "
                "describing how they handled resistance or measured "
                "buy-in."
            ),
        }

    # --- Default: first 30 days ---
    company_context = current_company if current_company else "our company"
    return {
        "question": (
            "Imagine it is your first day here. You inherit a ranking "
            "service that is live but under-documented. Walk me through "
            "your first 30 days: what do you learn first, what do you "
            "ship first, and how do you build trust with the team?"
        ),
        "why_this_question": (
            "No strong career-pattern signal was detected, so we use a "
            "general onboarding question to assess initiative, learning "
            "speed, and collaboration style."
        ),
        "strong_answer": (
            "Describes a structured ramp: read existing metrics and "
            "on-call logs, shadow production, pick a small win (fix a "
            "bug, add a dashboard), and schedule 1-on-1s with "
            "stakeholders."
        ),
        "weak_answer": (
            "Jumps straight to 'I would refactor the whole system' "
            "without understanding the current state or building "
            "relationships first."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_interview_blueprint(candidate: dict, scores: dict) -> list:
    """
    Generate 3 candidate-specific interview questions.

    Parameters
    ----------
    candidate : dict
        Candidate data following the standard schema (profile,
        career_history, skills, redrob_signals, education).
    scores : dict
        Pre-computed ranking scores for the candidate (used for
        potential future enrichment; currently informational).

    Returns
    -------
    list[dict]
        A list of exactly 3 question dicts, each containing:
        ``question``, ``why_this_question``, ``strong_answer``,
        ``weak_answer``.
    """
    if candidate is None:
        candidate = {}
    if scores is None:
        scores = {}

    questions = [
        _question_strongest_claim(candidate),
        _question_biggest_jd_gap(candidate),
        _question_culture_fit(candidate),
    ]

    return questions
