"""
hm_translator.py – Generate three substantively different hiring-manager
briefs (CTO, VP Engineering, Founder) from a scored candidate dict.

Each brief is ≤ 2 sentences and must reference at least one concrete value
from the candidate's profile.

Python 3.9 compatible – no f-string backslashes, no dict|None, no match/case.
"""

from __future__ import annotations

import math
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Big-tech / large-company names used by the Founder brief to gauge
# whether a candidate has only worked at large orgs.
# ---------------------------------------------------------------------------
_BIG_TECH = frozenset([
    "google", "alphabet", "meta", "facebook", "amazon", "apple",
    "microsoft", "netflix", "uber", "lyft", "airbnb", "salesforce",
    "oracle", "ibm", "intel", "cisco", "adobe", "twitter", "x corp",
    "snap", "bytedance", "tiktok", "stripe", "palantir", "databricks",
    "snowflake", "samsung", "tencent", "alibaba", "baidu",
])

# ---------------------------------------------------------------------------
# AI / ML sub-domain keywords used by the Founder brief to measure breadth.
# ---------------------------------------------------------------------------
_AI_ML_DOMAINS = [
    "embedding", "nlp", "natural language", "ranking", "recommendation",
    "computer vision", "cv", "image recognition", "object detection",
    "reinforcement learning", "generative", "llm", "large language model",
    "speech", "asr", "tts", "forecasting", "time series", "anomaly detection",
    "graph neural", "gnn", "transformer", "diffusion", "gan",
    "ml ops", "mlops", "feature engineering", "model serving",
    "search relevance", "information retrieval",
]

# ---------------------------------------------------------------------------
# Technical ownership keywords scanned in career_history descriptions.
# ---------------------------------------------------------------------------
_OWNERSHIP_VERBS = ["built", "owned", "designed", "architected", "led", "created"]


# ===================================================================
# Internal helpers
# ===================================================================

def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Nested dict lookup that never raises on None / missing keys."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    return current


def _fmt_months(months: Optional[int]) -> str:
    """Convert *months* to a human-friendly string like '3yr 2mo'."""
    if months is None or months < 0:
        return "unknown tenure"
    years = months // 12
    remaining = months % 12
    parts = []
    if years:
        parts.append("{}yr".format(years))
    if remaining or not parts:
        parts.append("{}mo".format(remaining))
    return " ".join(parts)


def _days_since(date_str: Optional[str]) -> Optional[int]:
    """Return number of days between *date_str* (ISO-ish) and today."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(date_str.strip(), fmt)
            delta = datetime.now() - parsed.replace(tzinfo=None)
            return max(int(delta.days), 0)
        except (ValueError, TypeError):
            continue
    return None


def _career_entries(candidate: dict) -> List[dict]:
    """Return career_history list, gracefully handling None."""
    hist = _safe_get(candidate, "career_history")
    if isinstance(hist, list):
        return hist
    return []


def _skills_list(candidate: dict) -> List[dict]:
    """Return skills list, gracefully handling None."""
    sk = _safe_get(candidate, "skills")
    if isinstance(sk, list):
        return sk
    return []


def _profile_val(candidate: dict, key: str) -> Any:
    """Shortcut into candidate['profile'][key]."""
    return _safe_get(candidate, "profile", key)


def _signal_val(candidate: dict, key: str) -> Any:
    """Shortcut into candidate['redrob_signals'][key]."""
    return _safe_get(candidate, "redrob_signals", key)


# -------------------------------------------------------------------
# Ownership-snippet extraction for CTO brief
# -------------------------------------------------------------------

def _ownership_snippets(candidate: dict) -> List[Tuple[str, str, str]]:
    """
    Scan career_history descriptions for ownership verbs.
    Returns list of (verb_found, company, title).
    """
    results = []  # type: List[Tuple[str, str, str]]
    for entry in _career_entries(candidate):
        desc = (entry.get("description") or "").lower()
        company = entry.get("company") or "unknown company"
        title = entry.get("title") or "unknown role"
        for verb in _OWNERSHIP_VERBS:
            if verb in desc:
                results.append((verb, company, title))
                break  # one verb per entry is enough
    return results


# -------------------------------------------------------------------
# Highest-endorsed skill (AI/ML bias)
# -------------------------------------------------------------------

def _top_endorsed_skill(candidate: dict) -> Optional[dict]:
    """Return the skill dict with the highest endorsements count."""
    best = None  # type: Optional[dict]
    best_endo = -1
    for sk in _skills_list(candidate):
        endo = sk.get("endorsements")
        if endo is None:
            continue
        try:
            endo_int = int(endo)
        except (ValueError, TypeError):
            continue
        if endo_int > best_endo:
            best_endo = endo_int
            best = sk
    return best


# -------------------------------------------------------------------
# Company count, average tenure, total experience
# -------------------------------------------------------------------

def _career_stats(candidate: dict) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Returns (num_companies, avg_tenure_months, total_months).
    avg_tenure_months is None when we have no duration data.
    """
    entries = _career_entries(candidate)
    num = len(entries)
    durations = []
    for e in entries:
        dur = e.get("duration_months")
        if dur is not None:
            try:
                durations.append(int(dur))
            except (ValueError, TypeError):
                pass
    if durations:
        avg = int(round(sum(durations) / len(durations)))
        total = sum(durations)
        return (num, avg, total)
    return (num, None, None)


# -------------------------------------------------------------------
# Small-company check for Founder brief
# -------------------------------------------------------------------

def _has_small_company_exp(candidate: dict) -> Tuple[bool, List[str]]:
    """
    Returns (has_small, list_of_non_big_tech_company_names).
    A company is "small" if its lowercased name is NOT in _BIG_TECH.
    """
    non_big = []  # type: List[str]
    for entry in _career_entries(candidate):
        company = entry.get("company") or ""
        if company.strip().lower() not in _BIG_TECH and company.strip():
            if company not in non_big:
                non_big.append(company)
    return (len(non_big) > 0, non_big)


# -------------------------------------------------------------------
# AI/ML sub-domain breadth
# -------------------------------------------------------------------

def _ai_ml_breadth(candidate: dict) -> Tuple[int, List[str]]:
    """
    Count distinct AI/ML sub-domains that appear in the candidate's
    skill names.  Returns (count, matched_domain_keywords).
    """
    skill_names_lower = []
    for sk in _skills_list(candidate):
        name = sk.get("name")
        if name:
            skill_names_lower.append(name.lower())
    joined = " ".join(skill_names_lower)
    matched = []  # type: List[str]
    for domain in _AI_ML_DOMAINS:
        if domain in joined:
            matched.append(domain)
    return (len(matched), matched)


# -------------------------------------------------------------------
# Career-trajectory helper for Founder brief
# -------------------------------------------------------------------

def _title_progression(candidate: dict) -> Optional[str]:
    """
    Try to describe career acceleration from earliest to latest title.
    Returns a short string like 'IC to Staff' or None.
    """
    entries = _career_entries(candidate)
    if len(entries) < 2:
        return None
    # career_history is assumed most-recent-first; reverse for chronological
    sorted_entries = list(entries)
    # Attempt to sort by start_date if available, else use list order
    def _parse_start(e: dict) -> str:
        return e.get("start_date") or ""
    try:
        sorted_entries = sorted(sorted_entries, key=_parse_start)
    except TypeError:
        pass
    first_title = (sorted_entries[0].get("title") or "").strip()
    last_title = (sorted_entries[-1].get("title") or "").strip()
    if first_title and last_title and first_title.lower() != last_title.lower():
        return "{} to {}".format(first_title, last_title)
    return None


# ===================================================================
# Brief generators
# ===================================================================

def _build_cto_brief(candidate: dict, scores: dict) -> str:
    """
    CTO brief — technical depth:
    - Production ownership from career_history descriptions
    - Strongest endorsed skill with duration
    - GitHub activity score
    - Assessment scores
    """
    sentences = []

    # --- Sentence 1: ownership + experience summary -----------------
    yoe = _profile_val(candidate, "years_of_experience")
    title = _profile_val(candidate, "current_title") or "engineer"
    company = _profile_val(candidate, "current_company")
    snippets = _ownership_snippets(candidate)

    if yoe is not None and snippets:
        verb, own_company, own_title = snippets[0]
        # Find tenure at that company
        tenure_str = ""
        for entry in _career_entries(candidate):
            if (entry.get("company") or "") == own_company:
                dur = entry.get("duration_months")
                if dur is not None:
                    tenure_str = " ({} tenure)".format(_fmt_months(int(dur)))
                break
        s1 = "{}yr {} who {} production systems at {}{}".format(
            yoe, title, verb, own_company, tenure_str
        )
        sentences.append(s1 + ".")
    elif yoe is not None:
        loc = _profile_val(candidate, "location") or ""
        loc_part = ", based in {}".format(loc) if loc else ""
        s1 = "{}yr {}{} at {}".format(
            yoe, title, loc_part,
            company if company else "current employer"
        )
        sentences.append(s1 + ".")
    else:
        sentences.append("{} currently at {}.".format(
            title, company or "unknown company"
        ))

    # --- Sentence 2: top skill + github + assessments ---------------
    top_skill = _top_endorsed_skill(candidate)
    github = _signal_val(candidate, "github_activity_score")
    assessments = _signal_val(candidate, "skill_assessment_scores") or {}

    parts = []  # type: List[str]
    if top_skill:
        sk_name = top_skill.get("name", "top skill")
        sk_endo = top_skill.get("endorsements", 0)
        sk_dur = top_skill.get("duration_months")
        if sk_dur is not None:
            parts.append("{} has {} endorsements over {} months".format(
                sk_name, sk_endo, sk_dur
            ))
        else:
            parts.append("{} has {} endorsements".format(sk_name, sk_endo))

    if github is not None:
        parts.append("GitHub activity score {}".format(github))

    if assessments:
        # pick the best assessment
        best_assess_name = None  # type: Optional[str]
        best_assess_val = -1.0
        for aname, aval in assessments.items():
            try:
                fval = float(aval)
            except (ValueError, TypeError):
                continue
            if fval > best_assess_val:
                best_assess_val = fval
                best_assess_name = aname
        if best_assess_name is not None:
            parts.append("{} assessment {:.0f}%".format(
                best_assess_name, best_assess_val * 100
                if best_assess_val <= 1.0 else best_assess_val
            ))

    if parts:
        qualifier = " — genuine production depth" if top_skill else ""
        sentences.append(", ".join(parts) + qualifier + ".")
    else:
        sentences.append("No endorsed skills or assessment data available.")

    return " ".join(sentences[:2])


def _build_vp_brief(candidate: dict, scores: dict) -> str:
    """
    VP Engineering brief — execution and delivery:
    - Company count, average tenure
    - Recruiter response rate, interview completion, notice period
    - Days since last active
    """
    sentences = []

    # --- Sentence 1: career stability --------------------------------
    num_companies, avg_tenure, total_months = _career_stats(candidate)
    yoe = _profile_val(candidate, "years_of_experience")

    if num_companies > 0 and avg_tenure is not None:
        yoe_str = "in {} years".format(yoe) if yoe else ""
        stability = "stable" if avg_tenure >= 18 else "short stints"
        s1 = "{} companies {}, avg {}mo tenure — {}".format(
            num_companies, yoe_str, avg_tenure, stability
        )
        if avg_tenure >= 24:
            s1 += ", low flight risk"
        sentences.append(s1 + ".")
    elif num_companies > 0:
        sentences.append("{} companies on record.".format(num_companies))
    else:
        sentences.append("No career history entries available.")

    # --- Sentence 2: availability signals ----------------------------
    response_rate = _signal_val(candidate, "recruiter_response_rate")
    completion = _signal_val(candidate, "interview_completion_rate")
    notice = _signal_val(candidate, "notice_period_days")
    last_active = _signal_val(candidate, "last_active_date")

    avail_parts = []  # type: List[str]
    if response_rate is not None:
        pct = response_rate
        if isinstance(pct, float) and pct <= 1.0:
            pct = int(round(pct * 100))
        avail_parts.append("response rate {}%".format(pct))

    if notice is not None:
        avail_parts.append("{}-day notice".format(notice))

    if completion is not None:
        comp_pct = completion
        if isinstance(comp_pct, float) and comp_pct <= 1.0:
            comp_pct = int(round(comp_pct * 100))
        avail_parts.append("interview completion {}%".format(comp_pct))

    days_ago = _days_since(last_active)
    if days_ago is not None:
        avail_parts.append("active {} days ago".format(days_ago))
        if days_ago <= 14:
            avail_parts.append("should respond to outreach within the week")
        elif days_ago <= 60:
            avail_parts.append("moderately active")
        else:
            avail_parts.append("may be slow to respond")

    if avail_parts:
        sentences.append(", ".join(avail_parts) + ".")
    else:
        sentences.append("No availability signals on file.")

    return " ".join(sentences[:2])


def _build_founder_brief(candidate: dict, scores: dict) -> str:
    """
    Founder brief — speed, scrappiness, startup fit:
    - Notice period (how fast can they start)
    - Small-company experience
    - open_to_work_flag
    - star_predictor score (career acceleration)
    - AI/ML skill breadth
    """
    sentences = []
    scores = scores if isinstance(scores, dict) else {}

    # --- Sentence 1: acceleration + openness -------------------------
    star_score = scores.get("star_predictor")
    otw = _signal_val(candidate, "open_to_work_flag")
    notice = _signal_val(candidate, "notice_period_days")
    progression = _title_progression(candidate)
    yoe = _profile_val(candidate, "years_of_experience")

    s1_parts = []  # type: List[str]
    if star_score is not None:
        try:
            star_f = float(star_score)
            label = "accelerating" if star_f >= 0.6 else (
                "steady" if star_f >= 0.4 else "plateauing"
            )
            s1_parts.append("Career {} (star score {:.2f})".format(label, star_f))
        except (ValueError, TypeError):
            pass

    if progression and yoe is not None:
        s1_parts.append("{} in {} years".format(progression, yoe))
    elif progression:
        s1_parts.append(progression)

    if notice is not None:
        s1_parts.append("{}-day notice".format(notice))

    if otw is True:
        s1_parts.append("open to work")
    elif otw is False:
        s1_parts.append("not flagged as open to work")

    if s1_parts:
        sentences.append(" — ".join(s1_parts) + ".")
    else:
        title = _profile_val(candidate, "current_title") or "Candidate"
        sentences.append("{} with limited acceleration data on file.".format(title))

    # --- Sentence 2: startup fit + breadth ---------------------------
    has_small, small_names = _has_small_company_exp(candidate)
    breadth_count, breadth_domains = _ai_ml_breadth(candidate)

    s2_parts = []  # type: List[str]
    if has_small:
        example = small_names[0]
        extra = " and {} others".format(len(small_names) - 1) if len(small_names) > 1 else ""
        s2_parts.append("Has worked outside big-tech ({}{})".format(example, extra))
    else:
        num_companies, _, _ = _career_stats(candidate)
        if num_companies > 0:
            s2_parts.append(
                "Has never been outside big-tech across {} roles, "
                "so probe comfort with ambiguity".format(num_companies)
            )

    if breadth_count >= 3:
        sample = ", ".join(breadth_domains[:3])
        s2_parts.append(
            "{} AI/ML sub-domains (e.g. {})".format(breadth_count, sample)
        )
    elif breadth_count > 0:
        sample = ", ".join(breadth_domains)
        s2_parts.append(
            "narrow AI/ML breadth ({})".format(sample)
        )

    if s2_parts:
        sentences.append("; ".join(s2_parts) + ".")
    else:
        sentences.append("No startup-fit signals available.")

    return " ".join(sentences[:2])


# ===================================================================
# Public API
# ===================================================================

def generate_hm_briefs(candidate: dict, scores: dict) -> dict:
    """
    Generate three substantively different hiring-manager briefs.

    Parameters
    ----------
    candidate : dict
        Full candidate record following the standard schema.
    scores : dict
        Pre-computed scores dict (must include 'star_predictor' key
        for the founder brief).

    Returns
    -------
    dict
        {
            "cto": "...",            # technical depth
            "vp_engineering": "...", # execution & delivery
            "founder": "...",        # speed & startup fit
        }
    """
    if not isinstance(candidate, dict):
        candidate = {}
    if not isinstance(scores, dict):
        scores = {}

    return {
        "cto": _build_cto_brief(candidate, scores),
        "vp_engineering": _build_vp_brief(candidate, scores),
        "founder": _build_founder_brief(candidate, scores),
    }
