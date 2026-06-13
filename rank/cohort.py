"""
cohort.py — Cohort Comparator: post-processing analysis of the top-100 ranked candidates.

This is NOT a bar chart. It answers specific questions:
  - Which JD requirements does the shortlist actually cover?
  - Where is every top-10 candidate weak on the same dimension? (shared gaps)
  - What are the real tradeoffs between rank N and rank N+1?
  - Is the shortlist suspiciously homogeneous? (anti-bias check)
  - Which JD requirements no candidate in the top-10 fully satisfies? (supply gaps)

Output: a structured dict saved to outputs/cohort_report.json
"""

from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# JD REQUIREMENTS — the must-haves and strong-positives to audit against
# ─────────────────────────────────────────────────────────────────────────────

JD_REQUIREMENTS = [
    {
        "id": "vector_search",
        "label": "Production vector search (FAISS/Pinecone/Qdrant/Weaviate)",
        "skill_keywords": ["faiss", "pinecone", "qdrant", "weaviate", "milvus", "vector search", "dense retrieval"],
        "title_keywords": ["search engineer", "retrieval", "recommendation"],
        "weight": "must_have",
    },
    {
        "id": "python_depth",
        "label": "Strong Python proficiency (ML engineering level)",
        "skill_keywords": ["python"],
        "title_keywords": [],
        "weight": "must_have",
    },
    {
        "id": "ranking_systems",
        "label": "Ranking / recommendation system ownership",
        "skill_keywords": ["ranking", "recommendation", "learning to rank", "ltr", "bm25", "ndcg", "mrr"],
        "title_keywords": ["ranking", "recommendation", "search"],
        "weight": "must_have",
    },
    {
        "id": "nlp_llm",
        "label": "NLP / LLM application engineering",
        "skill_keywords": ["nlp", "bert", "transformer", "llm", "rag", "fine-tun", "lora", "qlora",
                           "sentence transformer", "embedding"],
        "title_keywords": ["nlp", "language", "ai engineer"],
        "weight": "strong_positive",
    },
    {
        "id": "eval_framework",
        "label": "Offline/online evaluation pipeline experience",
        "skill_keywords": ["a/b test", "experiment", "ndcg", "mrr", "map@", "evaluation", "mlflow",
                           "wandb", "weights & biases"],
        "title_keywords": [],
        "weight": "strong_positive",
    },
    {
        "id": "product_company",
        "label": "Product company experience (non-consulting)",
        "skill_keywords": [],
        "title_keywords": [],
        "weight": "must_have",
        "special": "product_company_check",
    },
    {
        "id": "startup_ready",
        "label": "Early-stage / founding team experience",
        "skill_keywords": ["founding", "early stage", "0-to-1", "zero to one", "seed", "series a"],
        "title_keywords": ["founding", "first engineer", "head of"],
        "weight": "strong_positive",
    },
    {
        "id": "availability",
        "label": "Available within 60 days",
        "skill_keywords": [],
        "title_keywords": [],
        "weight": "strong_positive",
        "special": "availability_check",
    },
]

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "mphasis", "hexaware",
}

PREFERRED_LOCATIONS = {"pune", "noida"}
TIER1_LOCATIONS = {"delhi", "gurgaon", "hyderabad", "bengaluru", "bangalore", "mumbai"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: does a candidate satisfy a requirement?
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_satisfies(candidate: dict, req: dict, scores: dict) -> Tuple[bool, str]:
    """
    Returns (satisfied: bool, evidence: str).
    """
    special = req.get("special")

    # Special checks
    if special == "product_company_check":
        career = candidate.get("career_history", [])
        consulting_count = sum(
            1 for ch in career
            if any(cf in (ch.get("company", "") or "").lower() for cf in CONSULTING_FIRMS)
        )
        all_consulting = consulting_count == len(career) and len(career) > 0
        if all_consulting:
            companies = [ch.get("company", "") for ch in career[:2]]
            return False, f"All roles at consulting firms: {', '.join(str(c) for c in companies)}"
        elif consulting_count > 0:
            return True, f"Mixed: {consulting_count}/{len(career)} roles at consulting firms"
        return True, "All roles at product companies"

    if special == "availability_check":
        rs = candidate.get("redrob_signals", {})
        notice = int(rs.get("notice_period_days", 90) or 90)
        if notice <= 60:
            return True, f"{notice}-day notice period"
        return False, f"{notice}-day notice period — exceeds 60-day threshold"

    # Skill-based check
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    skill_names = [s.get("name", "").lower() for s in skills]
    career_text = " ".join(
        (ch.get("title", "") + " " + ch.get("description", "")).lower()
        for ch in career
    )
    current_title = (profile.get("current_title", "") or "").lower()
    headline = (profile.get("headline", "") or "").lower()
    all_text = career_text + " " + current_title + " " + headline

    # Check skill keywords
    skill_matches = [kw for kw in req.get("skill_keywords", []) if any(kw in sn for sn in skill_names)]
    text_matches = [kw for kw in req.get("skill_keywords", []) if kw in all_text]
    title_matches = [kw for kw in req.get("title_keywords", []) if kw in all_text]

    if skill_matches:
        evidence = f"Skills: {', '.join(skill_matches[:2])}"
        if title_matches:
            evidence += f"; Title/career: {', '.join(title_matches[:1])}"
        return True, evidence
    elif text_matches or title_matches:
        matches = (text_matches + title_matches)[:2]
        return True, f"Career/title mentions: {', '.join(matches)} (not in skills list)"
    else:
        return False, f"No evidence of {', '.join((req.get('skill_keywords', []) + req.get('title_keywords', []))[:3])}"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: JD Coverage Report
# ─────────────────────────────────────────────────────────────────────────────

def jd_coverage_report(top_candidates: List[Tuple[dict, dict, int]]) -> dict:
    """
    For each JD requirement, what fraction of the top-10 and top-100 satisfy it?
    Flags any requirement that no top-10 candidate meets.
    """
    top10 = [t for t in top_candidates if t[2] <= 10]
    results = []

    for req in JD_REQUIREMENTS:
        top10_satisfying = []
        top100_count = 0

        for candidate, scores, rank in top_candidates:
            satisfied, evidence = _candidate_satisfies(candidate, req, scores)
            if rank <= 10:
                top10_satisfying.append({
                    "rank": rank,
                    "candidate_id": candidate.get("candidate_id"),
                    "satisfied": satisfied,
                    "evidence": evidence,
                })
            if satisfied:
                top100_count += 1

        top10_count = sum(1 for x in top10_satisfying if x["satisfied"])
        coverage_pct_10 = (top10_count / max(1, len(top10))) * 100
        coverage_pct_100 = (top100_count / max(1, len(top_candidates))) * 100

        results.append({
            "requirement_id": req["id"],
            "label": req["label"],
            "weight": req["weight"],
            "top10_coverage_pct": round(coverage_pct_10, 1),
            "top100_coverage_pct": round(coverage_pct_100, 1),
            "top10_detail": top10_satisfying,
            "supply_gap": coverage_pct_10 < 50,
            "gap_note": (
                f"Only {top10_count}/{len(top10)} top-10 candidates satisfy this requirement."
                " Consider whether the shortlist is weaker than it looks on this dimension."
            ) if coverage_pct_10 < 50 else None,
        })

    return {"jd_coverage": results}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Shared Gaps — what does the entire top-10 fail at?
# ─────────────────────────────────────────────────────────────────────────────

def shared_gaps(top_candidates: List[Tuple[dict, dict, int]]) -> dict:
    """
    Find dimensions where ALL top-10 candidates are below a threshold.
    These are supply-side constraints, not individual weaknesses.
    """
    top10 = [(c, s, r) for c, s, r in top_candidates if r <= 10]
    if not top10:
        return {"shared_gaps": []}

    gaps = []

    # Check each JD requirement
    for req in JD_REQUIREMENTS:
        missing_candidates = []
        for candidate, scores, rank in top10:
            satisfied, evidence = _candidate_satisfies(candidate, req, scores)
            if not satisfied:
                p = candidate.get("profile", {})
                missing_candidates.append({
                    "rank": rank,
                    "candidate_id": candidate.get("candidate_id"),
                    "title": p.get("current_title"),
                    "reason": evidence,
                })

        if len(missing_candidates) >= 7:  # 7/10 or more fail = shared gap
            gaps.append({
                "requirement": req["label"],
                "severity": "critical" if req["weight"] == "must_have" else "notable",
                "candidates_missing": len(missing_candidates),
                "note": (
                    f"{len(missing_candidates)}/10 top candidates don't clearly satisfy this."
                    " This is a pool-wide supply gap, not individual weakness."
                ),
                "examples": missing_candidates[:3],
            })

    # Check availability gap
    slow_notice = [
        (c, s, r) for c, s, r in top10
        if int((c.get("redrob_signals", {}).get("notice_period_days") or 90)) > 60
    ]
    if len(slow_notice) >= 6:
        gaps.append({
            "requirement": "Available within 60 days",
            "severity": "notable",
            "candidates_missing": len(slow_notice),
            "note": (
                f"{len(slow_notice)}/10 top candidates have >60-day notice periods."
                " The best candidates in this pool aren't immediately available."
            ),
            "examples": [
                {
                    "rank": r,
                    "candidate_id": c.get("candidate_id"),
                    "notice_days": int((c.get("redrob_signals", {}).get("notice_period_days") or 90)),
                }
                for c, s, r in slow_notice[:3]
            ],
        })

    return {"shared_gaps": gaps}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: Pairwise Tradeoffs
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_tradeoffs(top_candidates: List[Tuple[dict, dict, int]]) -> dict:
    """
    For adjacent rank pairs in the top-20, explain what actually differs.
    "Rank 3 vs Rank 5: Rank 3 is stronger technically but 150-day notice.
     Rank 5 is 30 days away and 85% of the technical score."
    """
    top20 = sorted(
        [(c, s, r) for c, s, r in top_candidates if r <= 20],
        key=lambda x: x[2]
    )

    comparisons = []
    # Compare pairs: (1,2), (2,3), (4,5), (9,10) — key decision boundaries
    pairs_to_compare = [(0, 1), (1, 2), (4, 5), (9, 10) if len(top20) > 10 else (8, 9)]

    for i, j in pairs_to_compare:
        if j >= len(top20):
            continue
        c1, s1, r1 = top20[i]
        c2, s2, r2 = top20[j]
        p1 = c1.get("profile", {})
        p2 = c2.get("profile", {})
        rs1 = c1.get("redrob_signals", {})
        rs2 = c2.get("redrob_signals", {})

        title1 = p1.get("current_title", "?")
        title2 = p2.get("current_title", "?")
        notice1 = int(rs1.get("notice_period_days", 90) or 90)
        notice2 = int(rs2.get("notice_period_days", 90) or 90)
        rrr1 = float(rs1.get("recruiter_response_rate", 0.3) or 0.3)
        rrr2 = float(rs2.get("recruiter_response_rate", 0.3) or 0.3)

        # Identify key differences
        diffs = []

        # Technical (career + skill)
        tech1 = s1.get("career_substance", 0) + s1.get("skill_credibility", 0) * 0.5
        tech2 = s2.get("career_substance", 0) + s2.get("skill_credibility", 0) * 0.5
        if abs(tech1 - tech2) > 0.05:
            stronger = r1 if tech1 > tech2 else r2
            diffs.append(
                f"Technical depth: Rank {r1} {'leads' if tech1 > tech2 else 'trails'}"
                f" ({tech1:.2f} vs {tech2:.2f} combined career+skill score)."
            )

        # Availability
        if abs(notice1 - notice2) >= 30:
            faster = r1 if notice1 < notice2 else r2
            diffs.append(
                f"Availability: Rank {faster} can start ~{abs(notice1-notice2)} days sooner"
                f" ({notice1}d vs {notice2}d notice)."
            )

        # Response rate
        if abs(rrr1 - rrr2) > 0.2:
            better_rrr = r1 if rrr1 > rrr2 else r2
            diffs.append(
                f"Reachability: Rank {better_rrr} has meaningfully higher recruiter response rate"
                f" ({rrr1:.0%} vs {rrr2:.0%})."
            )

        # Star predictor (career arc)
        star1 = s1.get("star_predictor", 0)
        star2 = s2.get("star_predictor", 0)
        if abs(star1 - star2) > 0.1:
            better_arc = r1 if star1 > star2 else r2
            diffs.append(
                f"Career arc: Rank {better_arc} shows stronger growth trajectory"
                f" (star={max(star1, star2):.2f} vs {min(star1, star2):.2f})."
            )

        if not diffs:
            diffs.append("Scores are nearly identical — toss-up. Use interview to differentiate.")

        comparisons.append({
            "rank_a": r1,
            "rank_b": r2,
            "candidate_a": c1.get("candidate_id"),
            "candidate_b": c2.get("candidate_id"),
            "title_a": title1,
            "title_b": title2,
            "score_a": round(s1.get("composite", 0), 4),
            "score_b": round(s2.get("composite", 0), 4),
            "key_differences": diffs,
            "recommendation": (
                f"Choose Rank {r1} if technical depth is the priority."
                f" Choose Rank {r2} if availability matters more."
            ) if notice1 > notice2 and tech1 > tech2 else (
                f"Rank {r1} leads on composite — move it first unless a specific gap applies."
            ),
        })

    return {"pairwise_tradeoffs": comparisons}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Anti-Bias Audit (Homogeneity Check)
# ─────────────────────────────────────────────────────────────────────────────

def anti_bias_audit(top_candidates: List[Tuple[dict, dict, int]]) -> dict:
    """
    Check whether the shortlist is suspiciously homogeneous.
    We can't audit for protected characteristics without that data.
    What we can check:
      - Geographic clustering (all from one city?)
      - Title clustering (all the same role type?)
      - YoE clustering (all from a narrow band?)
      - Company type clustering (all from FAANG?)
      - Hidden gems: non-traditional paths buried in ranking
    """
    top10 = [(c, s, r) for c, s, r in top_candidates if r <= 10]
    top20 = [(c, s, r) for c, s, r in top_candidates if r <= 20]

    flags = []
    insights = []

    # Geographic clustering
    locations = []
    for c, s, r in top10:
        loc = (c.get("profile", {}).get("location", "") or "").lower()
        for city in ["pune", "noida", "delhi", "gurgaon", "hyderabad", "bengaluru",
                     "bangalore", "mumbai", "chennai", "kochi"]:
            if city in loc:
                locations.append(city)
                break
        else:
            locations.append("other")

    if locations:
        from collections import Counter
        loc_counts = Counter(locations)
        top_city, top_count = loc_counts.most_common(1)[0]
        if top_count >= 7:
            flags.append(
                f"Geographic concentration: {top_count}/10 top candidates are from {top_city.title()}."
                " Consider whether this reflects true supply or a geographic filter artifact."
            )
        else:
            insights.append(
                f"Geographic spread looks healthy: {len(loc_counts)} different cities in top-10."
            )

    # YoE clustering
    yoes = [float(c.get("profile", {}).get("years_of_experience", 0) or 0) for c, s, r in top10]
    if yoes:
        yoe_min, yoe_max = min(yoes), max(yoes)
        yoe_range = yoe_max - yoe_min
        if yoe_range <= 2:
            flags.append(
                f"Experience band very narrow: top-10 all have {yoe_min:.0f}–{yoe_max:.0f} years."
                " The pool may be filtering out strong 4-year or 11-year candidates."
            )
        else:
            insights.append(f"YoE range in top-10: {yoe_min:.0f}–{yoe_max:.0f} years — healthy spread.")

    # Title clustering
    titles = [(c.get("profile", {}).get("current_title", "") or "").lower() for c, s, r in top10]
    ml_eng_count = sum(1 for t in titles if any(kw in t for kw in ["machine learning", "ml engineer"]))
    if ml_eng_count >= 7:
        flags.append(
            f"Title monoculture: {ml_eng_count}/10 are 'ML Engineer' variants."
            " NLP Engineers, Search Engineers, and Applied Scientists may be underrepresented."
        )

    # Hidden gems surfaced
    gem_count = sum(1 for c, s, r in top20 if s.get("hidden_gem"))
    if gem_count > 0:
        insights.append(
            f"{gem_count} hidden gems surfaced in top-20 — plain-language engineers"
            " who don't keyword-stuff but have strong career substance."
        )
    else:
        flags.append(
            "No hidden gems in top-20 — all top candidates are heavy self-promoters."
            " Consider whether some strong plain-language engineers were ranked lower than they should be."
        )

    # Consulting vs product
    consulting_count = 0
    for c, s, r in top10:
        career = c.get("career_history", [])
        all_consulting = all(
            any(cf in (ch.get("company", "") or "").lower() for cf in CONSULTING_FIRMS)
            for ch in career
        ) if career else False
        if all_consulting:
            consulting_count += 1

    if consulting_count >= 3:
        flags.append(
            f"{consulting_count}/10 top candidates have exclusively consulting backgrounds."
            " May lack product company context for a founding-team hire."
        )

    return {
        "anti_bias_audit": {
            "flags": flags,
            "insights": insights,
            "note": (
                "Note: This audit checks structural homogeneity only."
                " Protected characteristic data (gender, caste, etc.) is not available"
                " in this dataset, so those dimensions cannot be audited."
            ),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_cohort_analysis(top_candidates: List[Tuple[dict, dict, int]]) -> dict:
    """
    Run all four modules and return the full cohort report.

    Args:
        top_candidates: list of (candidate_dict, scores_dict, rank_int)

    Returns:
        dict with keys: jd_coverage, shared_gaps, pairwise_tradeoffs, anti_bias_audit,
                        summary
    """
    report = {}

    coverage = jd_coverage_report(top_candidates)
    report.update(coverage)

    gaps = shared_gaps(top_candidates)
    report.update(gaps)

    tradeoffs = pairwise_tradeoffs(top_candidates)
    report.update(tradeoffs)

    audit = anti_bias_audit(top_candidates)
    report.update(audit)

    # Summary
    n_supply_gaps = sum(1 for r in report["jd_coverage"] if r["supply_gap"])
    n_shared_gaps = len(report["shared_gaps"])
    n_bias_flags = len(report["anti_bias_audit"]["flags"])

    report["summary"] = {
        "total_candidates_analyzed": len(top_candidates),
        "top10_jd_requirements_unmet": n_supply_gaps,
        "shared_gaps_found": n_shared_gaps,
        "bias_flags": n_bias_flags,
        "overall_pool_quality": (
            "strong" if n_supply_gaps == 0 and n_shared_gaps == 0
            else "adequate" if n_supply_gaps <= 2
            else "constrained"
        ),
        "recommendation": (
            "Shortlist looks well-matched to JD requirements — proceed to outreach."
            if n_supply_gaps == 0 and n_shared_gaps == 0
            else f"Pool has {n_supply_gaps} unmet JD requirements and {n_shared_gaps} shared gaps."
                 " Review the detailed sections before finalizing outreach."
        ),
    }

    return report
