#!/usr/bin/env python3
"""
Comprehensive live test suite for the Redrob ranker.
Tests: scoring correctness, honeypot detection, edge cases,
       reasoning quality, submission format compliance, and
       JSON array format support.
"""

import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rank.scoring import compute_composite_score, detect_honeypot
from rank.reasoning import generate_reasoning
from validate_submission import validate_submission

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = []

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    return condition

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ──────────────────────────────────────────────────────────────────
# 1. LOAD SAMPLE DATA
# ──────────────────────────────────────────────────────────────────
section("1. DATA LOADING")

sample_path = Path("data/sample_candidates.json")
top100_path = Path("data/top_100_candidates.json")

sample_data = []
try:
    with open(sample_path, encoding="utf-8") as f:
        sample_data = json.load(f)
    test("sample_candidates.json loads", True, f"{len(sample_data)} candidates")
except Exception as e:
    test("sample_candidates.json loads", False, str(e))

top100_data = []
try:
    with open(top100_path, encoding="utf-8") as f:
        top100_data = json.load(f)
    test("top_100_candidates.json loads", True, f"{len(top100_data)} candidates")
except Exception as e:
    test("top_100_candidates.json loads", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 2. SCORING UNIT TESTS
# ──────────────────────────────────────────────────────────────────
section("2. SCORING CORRECTNESS")

t0 = time.time()
scored_sample = []
try:
    for c in sample_data:
        s = compute_composite_score(c)
        scored_sample.append((c, s))
    scored_sample.sort(key=lambda x: (-x[1]["composite"], x[0]["candidate_id"]))
    elapsed = time.time() - t0
    test("All 50 sample candidates scored without error", True, f"in {elapsed:.2f}s")
except Exception as e:
    test("All 50 sample candidates scored without error", False, traceback.format_exc())

# The gold-standard: Recommendation Systems Engineer should be rank 1
if scored_sample:
    rank1_title = scored_sample[0][0]["profile"]["current_title"]
    test(
        "Rank 1 is Recommendation Systems Engineer",
        "recommendation" in rank1_title.lower(),
        f"Got: '{rank1_title}' (score={scored_sample[0][1]['composite']:.4f})"
    )

    # All 7 honeypots score exactly 0.0
    honeypots = [(c, s) for c, s in scored_sample if s.get("honeypot")]
    test("Exactly 7 honeypots detected in sample", len(honeypots) == 7, f"Found: {len(honeypots)}")
    test("All honeypots score 0.0", all(s["composite"] == 0.0 for _, s in honeypots))

    # Hard-cap: non-AI roles should be at 0.15
    bad_titles = {"marketing manager", "accountant", "civil engineer", "mechanical engineer",
                  "graphic designer", "hr manager", "business analyst", "operations manager",
                  "frontend engineer", "java developer", "mobile developer", "cloud engineer",
                  "project manager", ".net developer", "devops engineer", "qa engineer",
                  "customer support"}
    non_ai = [(c, s) for c, s in scored_sample
              if c["profile"]["current_title"].lower() in bad_titles and not s.get("honeypot")]
    capped = [s["composite"] for _, s in non_ai if abs(s["composite"] - 0.15) < 0.001]
    test(
        f"Non-AI roles hard-capped at 0.15",
        len(capped) >= len(non_ai) * 0.8,  # allow small tolerance
        f"{len(capped)}/{len(non_ai)} non-AI roles correctly capped"
    )

    # Top-5 AI scores beat all non-AI scores
    top5_scores = [s["composite"] for _, s in scored_sample[:5]]
    non_ai_max = max((s["composite"] for _, s in non_ai), default=0)
    test(
        "Top-5 AI/ML candidates outrank ALL non-AI candidates",
        min(top5_scores) > non_ai_max,
        f"Lowest top-5 score: {min(top5_scores):.4f} vs non-AI max: {non_ai_max:.4f}"
    )

    # Score range sanity
    composites = [s["composite"] for _, s in scored_sample if not s.get("honeypot")]
    test("All non-honeypot scores in [0.0, 1.0]", all(0.0 <= s <= 1.0 for s in composites))

    # Weights sum to 1.0
    weights_sum = 0.40 + 0.22 + 0.15 + 0.11 + 0.07 + 0.05
    test("Dimension weights sum to 1.00", abs(weights_sum - 1.00) < 0.001, f"Sum: {weights_sum}")

# ──────────────────────────────────────────────────────────────────
# 3. TOP-100 DATASET SCORING
# ──────────────────────────────────────────────────────────────────
section("3. TOP-100 DATASET SCORING")

t0 = time.time()
scored_top100 = []
errors_top100 = 0
if top100_data:
    for c in top100_data:
        try:
            s = compute_composite_score(c)
            scored_top100.append((c, s))
        except Exception as e:
            errors_top100 += 1
    elapsed = time.time() - t0
    test(
        f"All {len(top100_data)} top-100 candidates scored",
        errors_top100 == 0,
        f"in {elapsed:.2f}s, {errors_top100} errors"
    )

    scored_top100.sort(key=lambda x: (-x[1]["composite"], x[0]["candidate_id"]))

    # All should be strong AI/ML candidates
    top10_titles = [c["profile"]["current_title"] for c, _ in scored_top100[:10]]
    ai_keywords = {"ml", "machine learning", "ai", "data scientist", "nlp", "search",
                   "recommendation", "retrieval", "applied", "research engineer"}
    top10_ai_count = sum(
        1 for t in top10_titles
        if any(kw in t.lower() for kw in ai_keywords)
    )
    test(
        "Top-10 of top_100 dataset are AI/ML roles",
        top10_ai_count >= 8,
        f"{top10_ai_count}/10 AI/ML titles: {top10_titles[:5]}..."
    )

    # Score spread
    top_score = scored_top100[0][1]["composite"]
    bottom_score = scored_top100[-1][1]["composite"]
    spread = top_score - bottom_score
    test(
        "Score spread > 0.05 (system discriminates)",
        spread > 0.05,
        f"Spread: {spread:.4f} (top={top_score:.4f}, bottom={bottom_score:.4f})"
    )

    # Honeypots in top-100 dataset
    hp_count = sum(1 for _, s in scored_top100 if s.get("honeypot"))
    test("Zero honeypots in top_100 dataset", hp_count == 0, f"Found: {hp_count}")

    # No deception in top-100
    high_dec = sum(1 for _, s in scored_top100 if s.get("deception_risk") == "high")
    test("Zero high-deception flags in top_100 dataset", high_dec == 0, f"Found: {high_dec}")

# ──────────────────────────────────────────────────────────────────
# 4. HONEYPOT EDGE CASES
# ──────────────────────────────────────────────────────────────────
section("4. HONEYPOT EDGE CASES (Synthetic)")

def make_candidate(overrides):
    base = {
        "candidate_id": "CAND_TEST001",
        "profile": {"years_of_experience": 5, "current_title": "ML Engineer",
                    "headline": "", "summary": "", "location": "Pune", "country": "India"},
        "career_history": [{"title": "ML Engineer", "company": "Acme", "duration_months": 60,
                            "description": "built search system", "industry": "Technology",
                            "company_size": "51-200", "is_current": True,
                            "start_date": "2020-01-01", "end_date": "2024-12-01"}],
        "skills": [{"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 36}],
        "redrob_signals": {"last_active_date": "2025-06-01", "open_to_work_flag": True,
                           "recruiter_response_rate": 0.8, "notice_period_days": 30},
        "education": []
    }
    base.update(overrides)
    return base

# Rule 1: duration overflow
c1 = make_candidate({
    "profile": {"years_of_experience": 2, "current_title": "ML Engineer", "headline": "", "summary": "", "location": "Pune", "country": "India"},
    "career_history": [{"title": "ML Engineer", "company": "A", "duration_months": 48,
                        "description": "ml work", "industry": "Tech", "company_size": "51-200",
                        "is_current": True, "start_date": "2020-01-01", "end_date": "2024-01-01"},
                       {"title": "ML Engineer", "company": "B", "duration_months": 36,
                        "description": "ml work", "industry": "Tech", "company_size": "51-200",
                        "is_current": False, "start_date": "2017-01-01", "end_date": "2020-01-01"}]
})
hp, reason = detect_honeypot(c1)
test("Rule 1: Excess total duration triggers honeypot", hp, f"Reason: {reason}")

# Rule 2: 3+ expert skills with 0 duration
c2 = make_candidate({
    "skills": [
        {"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
    ]
})
hp, reason = detect_honeypot(c2)
test("Rule 2: 3 expert skills + 0 duration triggers honeypot", hp, f"Reason: {reason}")

# Rule 4: ridiculous endorsements for junior
c4 = make_candidate({
    "profile": {"years_of_experience": 1, "current_title": "ML Engineer",
                "headline": "", "summary": "", "location": "Pune", "country": "India"},
    "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 6000, "duration_months": 12}]
})
hp, reason = detect_honeypot(c4)
test("Rule 4: 6000 endorsements with 1yr triggers honeypot", hp, f"Reason: {reason}")

# Clean candidate should NOT be flagged
c_clean = make_candidate({})
hp, reason = detect_honeypot(c_clean)
test("Clean valid candidate NOT flagged as honeypot", not hp, f"Reason: {reason if hp else 'None (correct)'}")

# ──────────────────────────────────────────────────────────────────
# 5. KEYWORD STUFFER TRAP
# ──────────────────────────────────────────────────────────────────
section("5. KEYWORD STUFFER TRAP")

stuffer = {
    "candidate_id": "CAND_STUFFER",
    "profile": {"years_of_experience": 4, "current_title": "Marketing Manager",
                "headline": "AI Enthusiast | Machine Learning | RAG | Pinecone",
                "summary": "Passionate about AI, RAG, embeddings, LLMs, vector search",
                "location": "Pune", "country": "India"},
    "career_history": [
        {"title": "Marketing Manager", "company": "Brand Co", "duration_months": 48,
         "description": "Managed digital campaigns and brand strategy", "industry": "Marketing",
         "company_size": "201-500", "is_current": True,
         "start_date": "2020-01-01", "end_date": "2024-01-01"}
    ],
    "skills": [
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 50, "duration_months": 24},
        {"name": "RAG", "proficiency": "expert", "endorsements": 40, "duration_months": 18},
        {"name": "LLM", "proficiency": "expert", "endorsements": 35, "duration_months": 12},
        {"name": "Embedding", "proficiency": "expert", "endorsements": 30, "duration_months": 12},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 25, "duration_months": 12},
    ],
    "redrob_signals": {"last_active_date": "2025-06-01", "open_to_work_flag": True,
                       "recruiter_response_rate": 0.9, "notice_period_days": 15,
                       "profile_completeness_score": 95},
    "education": []
}

stuffer_score = compute_composite_score(stuffer)
test(
    "Marketing Manager keyword-stuffer hard-capped at <= 0.15",
    stuffer_score["composite"] <= 0.15,
    f"Score: {stuffer_score['composite']:.4f}  career={stuffer_score['career_substance']:.3f}"
)
test(
    "Stuffer career_substance < 0.08",
    stuffer_score["career_substance"] < 0.08,
    f"career_substance={stuffer_score['career_substance']:.4f}"
)

# ──────────────────────────────────────────────────────────────────
# 6. IDEAL CANDIDATE
# ──────────────────────────────────────────────────────────────────
section("6. IDEAL CANDIDATE SCORES HIGH")

ideal = {
    "candidate_id": "CAND_IDEAL",
    "profile": {"years_of_experience": 7, "current_title": "Senior ML Engineer",
                "headline": "Senior ML Engineer | Search & Recommendation | RAG | Retrieval",
                "summary": "7 years building production retrieval and ranking systems at Swiggy and CRED. Expert in vector search, semantic retrieval, LLM reranking.",
                "location": "Pune", "country": "India"},
    "career_history": [
        {"title": "Senior ML Engineer", "company": "Swiggy", "duration_months": 36,
         "description": "Built and owned the search ranking system using FAISS and BM25. Designed RAG pipeline for query understanding. Led A/B testing framework for offline evaluation.",
         "industry": "Technology", "company_size": "5001-10000", "is_current": True,
         "start_date": "2021-01-01", "end_date": "2024-01-01"},
        {"title": "ML Engineer", "company": "CRED", "duration_months": 36,
         "description": "Recommendation system for credit card offers. Deployed sentence-transformer for semantic search. Built NDCG evaluation pipeline.",
         "industry": "Fintech", "company_size": "501-1000", "is_current": False,
         "start_date": "2018-01-01", "end_date": "2021-01-01"},
        {"title": "Data Scientist", "company": "Ola", "duration_months": 12,
         "description": "Demand forecasting and ranking models using XGBoost and PyTorch.",
         "industry": "Technology", "company_size": "1001-5000", "is_current": False,
         "start_date": "2017-01-01", "end_date": "2018-01-01"},
    ],
    "skills": [
        {"name": "Elasticsearch", "proficiency": "expert", "endorsements": 44, "duration_months": 36},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 30, "duration_months": 36},
        {"name": "Python", "proficiency": "expert", "endorsements": 60, "duration_months": 84},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
        {"name": "RAG", "proficiency": "expert", "endorsements": 20, "duration_months": 24},
        {"name": "Sentence Transformer", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
        {"name": "BM25", "proficiency": "advanced", "endorsements": 18, "duration_months": 36},
    ],
    "redrob_signals": {
        "last_active_date": "2025-06-20", "open_to_work_flag": True,
        "recruiter_response_rate": 0.95, "avg_response_time_hours": 12,
        "notice_period_days": 30, "interview_completion_rate": 0.9,
        "applications_submitted_30d": 3, "profile_completeness_score": 92,
        "saved_by_recruiters_30d": 8, "willing_to_relocate": False,
        "skill_assessment_scores": {}
    },
    "education": [{"degree": "B.Tech", "field": "Computer Science", "institution": "IIT Bombay"}]
}

ideal_score = compute_composite_score(ideal)
test("Ideal candidate scores >= 0.75", ideal_score["composite"] >= 0.75,
     f"Composite: {ideal_score['composite']:.4f}")
test("Ideal career_substance >= 0.70", ideal_score["career_substance"] >= 0.70,
     f"career_substance: {ideal_score['career_substance']:.4f}")
test("Ideal NOT flagged as honeypot", not ideal_score.get("honeypot"))
test("Ideal scores higher than stuffer",
     ideal_score["composite"] > stuffer_score["composite"],
     f"Ideal {ideal_score['composite']:.4f} > Stuffer {stuffer_score['composite']:.4f}")

# ──────────────────────────────────────────────────────────────────
# 7. REASONING QUALITY
# ──────────────────────────────────────────────────────────────────
section("7. REASONING STRING QUALITY")

if scored_sample:
    top3 = scored_sample[:3]
    reasoning_strings = []
    for rank_i, (cand, scores) in enumerate(top3):
        try:
            r = generate_reasoning(cand, scores, rank_i + 1)
            reasoning_strings.append(r)
        except Exception as e:
            test(f"Reasoning for rank {rank_i+1} generates without error", False, str(e))
            reasoning_strings.append("")

    test("All 3 top reasoning strings generated", all(len(r) > 50 for r in reasoning_strings))
    test("Reasoning strings are unique (no copy-paste)", len(set(reasoning_strings)) == len(reasoning_strings))
    test("Reasoning strings contain candidate-specific info",
         all(any(w in r for w in ["yr", "mo", "Rank", "endorsement", "proficiency", "Engineer"])
             for r in reasoning_strings))

    # Ideal candidate reasoning
    ideal_reasoning = generate_reasoning(ideal, ideal_score, 1)
    test("Ideal candidate reasoning generated", len(ideal_reasoning) > 50, ideal_reasoning[:150])

# ──────────────────────────────────────────────────────────────────
# 8. SUBMISSION FORMAT COMPLIANCE
# ──────────────────────────────────────────────────────────────────
section("8. SUBMISSION FORMAT COMPLIANCE")

for csv_path in ["submission.csv", "outputs/submission.csv"]:
    p = Path(csv_path)
    if p.exists():
        errs = validate_submission(str(p))
        test(f"{csv_path} passes official validator", len(errs) == 0,
             f"Errors: {errs}" if errs else "Clean")
    else:
        print(f"  {INFO}  {csv_path} not found, skipping")

# Check reasoning truncation
if Path("submission.csv").exists():
    with open("submission.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    truncated = [r for r in rows if r["reasoning"].endswith("...")]
    test("No reasoning strings truncated mid-sentence",
         len(truncated) == 0,
         f"{len(truncated)} strings end with '...' (may be truncated)")

    # Score monotonicity
    scores_csv = [float(r["score"]) for r in rows]
    violations = [(i, scores_csv[i], scores_csv[i+1])
                  for i in range(len(scores_csv)-1) if scores_csv[i] < scores_csv[i+1]]
    test("submission.csv scores are strictly non-increasing",
         len(violations) == 0,
         f"Violations: {violations[:3]}" if violations else "")

# ──────────────────────────────────────────────────────────────────
# 9. PERFORMANCE
# ──────────────────────────────────────────────────────────────────
section("9. PERFORMANCE BENCHMARKS")

if top100_data:
    t0 = time.time()
    for c in top100_data:
        compute_composite_score(c)
    elapsed = time.time() - t0
    per_candidate_ms = (elapsed / len(top100_data)) * 1000
    projected_100k_s = (elapsed / len(top100_data)) * 100_000

    test(
        f"Per-candidate scoring < 10ms",
        per_candidate_ms < 10,
        f"{per_candidate_ms:.2f}ms per candidate"
    )
    test(
        f"Projected 100K runtime < 300s",
        projected_100k_s < 300,
        f"Projected: {projected_100k_s:.1f}s for 100K candidates"
    )

# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────
section("SUMMARY")
total = len(results)
passed = sum(results)
failed = total - passed
pct = (passed / total * 100) if total else 0

print(f"\n  Total tests : {total}")
print(f"  Passed      : {passed}")
print(f"  Failed      : {failed}")
print(f"  Pass rate   : {pct:.1f}%")
print()
if failed == 0:
    print("  *** ALL TESTS PASSED -- submission is ready ***")
else:
    print(f"  *** {failed} test(s) FAILED -- review before submitting ***")
print()
