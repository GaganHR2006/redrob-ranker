#!/usr/bin/env python3
"""
Quick test of scoring on sample_candidates.json.
Verifies our ranker correctly:
1. Ranks Recommendation Systems Engineer and actual ML engineers high
2. Ranks Marketing Managers, Civil Engineers, etc. low
3. Does NOT rank keyword-stuffers high based on skills alone
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rank.scoring import compute_composite_score
from rank.reasoning import generate_reasoning

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_candidates.json"

def main():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sample candidates\n")

    results = []
    for candidate in data:
        scores = compute_composite_score(candidate)
        results.append((candidate, scores))

    results.sort(key=lambda x: (-x[1]["composite"], x[0]["candidate_id"]))

    print("=" * 90)
    print("RANKING RESULTS (our improved ranker)")
    print("=" * 90)
    print(f"{'Rank':>4} {'Candidate':12} {'Score':>6} {'Career':>6} {'Skill':>5} {'Exp':>5} {'Avail':>5}  Title (YoE)")
    print("-" * 90)

    for rank_idx, (cand, s) in enumerate(results):
        rank = rank_idx + 1
        p = cand["profile"]
        cid = cand["candidate_id"]
        title = p["current_title"][:28]
        yoe = p["years_of_experience"]
        gem = " [GEM]" if s.get("hidden_gem") else ""
        deception = " [WARN]" if s.get("deception_risk") == "high" else ""
        honeypot = " [HONEYPOT]" if s.get("honeypot") else ""
        
        print(
            f"{rank:>4} {cid:12} {s['composite']:>6.4f} "
            f"{s['career_substance']:>6.3f} {s['skill_credibility']:>5.3f} "
            f"{s['experience_quality']:>5.3f} {s['behavioral_availability']:>5.3f}  "
            f"{title:<28} ({yoe:.0f}yr){gem}{deception}{honeypot}"
        )

    print()
    print("Key: [GEM]=hidden gem  [WARN]=high deception  [HONEYPOT]=honeypot")
    print()

    # Show reasoning for top 5 and bottom 3
    print("=" * 90)
    print("REASONING STRINGS (top 5 + bottom 3)")
    print("=" * 90)
    combined = [(i+1, cand, s) for i, (cand, s) in enumerate(results[:5])]
    combined += [(len(results)-2+i, cand, s) for i, (cand, s) in enumerate(results[-3:])]
    for rank, cand, s in combined:
        reasoning = generate_reasoning(cand, s, rank)
        print("\nRank %d (%s -- %s):" % (rank, cand['candidate_id'], cand['profile']['current_title']))
        print("  " + reasoning)

    # Validation checks
    print()
    print("=" * 90)
    print("QUALITY CHECKS")
    print("=" * 90)
    
    top5_titles = [results[i][0]["profile"]["current_title"] for i in range(5)]
    print(f"Top 5 titles: {top5_titles}")
    
    bad_in_top10 = [
        results[i][0]["profile"]["current_title"]
        for i in range(min(10, len(results)))
        if any(bad in results[i][0]["profile"]["current_title"].lower()
               for bad in ["marketing", "civil", "mechanical", "accountant", "graphic", "customer support"])
    ]
    if bad_in_top10:
        print(f"WARNING: Non-AI roles in top 10: {bad_in_top10}")
    else:
        print("PASS: No clearly non-AI roles in top 10")
    
    honeypots = sum(1 for _, s in results if s.get("honeypot"))
    print(f"Honeypots detected: {honeypots}")
    
    high_deception = sum(1 for _, s in results if s.get("deception_risk") == "high")
    print(f"High-deception flags: {high_deception}")
    
    gems = sum(1 for _, s in results if s.get("hidden_gem"))
    print(f"Hidden gems surfaced: {gems}")

if __name__ == "__main__":
    main()
