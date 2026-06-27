#!/usr/bin/env python3
"""
ranker.py — Redrob AI Candidate Ranker
Phase 2: Fast CPU-only ranking step. No network. No GPU. Must complete in <5 minutes.

Usage:
    python rank/ranker.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv

The ranker:
1. Streams candidates.jsonl (never loads all 100K into memory at once)
2. Computes composite score for each candidate
3. Keeps running top-N heap (memory efficient)
4. Re-loads top-100 candidates for reasoning generation
5. Writes submission.csv
"""

import argparse
import csv
import gzip
import heapq
import json
import sys
import time
from pathlib import Path

# Add rank/ to path so we can import scoring and reasoning
sys.path.insert(0, str(Path(__file__).parent))
from scoring import compute_composite_score
from reasoning import generate_reasoning
from cohort import run_cohort_analysis


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING LOADER
# ─────────────────────────────────────────────────────────────────────────────

def stream_candidates(path: str):
    """Stream candidates from JSONL or JSONL.GZ. Never loads all into memory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    
    opener = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, "r", encoding="utf-8")
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines


def load_top_candidates_by_ids(path: str, target_ids: set[str]) -> dict[str, dict]:
    """Second pass: load only the candidates we need for reasoning generation."""
    found: dict[str, dict] = {}
    for candidate in stream_candidates(path):
        cid = candidate.get("candidate_id", "")
        if cid in target_ids:
            found[cid] = candidate
        if len(found) == len(target_ids):
            break  # Found all — stop early
    return found


# ─────────────────────────────────────────────────────────────────────────────
# RANKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    candidates_path: str,
    output_path: str,
    top_n: int = 100,
    verbose: bool = True,
) -> list[dict]:
    """
    Main ranking pipeline.
    
    Memory strategy: uses a min-heap of size top_n to track the top candidates
    without loading all 100K scores into memory simultaneously.
    The heap holds (composite_score, candidate_id) tuples.
    """
    t_start = time.time()
    
    if verbose:
        print("=" * 60)
        print("  Redrob AI Candidate Ranker")
        print("=" * 60)
        print(f"  Input:  {candidates_path}")
        print(f"  Output: {output_path}")
        print(f"  Ranking top {top_n} candidates")
        print()
    
    # ── Pass 1: Score all candidates, keep top-N in a min-heap ──────────────
    if verbose:
        print("[1/3] Scoring candidates...")
    
    # Min-heap: (composite_score, candidate_id, score_dict)
    # Use min-heap so we can efficiently drop the lowest when heap is full
    heap: list[tuple] = []
    count = 0
    honeypot_count = 0
    
    for candidate in stream_candidates(candidates_path):
        scores = compute_composite_score(candidate)
        count += 1
        
        if scores.get("honeypot"):
            honeypot_count += 1
            continue  # Skip honeypots entirely
        
        composite = scores["composite"]
        cid = scores["candidate_id"]
        
        # Push to min-heap (negate for max-heap behavior, tiebreak by cid ascending)
        entry = (composite, cid, scores)  
        
        if len(heap) < top_n:
            heapq.heappush(heap, entry)
        elif composite > heap[0][0]:
            heapq.heapreplace(heap, entry)
        
        if verbose and count % 10000 == 0:
            elapsed = time.time() - t_start
            rate = count / elapsed
            print(f"  Scored {count:>7,} candidates... "
                  f"({rate:.0f}/sec, {elapsed:.1f}s elapsed, "
                  f"honeypots={honeypot_count})")
    
    t_scored = time.time()
    
    if verbose:
        print(f"  Done! Scored {count:,} candidates in {t_scored - t_start:.1f}s")
        print(f"  Honeypots detected and excluded: {honeypot_count}")
        print()
    
    # ── Sort top-N by score descending (break ties by cid ascending) ─────────
    if verbose:
        print("[2/3] Sorting top candidates...")
    
    # Sort from heap: highest score first, tie-break by candidate_id ascending
    top_results = sorted(heap, key=lambda x: (-x[0], x[1]))[:top_n]
    
    if len(top_results) < top_n:
        if verbose:
            print(f"  WARNING: Only {len(top_results)} valid candidates found (requested {top_n})")
    
    # ── Load full candidate data for top-N (for reasoning generation) ────────
    top_ids = {entry[1] for entry in top_results}
    if verbose:
        print(f"  Loading full profiles for {len(top_ids)} top candidates...")
    
    top_candidates_data = load_top_candidates_by_ids(candidates_path, top_ids)
    
    t_loaded = time.time()
    
    # ── Pass 2: Generate reasoning and write CSV ──────────────────────────────
    if verbose:
        print()
        print("[3/3] Generating reasoning and writing submission CSV...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    rows_written = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx, (composite, cid, scores) in enumerate(top_results):
            rank = rank_idx + 1
            candidate = top_candidates_data.get(cid, {
                "candidate_id": cid,
                "profile": {},
                "career_history": [],
                "skills": [],
                "redrob_signals": {},
                "education": [],
            })
            
            # Ensure score is non-increasing (floating point safety)
            # The sort guarantees this — but round for CSV
            score_val = round(composite, 6)
            
            reasoning = generate_reasoning(candidate, scores, rank)
            writer.writerow([cid, rank, score_val, reasoning])
            rows_written += 1
    
    t_end = time.time()
    total_time = t_end - t_start

    # ── Cohort Analysis (post-processing) ───────────────────────────────
    # Build list of (candidate, scores, rank) tuples for the cohort module
    top_candidates_for_cohort = []
    for rank_idx, (composite, cid, scores) in enumerate(top_results):
        candidate = top_candidates_data.get(cid, {})
        top_candidates_for_cohort.append((candidate, scores, rank_idx + 1))

    cohort_report = run_cohort_analysis(top_candidates_for_cohort)

    # Save cohort report
    cohort_path = Path(output_path).parent / "cohort_report.json"
    with open(cohort_path, "w", encoding="utf-8") as f:
        json.dump(cohort_report, f, indent=2, default=str)
    if verbose:
        print(f"  Cohort report written to: {cohort_path}")
    
    if verbose:
        print(f"  Submission written to: {output_path}")
        print()
        print("=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        
        # Show top 10
        print("\n  TOP 10 CANDIDATES:")
        for rank_idx, (composite, cid, scores) in enumerate(top_results[:10]):
            cand = top_candidates_data.get(cid, {})
            p = cand.get("profile", {})
            title = p.get("current_title", "?")[:30]
            yoe = p.get("years_of_experience", 0)
            loc = p.get("location", "?")[:20]
            print(
                f"  {rank_idx+1:3d}. {cid} | {composite:.4f} | "
                f"{title:<30s} | {yoe:.0f}yr | {loc}"
            )
        
        # Score distribution
        composites = [entry[0] for entry in top_results]
        print(f"\n  SCORE DISTRIBUTION:")
        print(f"    Top-1:    {composites[0]:.4f}")
        if len(composites) >= 10:
            print(f"    Top-10:   {composites[9]:.4f}")
        if len(composites) >= 50:
            print(f"    Top-50:   {composites[49]:.4f}")
        if len(composites) >= 100:
            print(f"    Top-100:  {composites[99]:.4f}")
        
        # Deception / quality stats
        high_deception = sum(1 for _, _, s in top_results if s.get("deception_risk") == "high")
        medium_deception = sum(1 for _, _, s in top_results if s.get("deception_risk") == "medium")
        hidden_gems = sum(1 for _, _, s in top_results if s.get("hidden_gem"))
        
        print(f"\n  QUALITY CHECKS:")
        print(f"    Honeypots excluded:          {honeypot_count}")
        print(f"    High-deception in top-100:   {high_deception} (should be <5)")
        print(f"    Medium-deception in top-100: {medium_deception}")
        print(f"    Hidden gems surfaced:         {hidden_gems}")
        print(f"    Total candidates scored:     {count:,}")
        print(f"    Total runtime:               {total_time:.1f}s")
        print(f"    Rows in CSV:                 {rows_written}")

        # Cohort summary
        cohort_summary = cohort_report.get("summary", {})
        n_supply_gaps = cohort_summary.get("top10_jd_requirements_unmet", 0)
        n_shared = cohort_summary.get("shared_gaps_found", 0)
        n_bias = cohort_summary.get("bias_flags", 0)
        pool_quality = cohort_summary.get("overall_pool_quality", "?")
        print(f"\n  COHORT ANALYSIS:")
        print(f"    Pool quality:                {pool_quality}")
        print(f"    JD requirements with gaps:   {n_supply_gaps}")
        print(f"    Shared gaps in top-10:       {n_shared}")
        print(f"    Homogeneity flags:           {n_bias}")
        print()
        
        if total_time > 300:
            print("  WARNING: Runtime exceeded 5-minute target!")
        else:
            headroom = 300 - total_time
            print(f"  Runtime {total_time:.1f}s — {headroom:.0f}s under 5-minute limit")
        
        print()
        print("  Run: python validate_submission.py", output_path)
    
    return [
        {"candidate_id": cid, "rank": i+1, "score": composite, "scores": scores}
        for i, (composite, cid, scores) in enumerate(top_results)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranker — ranks top 100 candidates for Senior AI Engineer role",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rank/ranker.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
  python rank/ranker.py --candidates ./data/candidates.jsonl --out ./outputs/test.csv --top 100
  python rank/ranker.py --candidates ./data/sample_candidates.json --out ./outputs/sample_test.csv
        """
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz (default: ./data/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="./outputs/submission.csv",
        help="Output CSV path (default: ./outputs/submission.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of candidates to rank (default: 100)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()
    
    if not Path(args.candidates).exists():
        print(f"ERROR: Candidates file not found: {args.candidates}")
        print("\nMake sure you've placed candidates.jsonl in ./data/ or pass --candidates <path>")
        sys.exit(1)
    
    rank_candidates(
        candidates_path=args.candidates,
        output_path=args.out,
        top_n=args.top,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
