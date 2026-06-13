# Redrob AI Candidate Ranker

Intelligent candidate ranking system for the Redrob hackathon challenge. Ranks 100,000 candidates against a specific Senior AI Engineer job description using a 6-dimension composite scoring model — **without keyword matching, API calls, or GPU**.

The system reads *careers*, not skills lists. A Marketing Manager who listed 20 AI keywords doesn't rank. An ML engineer whose career descriptions mention "built search system" and has 44 endorsements on Elasticsearch does.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker (produces outputs/submission.csv)
python rank/ranker.py --candidates data/candidates.jsonl --out outputs/submission.csv

# 3. Validate the submission
python validate_submission.py outputs/submission.csv

# 4. Run the demo app
streamlit run demo/app.py

# 5. Generate the PDF deck
python deck/build_deck.py
```

**Runtime:** ~44 seconds on 100K candidates. No network. No GPU. Pure Python stdlib in the ranking step.

## Architecture

```
┌──────────────────────┐     ┌─────────────────────────────┐
│   OFFLINE PHASE      │     │      RANKING PHASE          │
│                      │     │                             │
│  JD Decomposition    │     │  Stream 100K JSONL          │
│  (precompute/)       │────▶│  6-Dimension Scoring        │
│                      │     │  heapq Top-100 Selection    │
│  role_requirements   │     │  Shadow Recruiter Narrative  │
│  .json               │     │  Cohort Analysis            │
└──────────────────────┘     │                             │
                             │  → submission.csv           │
                             │  → cohort_report.json       │
                             └─────────────────────────────┘
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/submission.csv` | 100 ranked candidates with scores and reasoning strings |
| `outputs/cohort_report.json` | JD coverage analysis, shared gaps, pairwise tradeoffs, anti-bias audit |
| `deck/presentation.pdf` | 12-slide PDF deck summarizing the system and results |

## 6 Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Career Substance | 40% | Job titles, career descriptions, domain depth. Reads the career trajectory, not the skills list. |
| Skill Credibility | 22% | Endorsements × duration × proficiency. Uncorroborated skills penalized to 0.35×. |
| Behavioral Availability | 15% | Last active date, recruiter response rate, notice period, open-to-work flag. |
| Experience Quality | 11% | Years of experience band (5-9yr sweet spot), tenure stability, product company bias. |
| Star Predictor | 7% | Career arc velocity: title progression, company quality, ownership language growth. |
| Location | 5% | Pune/Noida preferred, Tier-1 India cities accepted, willing-to-relocate signal. |

## Anti-Trap Mechanisms

| Trap | Detection | Effect |
|------|-----------|--------|
| **Keyword Stuffer** | `career_substance < 0.08` (e.g., Marketing Manager with AI skills) | Hard-capped composite at 0.15 |
| **Honeypot** | 5 impossibility rules (100yr experience, future dates, etc.) | Score = 0.0, excluded |
| **Behavioral Ghost** | High inactivity + low response rate | Heavy down-weight via availability dimension |
| **Hidden Gem** | Strong career descriptions but weak skills list | +0.05 bonus applied |

## Repository Structure

```
redrob-ranker/
├── rank/                          # Core ranking engine (pure stdlib, no API calls)
│   ├── ranker.py                  # CLI entry point — streams JSONL, scores, outputs CSV
│   ├── scoring.py                 # 6-dimension composite scoring with all weights
│   ├── reasoning.py               # Shadow Recruiter: 5-stage narrative engine
│   ├── cohort.py                  # Cohort Comparator: JD coverage, gaps, pairwise, bias audit
│   ├── prove_it.py                # Prove It Engine: cross-references claims vs evidence
│   ├── interview_blueprint.py     # Interview Blueprint: 3 candidate-specific questions
│   └── hm_translator.py           # HM Translator: CTO/VP/Founder briefs
├── demo/
│   └── app.py                     # Streamlit demo with 6 interactive panels
├── deck/
│   ├── build_deck.py              # PDF generation script
│   └── presentation.pdf           # Generated 12-slide deck
├── precompute/
│   └── jd_decompose.py            # JD → role_requirements.json extraction
├── artifacts/
│   └── role_requirements.json     # Pre-computed JD decomposition
├── data/
│   ├── candidates.jsonl           # Full 100K candidate dataset (not in repo)
│   └── sample_candidates.json     # 50-candidate sample for testing
├── outputs/
│   ├── submission.csv             # Final ranked submission
│   └── cohort_report.json         # Pool-level analysis
├── validate_submission.py         # Official submission validator
├── test_sample.py                 # Sample test runner
└── requirements.txt               # Python dependencies
```

## Demo App

The Streamlit demo provides 6 interactive panels per candidate:

1. **Shadow Recruiter** — Full narrative reasoning string with career arc analysis
2. **Prove It Engine** — Cross-references profile claims against career evidence (corroborated / unverified / contradicted)
3. **Deception Detector** — Risk flags and honeypot detection
4. **Interview Blueprint** — 3 candidate-specific questions grounded in profile gaps
5. **Hiring Manager Translator** — CTO / VP Engineering / Founder briefs (substantively different focus per audience)
6. **Behavioral Signals** — Raw platform signals: response rate, GitHub, assessments

Plus a **Cohort Analysis** tab showing pool-level insights: JD coverage gaps, shared weaknesses, pairwise tradeoffs between adjacent ranks, and anti-bias audit.

## Key Design Decisions

- **No API calls at rank time.** The entire ranking step runs on stdlib Python. JD decomposition is pre-computed.
- **Streaming architecture.** Candidates are processed one-at-a-time from JSONL using `heapq` for O(1) memory. Only the top-100 are kept in memory.
- **Deterministic phrase variation.** Reasoning strings use `md5(candidate_id + salt)` to select from phrase banks — reproducible across runs, unique per candidate.
- **Career-first scoring.** The 40% weight on career substance means the system reads job descriptions and titles, not just skills keywords.
- **Hard gates.** Keyword stuffers (`career_substance < 0.08`) are hard-capped regardless of how many AI skills they list.
