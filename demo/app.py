"""
Redrob AI Candidate Ranker — Streamlit Demo App
================================================
Upload a sample of candidates (JSON format) and see the full evaluation:
- Shadow Recruiter reasoning
- Prove It Engine results  
- Deception Detector flags
- Star Predictor assessment
- Interview Blueprint
- Hiring Manager Translator (CTO / VP / Founder views)
- Cohort Analysis across the pool
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rank.scoring import compute_composite_score
from rank.reasoning import generate_reasoning
from rank.prove_it import run_prove_it, prove_it_summary
from rank.interview_blueprint import generate_interview_blueprint
from rank.hm_translator import generate_hm_briefs

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 12px; color: #94a3b8; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }
    
    .score-bar-container { margin: 6px 0; }
    .score-label { font-size: 13px; color: #94a3b8; margin-bottom: 2px; }
    
    .verdict-strong { background: #064e3b; color: #34d399; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .verdict-likely { background: #1e3a5f; color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .verdict-possible { background: #451a03; color: #fb923c; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .verdict-poor { background: #450a0a; color: #f87171; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
    
    .corroborated { color: #34d399; font-weight: 500; }
    .unverified { color: #fbbf24; font-weight: 500; }
    .contradicted { color: #f87171; font-weight: 500; }
    
    .gem-badge { background: linear-gradient(135deg, #f59e0b, #ef4444); color: white; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
    .honeypot-badge { background: #7f1d1d; color: #fca5a5; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
    .warn-badge { background: #78350f; color: #fcd34d; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
    
    .shadow-recruiter-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-left: 4px solid #818cf8;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        font-style: italic;
        color: #c7d2fe;
        line-height: 1.7;
        margin: 12px 0;
    }
    
    .interview-q {
        background: #0f172a;
        border: 1px solid #1e40af;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
    
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 18px;
        font-weight: 800;
        width: 44px; height: 44px;
        border-radius: 50%;
        text-align: center;
        line-height: 44px;
        margin-right: 12px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 0.7: return "#34d399"
    if score >= 0.4: return "#fbbf24"
    return "#f87171"

def score_bar(label: str, value: float) -> str:
    color = score_color(value)
    pct = int(value * 100)
    return f"""
    <div class="score-bar-container">
        <div class="score-label">{label}: <strong style="color:{color}">{pct}%</strong></div>
        <div style="height:6px; background:#1e293b; border-radius:3px; overflow:hidden;">
            <div style="height:100%; width:{pct}%; background:{color}; border-radius:3px; transition:width 0.4s;"></div>
        </div>
    </div>
    """

def format_verdict(fit: str) -> str:
    mapping = {
        "strong": '<span class="verdict-strong">Strong Fit</span>',
        "likely": '<span class="verdict-likely">Likely Fit</span>',
        "possible": '<span class="verdict-possible">Possible Fit</span>',
        "poor": '<span class="verdict-poor">Poor Fit</span>',
    }
    return mapping.get(fit, f'<span>{fit}</span>')

def _days_since(date_str):
    from datetime import date, datetime
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (date.today() - d).days
    except:
        return 9999

def derive_verdict(scores: dict) -> str:
    composite = scores.get("composite", 0)
    if composite >= 0.75: return "strong"
    if composite >= 0.55: return "likely"
    if composite >= 0.35: return "possible"
    return "poor"

# ─── Candidate Evaluation Panel ───────────────────────────────────────────────

def show_candidate_evaluation(candidate: dict, scores: dict, rank: int):
    """Full evaluation panel for a single candidate."""
    p = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    title = p.get("current_title", "?")
    company = p.get("current_company", "?")
    yoe = p.get("years_of_experience", 0)
    location = p.get("location", "?")
    headline = p.get("headline", "")
    
    # Header
    verdict = derive_verdict(scores)
    col_rank, col_info, col_verdict = st.columns([1, 5, 2])
    with col_rank:
        st.markdown(f'<div class="rank-badge">#{rank}</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown(f"**{title}** at **{company}**")
        st.markdown(f"*{headline}*")
        st.markdown(f"📍 {location} &nbsp;|&nbsp; {yoe:.0f} years experience", unsafe_allow_html=True)
    with col_verdict:
        badges = ""
        if scores.get("honeypot"):
            badges += '<span class="honeypot-badge">HONEYPOT</span> '
        if scores.get("hidden_gem"):
            badges += '<span class="gem-badge">HIDDEN GEM</span> '
        if scores.get("deception_risk") == "high":
            badges += '<span class="warn-badge">HIGH DECEPTION RISK</span>'
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
        st.markdown(format_verdict(verdict), unsafe_allow_html=True)
    
    st.divider()
    
    # Score bars
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Score Breakdown**")
        st.markdown(score_bar("Career Substance", scores.get("career_substance", 0)), unsafe_allow_html=True)
        st.markdown(score_bar("Skill Credibility", scores.get("skill_credibility", 0)), unsafe_allow_html=True)
        st.markdown(score_bar("Experience Quality", scores.get("experience_quality", 0)), unsafe_allow_html=True)
    with col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(score_bar("Behavioral Availability", scores.get("behavioral_availability", 0)), unsafe_allow_html=True)
        st.markdown(score_bar("Location Fit", scores.get("location", 0)), unsafe_allow_html=True)
        st.markdown(f"<br><b style='font-size:20px;color:{score_color(scores['composite'])}'>Composite: {scores['composite']:.4f}</b>", unsafe_allow_html=True)
    
    st.divider()
    
    # Tabs for the 8 components
    tabs = st.tabs(["Shadow Recruiter", "Prove It Engine", "Deception Detector", 
                    "Interview Blueprint", "Hiring Manager Briefs", "Signals"])
    
    # ── Shadow Recruiter ──
    with tabs[0]:
        reasoning = generate_reasoning(candidate, scores, rank)
        st.markdown(f'<div class="shadow-recruiter-box">{reasoning}</div>', unsafe_allow_html=True)
        
        st.markdown("**Career History**")
        for ch in career:
            with st.expander(f"{ch.get('title')} at {ch.get('company')} — {ch.get('duration_months')}mo"):
                st.markdown(ch.get("description", "No description"))
    
    # ── Prove It Engine ──
    with tabs[1]:
        st.markdown("**Claim Verification**")
        pi_results = run_prove_it(candidate)
        summary = prove_it_summary(pi_results)
        st.markdown(f"*Trust signal: {summary}*")
        st.markdown("---")
        
        for claim_item in pi_results:
            verdict_type = claim_item.get("verdict", "unverified")
            claim_text = claim_item.get("claim", "")
            evidence_text = claim_item.get("evidence", "")
            source = claim_item.get("source", "")
            
            col_v, col_c, col_e = st.columns([1, 3, 4])
            with col_v:
                cls = verdict_type
                st.markdown(f'<span class="{cls}">{verdict_type.upper()}</span>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f"**{claim_text}**")
            with col_e:
                st.markdown(f"*{evidence_text}*")
                if source:
                    st.caption(f"Source: {source}")
    
    # ── Deception Detector ──
    with tabs[2]:
        risk = scores.get("deception_risk", "low")
        flags = scores.get("deception_flags", [])
        
        risk_colors = {"low": "#34d399", "medium": "#fbbf24", "high": "#f87171", "n/a": "#94a3b8"}
        risk_color = risk_colors.get(risk, "white")
        st.markdown(f"**Deception Risk:** <span style='color:{risk_color};font-size:18px;font-weight:700'>{risk.upper()}</span>", unsafe_allow_html=True)
        
        if flags:
            st.warning("Flags detected:")
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("No deception flags detected — profile appears consistent")
        
        # Honeypot
        if scores.get("honeypot"):
            st.error(f"**HONEYPOT**: {scores.get('honeypot_reason', 'Unknown reason')}")
    
    # ── Interview Blueprint ──
    with tabs[3]:
        st.markdown("**Targeted Interview Questions**")
        
        questions = generate_interview_blueprint(candidate, scores)
        for i, q in enumerate(questions):
            q_text = q.get('question', '')
            label = q_text[:80] + '...' if len(q_text) > 80 else q_text
            with st.expander(f"Q{i+1}: {label}"):
                st.markdown(f"**Full question:** {q_text}")
                st.markdown(f"**Why this question:** {q.get('why_this_question', '')}")
                col_s, col_w = st.columns(2)
                with col_s:
                    st.markdown("✅ **Strong answer looks like:**")
                    st.info(q.get('strong_answer', q.get('strong', '')))
                with col_w:
                    st.markdown("❌ **Weak answer looks like:**")
                    st.error(q.get('weak_answer', q.get('weak', '')))
    
    # ── Hiring Manager Briefs ──
    with tabs[4]:
        briefs = generate_hm_briefs(candidate, scores)
        
        tab_cto, tab_vp, tab_founder = st.tabs(["CTO Brief", "VP Engineering Brief", "Founder Brief"])
        with tab_cto:
            st.markdown(briefs.get("cto", "N/A"))
        with tab_vp:
            st.markdown(briefs.get("vp_engineering", briefs.get("vp", "N/A")))
        with tab_founder:
            st.markdown(briefs.get("founder", "N/A"))
    
    # ── Behavioral Signals ──
    with tabs[5]:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            days = _days_since(rs.get("last_active_date", "2020-01-01"))
            st.metric("Days Since Last Active", f"{days}d", delta=None)
            st.metric("Recruiter Response Rate", f"{rs.get('recruiter_response_rate', 0):.0%}")
            st.metric("Notice Period", f"{rs.get('notice_period_days', '?')} days")
        with col_b:
            st.metric("Open To Work", "Yes" if rs.get("open_to_work_flag") else "No")
            st.metric("GitHub Score", f"{rs.get('github_activity_score', -1)}")
            st.metric("Applications (30d)", rs.get("applications_submitted_30d", 0))
        with col_c:
            st.metric("Profile Completeness", f"{rs.get('profile_completeness_score', 0):.0f}%")
            st.metric("Saved by Recruiters (30d)", rs.get("saved_by_recruiters_30d", 0))
            st.metric("Interview Completion Rate", f"{rs.get('interview_completion_rate', 0):.0%}")
        
        # Skill assessments
        assessments = rs.get("skill_assessment_scores", {}) or {}
        if assessments:
            st.markdown("**Platform Skill Assessment Scores:**")
            df_assess = pd.DataFrame(list(assessments.items()), columns=["Skill", "Score"])
            df_assess = df_assess.sort_values("Score", ascending=False)
            st.bar_chart(df_assess.set_index("Skill")["Score"])



# Old inline _generate_interview_questions and _generate_hm_briefs removed.
# Now using real modules: rank.interview_blueprint and rank.hm_translator


# ─── Cohort Analysis ──────────────────────────────────────────────────────────

def show_cohort_analysis(candidates_with_scores: list):
    """Cross-candidate cohort analysis."""
    st.header("Cohort Analysis")
    
    all_scores = [s for _, s in candidates_with_scores if not s.get("honeypot")]
    
    # Pool health
    strong_fits = sum(1 for s in all_scores if s.get("composite", 0) >= 0.7)
    likely_fits = sum(1 for s in all_scores if 0.5 <= s.get("composite", 0) < 0.7)
    honeypots = sum(1 for _, s in candidates_with_scores if s.get("honeypot"))
    hidden_gems = sum(1 for s in all_scores if s.get("hidden_gem"))
    high_deception = sum(1 for s in all_scores if s.get("deception_risk") == "high")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, val, color in [
        (col1, "Strong Fits (≥0.7)", strong_fits, "#34d399"),
        (col2, "Likely Fits (≥0.5)", likely_fits, "#60a5fa"),
        (col3, "Honeypots Detected", honeypots, "#f87171"),
        (col4, "Hidden Gems", hidden_gems, "#f59e0b"),
        (col5, "High Deception Risk", high_deception, "#fb923c"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("")
    
    # Score distribution
    composites = [s.get("composite", 0) for s in all_scores]
    if composites:
        df_hist = pd.DataFrame({"Composite Score": composites})
        st.subheader("Score Distribution")
        st.bar_chart(df_hist["Composite Score"].value_counts(bins=20).sort_index())
    
    # Top candidates comparison
    sorted_candidates = sorted(candidates_with_scores, key=lambda x: (-x[1].get("composite", 0), x[0].get("candidate_id", "")))
    
    st.subheader("Ranked Candidates Table")
    rows = []
    for rank_idx, (c, s) in enumerate(sorted_candidates[:50]):
        p = c.get("profile", {})
        rs = c.get("redrob_signals", {})
        days = _days_since(rs.get("last_active_date", "2020-01-01"))
        rows.append({
            "Rank": rank_idx + 1,
            "Candidate ID": c.get("candidate_id"),
            "Title": p.get("current_title"),
            "Company": p.get("current_company"),
            "YoE": p.get("years_of_experience"),
            "Location": p.get("location"),
            "Score": round(s.get("composite", 0), 4),
            "Career": round(s.get("career_substance", 0), 3),
            "Skills": round(s.get("skill_credibility", 0), 3),
            "Notice (days)": rs.get("notice_period_days"),
            "Days Inactive": days,
            "Gem": "YES" if s.get("hidden_gem") else "",
            "Deception": s.get("deception_risk", ""),
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Common gaps in pool
    st.subheader("Pool Insights")
    avg_career = sum(s.get("career_substance", 0) for s in all_scores) / max(1, len(all_scores))
    avg_skill = sum(s.get("skill_credibility", 0) for s in all_scores) / max(1, len(all_scores))
    avg_avail = sum(s.get("behavioral_availability", 0) for s in all_scores) / max(1, len(all_scores))
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Avg Career Substance", f"{avg_career:.2f}")
    with col_b:
        st.metric("Avg Skill Credibility", f"{avg_skill:.2f}")
    with col_c:
        st.metric("Avg Behavioral Availability", f"{avg_avail:.2f}")
    
    if avg_career < 0.3:
        st.warning("**Pool Alert**: Low average career substance — most candidates don't have strong AI/ML backgrounds. Consider broadening sourcing strategy.")
    if avg_avail < 0.5:
        st.info("**Engagement Note**: Average platform activity is low. Recommend proactive outreach rather than waiting for candidates to respond.")


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("Redrob AI Candidate Ranker")
    st.markdown("**Senior AI Engineer — Founding Team** | Intelligent Candidate Discovery System")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Candidates")
        st.markdown("Upload a JSON file containing an array of candidate objects.")
        
        uploaded = st.file_uploader("Choose JSON file", type=["json"])
        
        st.divider()
        st.markdown("**Try with sample data:**")
        use_sample = st.button("Load sample_candidates.json")
        
        st.divider()
        st.markdown("**About this system:**")
        st.markdown("""
        - 5-dimension composite scoring
        - Honeypot detection (impossible profiles)
        - Keyword-stuffer detection
        - Hidden gem surfacing
        - No API keys required
        - Runs in <5 min on 100K candidates
        """)
    
    # Load candidates
    candidates_data = None
    
    if uploaded:
        try:
            content = uploaded.read()
            candidates_data = json.loads(content)
            if isinstance(candidates_data, dict):
                candidates_data = [candidates_data]
            st.sidebar.success(f"Loaded {len(candidates_data)} candidates")
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            return
    
    elif use_sample:
        sample_path = Path(__file__).parent.parent / "data" / "sample_candidates.json"
        if sample_path.exists():
            with open(sample_path, encoding="utf-8") as f:
                candidates_data = json.load(f)
            st.sidebar.success(f"Loaded {len(candidates_data)} sample candidates")
        else:
            st.error("sample_candidates.json not found in data/. Please upload a JSON file.")
            return
    
    if candidates_data is None:
        st.info("Upload a candidates JSON file or click 'Load sample_candidates.json' to get started.")
        
        st.markdown("""
        ### What this system evaluates
        
        | Component | Description |
        |-----------|-------------|
        | **Career Substance** | Did this person actually build ML/AI/search systems? Reads job titles + profile summary |
        | **Skill Credibility** | Are skills corroborated by endorsements, duration, and career descriptions? |
        | **Experience Quality** | Right amount of experience (5-9yr preferred), product company, not pure research |
        | **Behavioral Availability** | Last active, response rate, notice period, open-to-work signals |
        | **Location Fit** | Pune/Noida preferred, Tier-1 cities acceptable |
        
        **Anti-trap mechanisms:**
        - Keyword stuffers (Marketing Manager with AI skills) → capped at 0.15
        - Honeypots (impossible timelines) → zeroed completely  
        - Behavioral ghosts (inactive 180d + 5% response) → heavily down-weighted
        - Hidden gems (plain-language engineers) → 0.05 bonus applied
        """)
        return
    
    # Score all candidates
    with st.spinner(f"Scoring {len(candidates_data)} candidates..."):
        candidates_with_scores = []
        for c in candidates_data:
            s = compute_composite_score(c)
            candidates_with_scores.append((c, s))
        
        # Sort by score
        candidates_with_scores.sort(key=lambda x: (-x[1].get("composite", 0), x[0].get("candidate_id", "")))
    
    # Tabs
    tab_explorer, tab_cohort = st.tabs(["Candidate Explorer", "Cohort Analysis"])
    
    with tab_explorer:
        st.header("Ranked Candidates")
        
        if len(candidates_data) <= 20:
            # For small samples, show all with full evaluation expandable
            for rank_idx, (candidate, scores) in enumerate(candidates_with_scores):
                rank = rank_idx + 1
                p = candidate.get("profile", {})
                composite = scores.get("composite", 0)
                
                title = p.get("current_title", "?")
                company = p.get("current_company", "?")
                yoe = p.get("years_of_experience", 0)
                
                label = f"#{rank} — {title} at {company} ({yoe:.0f}yr) | Score: {composite:.4f}"
                if scores.get("honeypot"):
                    label += " 🚫"
                elif scores.get("hidden_gem"):
                    label += " 💎"
                
                with st.container(border=True):
                    st.markdown(f"### {label}")
                    show_candidate_evaluation(candidate, scores, rank)
        else:
            # For large samples, show table first, then click to expand
            rows = []
            for rank_idx, (c, s) in enumerate(candidates_with_scores[:100]):
                p = c.get("profile", {})
                rows.append({
                    "Rank": rank_idx + 1,
                    "Title": p.get("current_title"),
                    "Company": p.get("current_company"),
                    "YoE": p.get("years_of_experience"),
                    "Score": round(s.get("composite", 0), 4),
                    "Career": round(s.get("career_substance", 0), 3),
                    "Skills": round(s.get("skill_credibility", 0), 3),
                    "Status": "HONEYPOT" if s.get("honeypot") else ("GEM" if s.get("hidden_gem") else ""),
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            # Expand top 5
            st.subheader("Top 5 Full Evaluations")
            for rank_idx, (candidate, scores) in enumerate(candidates_with_scores[:5]):
                rank = rank_idx + 1
                p = candidate.get("profile", {})
                with st.container(border=True):
                    st.markdown(f"### #{rank} — {p.get('current_title')} at {p.get('current_company')}")
                    show_candidate_evaluation(candidate, scores, rank)
    
    with tab_cohort:
        show_cohort_analysis(candidates_with_scores)


if __name__ == "__main__":
    main()
