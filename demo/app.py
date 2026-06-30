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
import altair as alt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rank.scoring import compute_composite_score
try:
    from rank.jd_parser import validate_jd, parse_jd, get_default_config as get_default_jd
except ImportError:
    validate_jd = None
    parse_jd = None
    get_default_jd = None
from rank.reasoning import generate_reasoning
from rank.prove_it import run_prove_it, prove_it_summary
from rank.interview_blueprint import generate_interview_blueprint
from rank.hm_translator import generate_hm_briefs
from rank.cohort import jd_coverage_report, shared_gaps, pairwise_tradeoffs, anti_bias_audit

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

def score_bar(label: str, value: float, caption: str = "") -> str:
    color = score_color(value)
    pct = int(value * 100)
    caption_html = f'<div style="font-size:11px; color:#94a3b8; margin-top:2.5px; font-style:italic;">{caption}</div>' if caption else ""
    return f"""
    <div class="score-bar-container">
        <div class="score-label">{label}: <strong style="color:{color}">{pct}%</strong></div>
        <div style="height:6px; background:#1e293b; border-radius:3px; overflow:hidden;">
            <div style="height:100%; width:{pct}%; background:{color}; border-radius:3px; transition:width 0.4s;"></div>
        </div>
        {caption_html}
    </div>
    """

def generate_score_driver_captions(candidate: dict, scores: dict) -> dict:
    p = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    # 1. Career Substance
    ai_ml_titles = 0
    prod_companies = 0
    total_months = 0
    STRONG_TITLE_SIGNALS = {
        "ml engineer", "machine learning engineer", "ai engineer", "applied ml",
        "applied ai", "nlp engineer", "search engineer", "ranking engineer",
        "recommendation", "data scientist", "research engineer",
        "retrieval", "applied scientist", "founding ml", "founding ai",
    }
    for ch in career:
        title = (ch.get("title", "") or "").lower()
        company = (ch.get("company", "") or "").lower()
        company_size = ch.get("company_size", "")
        duration = int(ch.get("duration_months", 0) or 0)
        total_months += duration
        if any(kw in title for kw in STRONG_TITLE_SIGNALS):
            ai_ml_titles += 1
        is_consulting = any(cf in company for cf in ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"])
        if not is_consulting and company_size:
            prod_companies += 1
            
    yoe_years = total_months / 12.0
    career_sub = scores.get("career_substance", 0)
    if career_sub < 0.08:
        career_why = "No production AI/ML work history; non-matching background."
    elif career_sub < 0.30:
        career_why = "Some software engineering background; lacks direct production ML titles."
    else:
        career_why = f"{ai_ml_titles} AI/ML-titled roles across {yoe_years:.1f}yr exp at product companies."

    # 2. Skill Credibility
    top_skills = sorted(skills, key=lambda s: s.get("endorsements", 0), reverse=True)
    corroborated_skills = []
    unverified_skills = []
    career_text = " ".join([ch.get("title", "") + " " + ch.get("description", "") for ch in career]).lower()
    for sk in top_skills[:5]:
        sname = sk.get("name", "")
        if any(part in career_text for part in sname.lower().split()):
            corroborated_skills.append(sname)
        else:
            unverified_skills.append(sname)
            
    corr_str = ", ".join(corroborated_skills[:2])
    unv_str = ", ".join(unverified_skills[:2])
    if corr_str and unv_str:
        skills_why = f"{corr_str} corroborated by career; {unv_str} unverified."
    elif corr_str:
        skills_why = f"{corr_str} corroborated by career history."
    elif unv_str:
        skills_why = f"{unv_str} listed but unverified by career history."
    else:
        skills_why = "No relevant AI/ML skills matched or corroborated."

    # 3. Behavioral Availability
    days = _days_since(rs.get("last_active_date", "2020-01-01"))
    rrr = rs.get("recruiter_response_rate", 0)
    notice = rs.get("notice_period_days", "?")
    avail_why = f"Last active {days} days ago · {rrr:.0%} response rate · {notice}d notice."

    # 4. Experience Quality
    yoe = p.get("years_of_experience", 0)
    unique_companies = len(set(ch.get("company", "").lower() for ch in career))
    job_density_str = "stable career tenure" if (yoe > 0 and unique_companies / yoe <= 0.5) else "moderate job transitions"
    if yoe < 3:
        exp_why = f"Junior stage ({yoe:.0f}yr exp); lacks leadership exposure."
    elif 5 <= yoe <= 9:
        exp_why = f"Sweet-spot experience ({yoe:.0f}yr exp) with {job_density_str}."
    else:
        exp_why = f"{yoe:.0f}yr exp with {job_density_str}."

    # 5. Star Predictor
    star_score = scores.get("star_predictor", 0)
    if star_score >= 0.75:
        star_why = "Exceptional upward trajectory with clear progression."
    elif star_score >= 0.5:
        star_why = "Healthy career growth with consistent scope expansion."
    else:
        star_why = "Flat title growth or short stint durations."

    # 6. Location Fit
    loc = p.get("location", "").lower()
    if any(city in loc for city in ["pune", "noida"]):
        loc_why = "Located in preferred hubs (Pune/Noida)."
    elif any(city in loc for city in ["delhi", "new delhi", "ncr", "gurgaon", "gurugram", "hyderabad", "bengaluru", "bangalore", "mumbai"]):
        loc_why = "Located in acceptable Tier-1 hub."
    else:
        loc_why = "Out of target locations (relocation needed)."

    return {
        "career_substance": career_why,
        "skill_credibility": skills_why,
        "behavioral_availability": avail_why,
        "experience_quality": exp_why,
        "star_predictor": star_why,
        "location": loc_why,
    }


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
    name = p.get("anonymized_name", "Anonymous")
    cand_id = candidate.get("candidate_id", "Unknown ID")
    
    # Expected Salary
    sal = rs.get("expected_salary_range_inr_lpa", {})
    sal_str = f"₹{sal.get('min', '?')}–{sal.get('max', '?')} LPA" if sal else "Not disclosed"
    
    # Notice Period
    notice = rs.get("notice_period_days", "?")
    
    # Open to work status
    otw_val = rs.get("open_to_work_flag")
    otw_str = "🟢 Open to Work" if otw_val else "⚪ Passive"
    
    # Preferred work mode
    mode = rs.get("preferred_work_mode", "Not specified").capitalize()
    mode_icon = "🏢" if "onsite" in mode.lower() else ("🏠" if "remote" in mode.lower() else "🤝")
    
    verdict = derive_verdict(scores)
    col_rank, col_info, col_verdict = st.columns([1, 5, 2])
    with col_rank:
        st.markdown(f'<div class="rank-badge">#{rank}</div>', unsafe_allow_html=True)
    with col_info:
        st.markdown(f"### **{name}** &nbsp;`{cand_id}`", unsafe_allow_html=True)
        st.markdown(f"**{title}** at **{company}**")
        st.markdown(f"*{headline}*")
        st.markdown(
            f"📍 {location} &nbsp;|&nbsp; "
            f"⏳ {yoe:.0f}yr exp &nbsp;|&nbsp; "
            f"🗓 {notice}d notice &nbsp;|&nbsp; "
            f"💰 {sal_str} &nbsp;|&nbsp; "
            f"{mode_icon} {mode} &nbsp;|&nbsp; "
            f"{otw_str}",
            unsafe_allow_html=True
        )
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
    
    # Profile summary (UX-03)
    summary = p.get("summary", "")
    if summary:
        st.markdown("**Profile Summary / Pitch**")
        st.markdown(f"> {summary}")
        st.divider()
        
    # Score bars with explanations (UX-10)
    captions = generate_score_driver_captions(candidate, scores)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Score Breakdown** (6 dimensions)")
        st.markdown(score_bar("Career Substance (40%)", scores.get("career_substance", 0), captions.get("career_substance", "")), unsafe_allow_html=True)
        st.markdown(score_bar("Skill Credibility (22%)", scores.get("skill_credibility", 0), captions.get("skill_credibility", "")), unsafe_allow_html=True)
        st.markdown(score_bar("Behavioral Availability (15%)", scores.get("behavioral_availability", 0), captions.get("behavioral_availability", "")), unsafe_allow_html=True)
    with col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(score_bar("Experience Quality (11%)", scores.get("experience_quality", 0), captions.get("experience_quality", "")), unsafe_allow_html=True)
        st.markdown(score_bar("Star Predictor (7%)", scores.get("star_predictor", 0), captions.get("star_predictor", "")), unsafe_allow_html=True)
        st.markdown(score_bar("Location Fit (5%)", scores.get("location", 0), captions.get("location", "")), unsafe_allow_html=True)
    st.markdown(f"<br><b style='font-size:20px;color:{score_color(scores['composite'])}'>Composite: {scores['composite']:.3f}</b>", unsafe_allow_html=True)
    
    st.divider()
    
    # Tabs for the 6 evaluation panels
    tabs = st.tabs(["Shadow Recruiter", "Prove It Engine", "Deception Detector", 
                    "Interview Blueprint", "Hiring Manager Briefs", "Signals"])
    
    # ── Shadow Recruiter ──
    with tabs[0]:
        reasoning = generate_reasoning(candidate, scores, rank)
        st.markdown(f'<div class="shadow-recruiter-box">{reasoning}</div>', unsafe_allow_html=True)
        
        # UX-08: Top Skills
        st.markdown("### Top Skills")
        if skills:
            cols_skills = st.columns(2)
            top_skills = sorted(skills, key=lambda s: s.get("endorsements", 0), reverse=True)[:10]
            for idx, sk in enumerate(top_skills):
                prof = sk.get("proficiency", "").capitalize()
                end = sk.get("endorsements", 0)
                dur = sk.get("duration_months", 0)
                dur_str = f" · {dur}mo" if dur else ""
                col_target = cols_skills[idx % 2]
                col_target.markdown(f"• **{sk.get('name')}** — {prof} · {end} endorsements{dur_str}")
        else:
            st.write("No skills listed.")
        st.divider()
        
        # UX-09: Education
        st.markdown("### Education")
        education = candidate.get("education", [])
        if education:
            for edu in education:
                tier = edu.get("tier", "")
                tier_label = {"tier_1": "🏆 Tier 1", "tier_2": "Tier 2", "tier_3": "Tier 3", "tier_4": "Tier 4"}.get(tier, "")
                tier_str = f" ({tier_label})" if tier_label else ""
                st.markdown(f"🎓 **{edu.get('degree', '')} in {edu.get('field_of_study', '')}** — {edu.get('institution', '')}{tier_str} ({edu.get('start_year', '?')}–{edu.get('end_year', '?')})")
        else:
            st.write("No education listed.")
        st.divider()
        
        # UX-14: Certifications
        st.markdown("### Certifications")
        certs = candidate.get("certifications", [])
        if certs:
            for cert in certs:
                authority = cert.get("authority", "")
                auth_str = f" by {authority}" if authority else ""
                year = cert.get("year", "")
                year_str = f" ({year})" if year else ""
                st.markdown(f"• **{cert.get('name')}**{auth_str}{year_str}")
        else:
            st.write("No certifications listed.")
        st.divider()
        
        # Career history
        st.markdown("### Career History")
        for ch in career:
            st.markdown(f"**{ch.get('title')} at {ch.get('company')}** — {ch.get('duration_months')}mo")
            desc = ch.get("description", "No description")
            st.caption(desc[:300] + ("…" if len(desc) > 300 else ""))
            st.divider()
    
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
            gh = rs.get("github_activity_score", -1)
            st.metric("GitHub Score", "Not linked" if gh == -1 else str(gh))
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
        st.subheader("Score Distribution")
        bin_edges = [round(i * 0.05, 2) for i in range(21)]
        labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        
        df_hist = pd.DataFrame({"score": composites})
        df_hist["bucket"] = pd.cut(
            df_hist["score"],
            bins=bin_edges,
            labels=labels,
            include_lowest=True
        )
        
        dist = df_hist.groupby("bucket", observed=True).size().reset_index()
        dist.columns = ["Score Range", "Count"]
        dist["Score Range"] = dist["Score Range"].astype(str)
        
        chart = alt.Chart(dist).mark_bar(color="#4C9BE8").encode(
            x=alt.X("Score Range:N", sort=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Count:Q"),
            tooltip=["Score Range", "Count"]
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("Note: The spike at 0.00–0.15 represents candidates caught by the Keyword Stuffer Gate (hard-capped at 0.15).")
    
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
            "Score": f'{s.get("composite", 0):.3f}',
            "Career": round(s.get("career_substance", 0), 3),
            "Skills": round(s.get("skill_credibility", 0), 3),
            "Notice (days)": rs.get("notice_period_days"),
            "Days Inactive": int(days) if pd.notna(days) and days != "" else "—",
            "Gem": "💎 GEM" if s.get("hidden_gem") else "",
            "Deception": {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High", "n/a": "⚪ N/A"}.get(s.get("deception_risk", "n/a"), "⚪ N/A"),
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
    
    st.divider()
    
    # ── Advanced Cohort Panels (from cohort.py) ────────────────────────────────
    st.subheader("Advanced Cohort Intelligence")
    
    # Build top-10 list for cohort functions
    sorted_non_hp = [(c, s, i+1) for i, (c, s) in enumerate(
        sorted(candidates_with_scores, key=lambda x: -x[1].get("composite", 0))
    ) if not s.get("honeypot")][:10]
    
    if len(sorted_non_hp) < 2:
        st.info("Load at least 2 valid candidates to see advanced cohort analysis.")
        return
    
    tab_cov, tab_gaps, tab_pairs, tab_bias = st.tabs([
        "📋 JD Coverage", "🔍 Shared Gaps", "⚖️ Pairwise Tradeoffs", "🔎 Anti-Bias Audit"
    ])
    
    # ── JD Coverage ──────────────────────────────────────────────────────────
    with tab_cov:
        st.markdown("**Which JD requirements does the top-10 shortlist actually satisfy?**")
        try:
            cov_data = jd_coverage_report(sorted_non_hp)
            for req in cov_data.get("jd_coverage", []):
                pct = req.get("top10_coverage_pct", 0)
                label = req.get("label", "")
                weight = req.get("weight", "")
                weight_tag = "🔴 MUST HAVE" if weight == "must_have" else "🟡 Strong Positive"
                color = "#34d399" if pct >= 70 else ("#fbbf24" if pct >= 40 else "#f87171")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #30363d'>"
                    f"<span>{weight_tag} &nbsp;<b>{label}</b></span>"
                    f"<span style='color:{color};font-weight:700'>{pct:.0f}% coverage</span></div>",
                    unsafe_allow_html=True
                )
            st.caption("Coverage = % of top-10 candidates who satisfy this JD requirement. < 40% = probe in screening.")
        except Exception as e:
            st.error(f"JD Coverage error: {e}")
    
    # ── Shared Gaps ───────────────────────────────────────────────────────────
    with tab_gaps:
        st.markdown("**Requirements that most top-10 candidates don't satisfy — pool-wide gaps, not individual weakness.**")
        try:
            gaps_data = shared_gaps(sorted_non_hp)
            gap_list = gaps_data.get("shared_gaps", [])
            if not gap_list:
                st.success("No critical shared gaps — the top-10 collectively cover all JD requirements.")
            else:
                for gap in gap_list:
                    sev = gap.get("severity", "")
                    sev_color = "#f87171" if sev == "critical" else "#fbbf24"
                    n_missing = gap.get("candidates_missing", 0)
                    st.markdown(
                        f"<div style='background:#1e293b;border-left:3px solid {sev_color};padding:10px 14px;border-radius:0 8px 8px 0;margin:8px 0'>"
                        f"<b style='color:{sev_color}'>{sev.upper()}</b> &nbsp;·&nbsp; "
                        f"<b>{gap.get('requirement', '')}</b><br>"
                        f"<span style='color:#94a3b8'>{gap.get('note', '')} ({n_missing}/10 missing)</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Shared Gaps error: {e}")
    
    # ── Pairwise Tradeoffs ────────────────────────────────────────────────────
    with tab_pairs:
        st.markdown("**Direct comparisons between adjacent-ranked candidates with numeric score evidence.**")
        try:
            pairs_data = pairwise_tradeoffs(sorted_non_hp[:6])
            for pair in pairs_data.get("pairwise_tradeoffs", []):
                ra, rb = pair.get("rank_a"), pair.get("rank_b")
                ta = pair.get("title_a", "?")
                tb = pair.get("title_b", "?")
                sa = pair.get("score_a", 0)
                sb = pair.get("score_b", 0)
                with st.expander(f"Rank {ra} vs Rank {rb} — {ta} ({sa:.3f}) vs {tb} ({sb:.3f})"):
                    for diff in pair.get("key_differences", []):
                        st.markdown(f"- {diff}")
                    rec = pair.get("recommendation", "")
                    if rec:
                        st.info(f"**Recommendation:** {rec}")
        except Exception as e:
            st.error(f"Pairwise Tradeoffs error: {e}")
    
# ── Anti-Bias Audit ───────────────────────────────────────────────────────
    with tab_bias:
        st.markdown("**Structural diversity check across geographic spread and YoE range in the top-10.**")
        try:
            bias_data = anti_bias_audit(sorted_non_hp)
            audit = bias_data.get("anti_bias_audit", {})
            flags = audit.get("flags", [])
            insights = audit.get("insights", [])
            note = audit.get("note", "")
            
            if flags:
                for flag in flags:
                    st.warning(flag)
            else:
                st.success("No homogeneity flags — the top-10 shows healthy structural diversity.")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            if note:
                st.caption(note)
        except Exception as e:
            st.error(f"Anti-Bias Audit error: {e}")


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Resolve active JD config
    if "jd_config" not in st.session_state:
        st.session_state.jd_config = get_default_jd() if get_default_jd else None
    # Track whether user has EXPLICITLY selected a JD (vs relying on the silent default)
    if "jd_explicitly_set" not in st.session_state:
        st.session_state.jd_explicitly_set = False
    jd_cfg = st.session_state.jd_config
    jd_title = (jd_cfg or {}).get("title", "Senior AI Engineer — Founding Team")
    jd_company = (jd_cfg or {}).get("company_name", "")
    tagline = f"**{jd_title}**" + (f" @ {jd_company}" if jd_company else "")
    tagline += " | Intelligent Candidate Discovery System"

    st.title("Redrob AI Candidate Ranker")
    st.markdown(tagline)
    st.divider()
    
    # Sidebar
    with st.sidebar:
        # ── Section 1: Job Description ───────────────────────────────────
        st.header("📋 Job Description")
        st.markdown("Upload a custom JD **or** try the sample Redrob JD.")

        uploaded_jd = st.file_uploader("Upload JD (JSON)", type=["json"], key="jd_upload")

        col_jd1, col_jd2 = st.columns(2)
        with col_jd1:
            load_sample_jd = st.button("📌 Load Sample JD", help="Load the Redrob Senior AI Engineer JD", use_container_width=True)
        with col_jd2:
            reset_jd = st.button("↩️ Reset JD", help="Reset to default JD config", use_container_width=True)

        with st.expander("📐 JD Schema — required format"):
            st.markdown("""
**Required fields (minimum):**
```json
{
  "title": "Your Job Title",
  "required_skills": ["python", "pytorch", "skill3"],
  "preferred_locations": ["bangalore", "mumbai"],
  "experience_years": { "min": 3, "max": 8 }
}
```
**All optional fields:**
| Field | Type | Description |
|-------|------|-------------|
| `company_name` | string | Shown in UI header |
| `positive_title_tokens` | string[ ] | Titles that confirm fit |
| `hard_negative_title_tokens` | string[ ] | Titles that disqualify |
| `acceptable_locations` | string[ ] | Tier-2 preferred cities |
| `experience_years.ideal_min/max` | number | Sweet-spot band |
| `salary_range_inr_lpa` | `{min, max}` | Budget in LPA |
| `avoid_consulting_only` | boolean | Penalise pure-consulting careers |
| `cv_specialty_penalty_keywords` | string[ ] | Off-domain keywords to penalise |

[📄 Download full schema (jd_schema.json)](https://github.com/GaganHR2006/redrob-ranker/blob/main/jd_schema.json)
            """)

        # Handle JD file upload
        if uploaded_jd:
            try:
                jd_raw = json.loads(uploaded_jd.read().decode("utf-8"))
                errors = validate_jd(jd_raw) if validate_jd else []
                if errors:
                    for e in errors:
                        st.error(f"❌ {e}")
                else:
                    st.session_state.jd_config = parse_jd(jd_raw) if parse_jd else None
                    st.session_state.jd_explicitly_set = True
                    jd_cfg = st.session_state.jd_config
                    st.success(f"✅ JD loaded: **{jd_raw.get('title', 'Custom Role')}**")
                    st.session_state.candidates_data = None  # re-score with new JD
            except Exception as ex:
                st.error(f"Failed to parse JD JSON: {ex}")

        # Handle Sample JD button
        if load_sample_jd:
            sample_jd_paths = [
                Path(__file__).parent.parent / "sample_jd.json",
                Path(__file__).parent / "sample_jd.json",
            ]
            sample_jd_path = next((p for p in sample_jd_paths if p.exists()), None)
            if sample_jd_path:
                with open(sample_jd_path, encoding="utf-8") as f:
                    jd_raw = json.load(f)
                st.session_state.jd_config = parse_jd(jd_raw) if parse_jd else get_default_jd()
                st.session_state.jd_explicitly_set = True
                jd_cfg = st.session_state.jd_config
                st.success("✅ Sample JD loaded: **Senior AI Engineer — Founding Team (Redrob)**")
                st.session_state.candidates_data = None
            else:
                # Fallback: use the built-in default
                st.session_state.jd_config = get_default_jd() if get_default_jd else None
                st.session_state.jd_explicitly_set = True
                jd_cfg = st.session_state.jd_config
                st.success("✅ Default Redrob JD loaded")
                st.session_state.candidates_data = None
            st.rerun()

        # Handle Reset JD button
        if reset_jd:
            st.session_state.jd_config = get_default_jd() if get_default_jd else None
            st.session_state.candidates_data = None
            st.rerun()

        # Show active JD indicator
        active_title = (st.session_state.get("jd_config") or {}).get("title", "Redrob Default")
        st.caption(f"🟢 Active JD: **{active_title}**")

        st.divider()

        # ── Section 2: Candidates ─────────────────────────────────────────
        st.header("👥 Upload Candidates")
        st.markdown("Upload your own file **or** try without uploading:")

        with st.expander("📐 Candidates Schema — required format"):
            st.markdown("""
**Required top-level fields:**
```json
{
  "candidate_id": "CAND_0000001",
  "profile":        { ... },
  "career_history": [ ... ],
  "education":      [ ... ],
  "skills":         [ ... ],
  "redrob_signals": { ... },
  "certifications": [ ... ],  // optional
  "languages":      [ ... ]   // optional
}
```
**`profile` required fields:**
```json
{
  "anonymized_name": "Candidate A",
  "headline": "ML Engineer at Startup",
  "summary": "5+ years building recommenders...",
  "location": "Pune, Maharashtra",
  "country": "India",
  "years_of_experience": 6,
  "current_title": "Senior ML Engineer",
  "current_company": "Acme AI",
  "current_company_size": "51-200",
  "current_industry": "Technology"
}
// company_size enum: 1-10|11-50|51-200|201-500|501-1000|1001-5000|5001-10000|10001+
```
**`career_history` item required fields:**
```json
{
  "company": "Acme AI",
  "title": "ML Engineer",
  "start_date": "2021-06-01",
  "end_date": "2024-01-15",   // null if is_current=true
  "duration_months": 31,
  "is_current": false,
  "industry": "Technology",
  "company_size": "51-200",
  "description": "Built recommendation pipeline..."
}
```
**`skills` item required fields:**
```json
{ "name": "PyTorch", "proficiency": "expert",
  "endorsements": 25, "duration_months": 36 }
// proficiency enum: beginner|intermediate|advanced|expert
```
**`education` item required fields:**
```json
{ "institution": "IIT Bombay", "degree": "B.Tech",
  "field_of_study": "Computer Science",
  "start_year": 2016, "end_year": 2020 }
// optional: grade, tier (tier_1|tier_2|tier_3|tier_4|unknown)
```
**`redrob_signals` required fields (all 23):**
```json
{
  "profile_completeness_score": 87,
  "signup_date": "2023-01-15",
  "last_active_date": "2024-05-20",
  "open_to_work_flag": true,
  "profile_views_received_30d": 45,
  "applications_submitted_30d": 3,
  "recruiter_response_rate": 0.85,
  "avg_response_time_hours": 4.5,
  "skill_assessment_scores": {"Python": 88},
  "connection_count": 312,
  "endorsements_received": 120,
  "notice_period_days": 30,
  "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
  "preferred_work_mode": "hybrid",
  "willing_to_relocate": true,
  "github_activity_score": 72,
  "search_appearance_30d": 89,
  "saved_by_recruiters_30d": 5,
  "interview_completion_rate": 0.9,
  "offer_acceptance_rate": 0.75,
  "verified_email": true,
  "verified_phone": true,
  "linkedin_connected": true
}
// preferred_work_mode enum: remote|hybrid|onsite|flexible
// github_activity_score: 0-100 or -1 if no GitHub linked
// offer_acceptance_rate: 0.0-1.0 or -1 if no offer history
```
**File format:** `.json` (array) or `.jsonl` (one object per line)
            """)

        uploaded = st.file_uploader("Choose JSON/JSONL file", type=["json", "jsonl"])

        st.divider()
        st.markdown("**Try without uploading:**")
        use_top100 = st.button("🏆 Load Top 100 Ranked Candidates", type="primary",
                               use_container_width=True,
                               help="Load pre-ranked Top 100 from the 100K Redrob dataset")
        use_all = st.button("📂 Load All 100K Candidates (~60s)", use_container_width=True,
                            help="Stream and score all 465 MB candidates.jsonl locally")

        st.divider()
        st.markdown("**About this system:**")
        st.markdown("""
        - 6-dimension composite scoring
        - Honeypot detection (impossible profiles)
        - Keyword-stuffer detection
        - Hidden gem surfacing
        - Star Predictor (career arc velocity)
        - No API keys required
        - Runs in <60s on 100K candidates
        - Supports custom JD upload ↑
        """)
    
    # Load candidates
    if "candidates_data" not in st.session_state:
        st.session_state.candidates_data = None
    
    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                # Try parsing as JSON Lines
                data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        data.append(json.loads(line))
            st.session_state.candidates_data = data
            st.sidebar.success(f"Loaded {len(data)} candidates")
        except Exception as e:
            st.error(f"Error parsing JSON/JSONL: {e}")
            return

    elif use_top100:
        # Look for top_100_candidates.json in multiple locations
        search_paths = [
            Path(__file__).parent.parent / "data" / "top_100_candidates.json",
            Path(__file__).parent.parent / "top_100_candidates.json",
            Path(__file__).parent.parent / "outputs" / "top_100_candidates.json",
            Path(__file__).parent / "top_100_candidates.json",
        ]
        top100_path = next((p for p in search_paths if p.exists()), None)
        if top100_path:
            with open(top100_path, encoding="utf-8") as f:
                st.session_state.candidates_data = json.load(f)
            st.sidebar.success(f"Loaded {len(st.session_state.candidates_data)} top-ranked candidates")
        else:
            # Fall back: try outputs/submission.csv and parse it
            csv_path = Path(__file__).parent.parent / "outputs" / "submission.csv"
            if csv_path.exists():
                import csv as csv_mod
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv_mod.DictReader(f)
                    rows = list(reader)
                data = []
                for row in rows:
                    data.append({
                        "candidate_id": row.get("candidate_id", ""),
                        "profile": {
                            "current_title": row.get("current_title", ""),
                            "location": row.get("location", ""),
                            "years_of_experience": row.get("years_of_experience", 0),
                        },
                        "_from_csv": True,
                        "_score": float(row.get("score", 0) or 0),
                        "_reasoning": row.get("reasoning", ""),
                    })
                st.session_state.candidates_data = data
                st.sidebar.success(f"Loaded {len(data)} candidates from submission CSV")
            else:
                st.error("No candidate data found. Please upload a JSON/JSONL file using the uploader above.")
                return

    elif use_all:
        # Stream the full 465 MB candidates.jsonl from common local paths
        all_candidates_paths = [
            Path(r"C:/Users/gagan/Downloads/redrob_extracted/candidates.jsonl"),
            Path(__file__).parent.parent / "data" / "candidates.jsonl",
            Path(__file__).parent.parent / "candidates.jsonl",
        ]
        all_path = next((p for p in all_candidates_paths if p.exists()), None)
        if all_path:
            with st.spinner(f"Streaming all candidates from {all_path.name} — this may take ~60 seconds..."):
                data = []
                with open(all_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            st.session_state.candidates_data = data
            st.sidebar.success(f"✅ Loaded {len(data):,} candidates from {all_path.name}")
        else:
            st.error(
                "❌ Could not find candidates.jsonl locally.\n\n"
                "Please place your `candidates.jsonl` file in one of these locations:\n"
                "- `redrob-ranker/data/candidates.jsonl`\n"
                "- `redrob-ranker/candidates.jsonl`\n\n"
                "Or use the file uploader above."
            )
            return
    
    candidates_data = st.session_state.candidates_data
    jd_cfg = st.session_state.jd_config  # re-read in case it changed

    # ⚠️ JD transparency banner — show when candidates are loaded but no JD was explicitly chosen
    if candidates_data is not None and not st.session_state.get("jd_explicitly_set", False):
        active_jd_title = (jd_cfg or {}).get("title", "Senior AI Engineer — Founding Team")
        st.warning(
            f"⚠️ **No JD selected** — ranking candidates using the default JD: "
            f"*{active_jd_title}*.\n\n"
            "To rank for a different role, click **📄 Load Sample JD** or "
            "**Upload JD (JSON)** in the 📋 Job Description section of the sidebar."
        )

    if candidates_data is None:
        st.info(
            "Choose one of:\n"
            "1. 📋 **Upload your JD** (or Load Sample JD) in the sidebar\n"
            "2. 👥 **Upload candidates** or click **Load Top 100** / **Load All 100K**"
        )
        st.markdown("""
        ### What this system evaluates

        | Component | Weight | Description |
        |-----------|--------|-------------|
        | **Career Substance** | 40% | Did this person actually build AI/ML/search systems? Reads job titles + headline |
        | **Skill Credibility** | 22% | Are skills corroborated by endorsements, duration, platform assessments? |
        | **Experience Quality** | 18% | Right YoE range, product company, not pure research/consulting |
        | **Behavioral Availability** | 15% | Last active date, response rate, notice period, open-to-work flag |
        | **Star Predictor** | 7% | Career growth arc — learning velocity across roles |
        | **Location Fit** | 5% | JD preferred cities vs. candidate location |

        **Anti-trap mechanisms:**
        - Keyword stuffers (Marketing Manager with AI skills) → capped at 0.15
        - Honeypots (impossible timelines or duplicate profiles) → zeroed
        - Behavioral ghosts (inactive 180d + 5% response rate) → heavily down-weighted
        - Hidden gems (plain-language engineers without buzzwords) → +0.05 bonus
        - Consulting-only careers → 55% penalty multiplier
        """)
        return
    
    # Score all candidates
    with st.spinner(f"Scoring {len(candidates_data)} candidates..."):
        candidates_with_scores = []
        for c in candidates_data:
            s = compute_composite_score(c, jd_cfg)
            candidates_with_scores.append((c, s))
        
        # Sort by score
        candidates_with_scores.sort(key=lambda x: (-x[1].get("composite", 0), x[0].get("candidate_id", "")))
    
    # Tabs
    tab_explorer, tab_cohort = st.tabs(["Candidate Explorer", "Cohort Analysis"])
    
    with tab_explorer:
        st.header("Ranked Candidates")
        
        # UX-13: Actionable Output / Export - Download all ranked candidates as CSV
        all_export_rows = []
        for rank_idx, (c, s) in enumerate(candidates_with_scores):
            p_e = c.get("profile", {})
            rs_e = c.get("redrob_signals", {})
            sal_e = rs_e.get("expected_salary_range_inr_lpa", {})
            sal_str_e = f"₹{sal_e.get('min', '?')}–{sal_e.get('max', '?')} LPA" if sal_e else "N/D"
            all_export_rows.append({
                "Rank": rank_idx + 1,
                "Candidate ID": c.get("candidate_id"),
                "Name": p_e.get("anonymized_name", "Anonymous"),
                "Title": p_e.get("current_title"),
                "Company": p_e.get("current_company"),
                "YoE": p_e.get("years_of_experience"),
                "Location": p_e.get("location"),
                "Composite Score": f"{s.get('composite', 0):.3f}",
                "Expected Salary": sal_str_e,
                "Notice Period (days)": rs_e.get("notice_period_days"),
                "Open to Work": "Yes" if rs_e.get("open_to_work") else "No",
                "Days Inactive": rs_e.get("days_inactive", ""),
                "Deception Risk": s.get("deception_risk", "N/A").capitalize() if s.get("deception_risk") else "N/A",
                "Hidden Gem": "Yes" if s.get("hidden_gem") else "No",
                "Honeypot": "Yes" if s.get("honeypot") else "No",
                "Career Score": f"{s.get('career_substance', 0):.3f}",
                "Skill Score": f"{s.get('skill_credibility', 0):.3f}",
                "Experience Score": f"{s.get('experience_quality', 0):.3f}",
                "Availability Score": f"{s.get('behavioral_availability', 0):.3f}",
                "Star Score": f"{s.get('star_predictor', 0):.3f}",
            })
        df_all_export = pd.DataFrame(all_export_rows)
        
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇️ Export All Ranked Candidates (CSV)",
                data=df_all_export.to_csv(index=False).encode('utf-8-sig'),
                file_name="all_ranked_candidates.csv",
                mime="text/csv",
            )
        st.write("")

        # UX-07: Side-by-side comparison summary table above Top 5
        st.subheader("Top 5 Candidate Side-by-Side Comparison")
        comp_rows = []
        for rank_idx, (c, s) in enumerate(candidates_with_scores[:5]):
            p_c = c.get("profile", {})
            rs_c = c.get("redrob_signals", {})
            name_c = p_c.get("anonymized_name", "Anonymous")
            title_c = p_c.get("current_title", "?")
            company_c = p_c.get("current_company", "?")
            sal_c = rs_c.get("expected_salary_range_inr_lpa", {})
            sal_str_c = f"₹{sal_c.get('min', '?')}–{sal_c.get('max', '?')} LPA" if sal_c else "N/D"
            comp_rows.append({
                "Rank": f"#{rank_idx + 1}",
                "Candidate ID": c.get("candidate_id"),
                "Name": name_c,
                "Current Role": f"{title_c} @ {company_c}",
                "Score": f"{s.get('composite', 0):.3f}",
                "Location": p_c.get("location", "?"),
                "YoE": f"{p_c.get('years_of_experience', 0):.0f}yr",
                "Notice": f"{rs_c.get('notice_period_days', '?')}d",
                "Salary Expectation": sal_str_c,
                "Open to Work": "✅" if rs_c.get("open_to_work_flag") else "❌",
                "Deception": {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High", "n/a": "⚪ N/A"}.get(s.get("deception_risk", "n/a"), "⚪ N/A"),
            })
        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        st.write("")

        if len(candidates_data) <= 20:
            # For small samples, show all with full evaluation expandable
            st.subheader("Candidate Evaluations")
            for rank_idx, (candidate, scores) in enumerate(candidates_with_scores):
                rank = rank_idx + 1
                p = candidate.get("profile", {})
                rs = candidate.get("redrob_signals", {})
                
                title = p.get("current_title", "?")
                company = p.get("current_company", "?")
                composite = scores.get("composite", 0)
                location = p.get("location", "?")
                yoe = p.get("years_of_experience", 0)
                notice = rs.get("notice_period_days", "?")
                cand_id = candidate.get("candidate_id", "")
                
                # UX-04: Enriched expander label
                label = (
                    f"#{rank} — {title} @ {company} ({cand_id})  |  "
                    f"Score: {composite:.3f}  |  {location}  |  "
                    f"{yoe:.0f}yr exp  |  {notice}d notice"
                )
                if scores.get("honeypot"):
                    label += "  🚫 HONEYPOT"
                elif scores.get("hidden_gem"):
                    label += "  💎 GEM"
                # UX-12: Deception warning on collapsed expander
                if scores.get("deception_risk") == "high":
                    label += "  🔴 HIGH RISK"
                
                with st.container(border=True):
                    col_lbl, col_chk = st.columns([8, 2])
                    col_lbl.markdown(f"**{label}**")
                    show_details = col_chk.checkbox("👁 View Details", key=f"show_sm_{cand_id}_{rank}")
                    if show_details:
                        show_candidate_evaluation(candidate, scores, rank)
        else:
            # For large samples, show table first, then click to expand
            rows = []
            for rank_idx, (c, s) in enumerate(candidates_with_scores[:100]):
                p = c.get("profile", {})
                rows.append({
                    "Rank": rank_idx + 1,
                    "Candidate ID": c.get("candidate_id"),
                    "Name": p.get("anonymized_name", "Anonymous"),
                    "Title": p.get("current_title"),
                    "Company": p.get("current_company"),
                    "YoE": p.get("years_of_experience"),
                    "Score": f"{s.get('composite', 0):.3f}",
                    "Career": round(s.get("career_substance", 0), 3),
                    "Skills": round(s.get("skill_credibility", 0), 3),
                    "Status": "HONEYPOT" if s.get("honeypot") else ("💎 GEM" if s.get("hidden_gem") else ""),
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            # Expand top 5
            st.subheader("Top 5 Full Evaluations")
            for rank_idx, (candidate, scores) in enumerate(candidates_with_scores[:5]):
                rank = rank_idx + 1
                p = candidate.get("profile", {})
                rs = candidate.get("redrob_signals", {})
                
                title = p.get("current_title", "?")
                company = p.get("current_company", "?")
                composite = scores.get("composite", 0)
                location = p.get("location", "?")
                yoe = p.get("years_of_experience", 0)
                notice = rs.get("notice_period_days", "?")
                cand_id = candidate.get("candidate_id", "")
                
                # UX-04: Enriched expander label
                label = (
                    f"#{rank} — {title} @ {company} ({cand_id})  |  "
                    f"Score: {composite:.3f}  |  {location}  |  "
                    f"{yoe:.0f}yr exp  |  {notice}d notice"
                )
                if scores.get("honeypot"):
                    label += "  🚫 HONEYPOT"
                elif scores.get("hidden_gem"):
                    label += "  💎 GEM"
                # UX-12: Deception warning on collapsed expander
                if scores.get("deception_risk") == "high":
                    label += "  🔴 HIGH RISK"
                
                with st.container(border=True):
                    col_lbl, col_chk = st.columns([8, 2])
                    col_lbl.markdown(f"**{label}**")
                    show_details = col_chk.checkbox("👁 View Details", key=f"show_lg_{cand_id}_{rank}")
                    if show_details:
                        show_candidate_evaluation(candidate, scores, rank)
    
    with tab_cohort:
        show_cohort_analysis(candidates_with_scores)


if __name__ == "__main__":
    main()
