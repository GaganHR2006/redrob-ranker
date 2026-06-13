"""
reasoning.py — Shadow Recruiter narrative engine (v3 — phrase-diverse).

Design principle: every string should read like a senior recruiter thinking out loud
while reviewing a profile for the first time. NOT a data dump. NOT a template.

Structure (5 stages, used selectively based on what's actually interesting):
  1. Opening hook      — what stands out immediately about the career arc
  2. Evidence read     — which specific signals are trustworthy and why
  3. Skeptic's eye     — one concrete concern, named specifically
  4. Availability read — behavioral signals interpreted in plain language
  5. Bottom line       — the recruiter's instinct call

v3 changes:
  - Deterministic phrase variation using candidate_id hash (reproducible, not random)
  - 5+ phrasings per concern type so no repeated phrases across candidates
  - Bottom line varies by rank tier AND candidate_id
  - Evidence read uses different sentence shapes per candidate
"""

import hashlib
from datetime import date, datetime
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC PHRASE PICKER
# ─────────────────────────────────────────────────────────────────────────────

def _pick(candidate_id: str, options: list, salt: str = "") -> str:
    """Deterministically pick a phrase from options based on candidate_id.
    The salt ensures different picks for different contexts within the same candidate."""
    key = (candidate_id + salt).encode("utf-8")
    idx = int(hashlib.md5(key).hexdigest(), 16) % len(options)
    return options[idx]


def _pick_idx(candidate_id: str, n: int, salt: str = "") -> int:
    """Return a deterministic index in [0, n) based on candidate_id."""
    key = (candidate_id + salt).encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % n


# ─────────────────────────────────────────────────────────────────────────────
# PHRASE BANKS
# ─────────────────────────────────────────────────────────────────────────────

# Evidence read: high endorsements + long duration
DEEP_SKILL_PHRASES = [
    "{name} ({prof}) — {endorse} endorsements and {dur} months usage. That kind of consistency signals production depth, not a resume keyword.",
    "{name} at {prof} level: {endorse} people endorsed it, {dur} months logged. Numbers that high only come from working with it daily.",
    "{endorse} endorsements on {name} across {dur} months — this wasn't picked up from a tutorial. Real usage, sustained over years.",
    "{name} ({prof}): {dur} months of hands-on work, backed by {endorse} endorsements. These numbers speak to genuine, sustained practice.",
    "Strongest signal: {name} — {endorse} endorsements, {dur}mo duration, {prof} proficiency. That profile correlates with someone who shipped with it repeatedly.",
    "{dur} months using {name} with {endorse} people vouching for it. At {prof} level, this is the most credible skill on the profile.",
    "{name} stands out: {endorse} endorsements, {dur}mo of usage at {prof} level. This kind of evidence doesn't come from listing buzzwords.",
    "Lead skill: {name}. {endorse} endorsements over {dur}mo at {prof} — the strongest corroboration in the whole profile.",
    "{name} is {prof}-level with {endorse} endorsements ({dur}mo). That density of social proof usually means real production exposure.",
    "Top credibility marker: {name}. {dur} months of usage, {endorse} endorsements, {prof}. Hard to fake that combination.",
    "{prof}-level {name}: {endorse} endorsements accumulated over {dur} months. The numbers indicate daily production use, not weekend projects.",
    "{endorse} people endorsed {name} ({prof}) across a {dur}-month span. In this candidate pool, that level of corroboration is notable.",
    "{name} at {prof} with {dur}mo tenure and {endorse} endorsements — one of the better-evidenced skill claims in the entire pool.",
    "The {name} signal is strong: {prof} proficiency, {endorse} endorsements, {dur}mo duration. Consistent with hands-on production work.",
]

# Evidence read: moderate endorsements
DECENT_SKILL_PHRASES = [
    "{name} with {endorse} endorsements and {dur}mo tenure. A credible signal — enough to shortlist on, verify in the technical screen.",
    "{endorse} endorsements on {name} over {dur} months. Not overwhelming, but sufficient to treat this as a real skill rather than padding.",
    "{name}: {endorse} endorsements, {dur}mo duration. The numbers clear the 'probably real' bar but don't scream depth.",
    "{name} ({dur}mo, {endorse} endorsed). Solid enough to trust the claim at face value — probe the edges in a call.",
    "Credible {name} signal: {endorse} endorsements over {dur} months. Mid-range corroboration — neither thin nor exceptional.",
]

# Evidence read: long duration but low endorsements
DURATION_ONLY_PHRASES = [
    "{name}: {dur} months of stated usage but only {endorse} endorsements. Duration suggests genuine exposure; the endorsement gap might just mean they don't self-promote.",
    "{dur}mo with {name}, endorsements are modest ({endorse}). Long tenure with the tool is a real signal even when the social proof is thin.",
    "{name} listed for {dur} months — {endorse} endorsements is low but duration alone at that length usually means production use, not just coursework.",
    "Long history with {name} ({dur}mo) but light endorsement count ({endorse}). Likely a real skill; the candidate just doesn't network for endorsements.",
    "{dur} months using {name}, only {endorse} endorsements. The duration tells the real story here — that's years of hands-on work.",
]

# Evidence read: weak skill
WEAK_SKILL_PHRASES = [
    "{name} listed as {prof} but with {endorse} endorsements and {dur}mo tenure — treat as unverified until the technical screen confirms it.",
    "{name}: claims {prof} proficiency, but {endorse} endorsements and {dur}mo don't back it up. This needs validation before trusting.",
    "{prof}-level {name} is listed but the supporting numbers ({endorse} endorsements, {dur}mo) are too thin to rely on. Needs a practical test.",
    "{name} at {prof} level looks aspirational — {endorse} endorsements and {dur}mo isn't enough to corroborate. Proceed with a proof point in the interview.",
]

# Evidence read: career-based (no strong skill found)
CAREER_STRONG_PHRASES = [
    "Career history shows sustained AI/ML focus across multiple roles — this isn't someone who dipped into ML once and listed it.",
    "The career thread is genuinely AI/ML — multiple titles confirm this is their core domain, not an adjacent interest.",
    "Multiple AI/ML-titled roles across different companies. The career itself is the best evidence of domain commitment.",
    "Career arc reads as deeply AI/ML-native — title progression stays in-domain across companies.",
]

CAREER_MODERATE_PHRASES = [
    "AI/ML appears in the career history but isn't the dominant thread. Probably competent, but depth needs validation.",
    "Some ML signal in the career trajectory, though it shares space with broader engineering work. Not a pure-play ML engineer.",
    "Career shows AI/ML exposure without full immersion. Likely has foundations but would need ramp-up time on specialized systems.",
    "ML-adjacent career — the skills are plausible from the trajectory, but this isn't a specialist profile.",
]

# ── CONCERN PHRASE BANKS ─────────────────────────────────────────────────────

TITLE_INFLATION_PHRASES = [
    "Title says '{title}' but the descriptions read like IC work — no team management language. Worth clarifying: tech lead or people manager?",
    "The '{title}' title doesn't match the description's scope — reads more like a senior IC than a manager. Ask about team size and reporting lines.",
    "'{title}' in the title, but descriptions mention building and implementing, not managing or hiring. Is the seniority real or title-inflated?",
    "Descriptions under the '{title}' role focus on individual delivery, not team leadership. The title may overstate the actual scope.",
    "'{title}' sounds senior, but the role descriptions lack management verbs (hired, mentored, grew). Could be a technical lead without direct reports.",
]

CONSULTING_HEAVY_PHRASES = [
    "Entire career at services firms ({companies}). No product experience — a real gap for a founding-team role where you build from zero.",
    "All roles at {companies} — pure consulting background. Technically capable people come from services firms, but the startup context will be completely new.",
    "Services-only career ({companies}). Product company experience is missing — they've built for clients, not for a product they own.",
    "{companies}: all consulting. The candidate hasn't experienced the product ownership mindset that a founding team demands.",
    "Career is entirely services-firm based ({companies}). Can they thrive without requirements handed down from a client? That's the open question.",
]

CONSULTING_TRANSITION_PHRASES = [
    "Started at {early_co} but moved to product — that transition usually signals ambition. Is the product experience deep enough?",
    "Consulting-to-product arc (started at {early_co}). The move shows initiative, but check whether the product role involved real ownership.",
    "Began at {early_co}, now at a product company. Good trajectory — but verify the product stint isn't just re-badged consulting.",
    "Career started in services ({early_co}) then shifted to product. The direction is right — depth of the product experience is the question.",
]

THIN_SKILLS_PHRASES = [
    "Career substance is there but the skills list lacks endorsement backing. Probably a builder who doesn't optimize their profile — verify depth in conversation.",
    "Real career history but endorsement counts on skills are low. Could be a plain-language engineer; could be shallow. Technical screen will tell.",
    "The career trajectory looks real, but skills section is thin on corroboration. Worth an extra technical question to confirm the hands-on depth.",
    "Strong career signal, weak skills documentation. This pattern usually means they do the work but don't curate their profile. Probe with specifics.",
    "Skills endorsements don't match the career strength. Either a private person or skills are more claimed than practiced. Interview will distinguish.",
]

GHOST_PHRASES = [
    "Recruiter response rate is {rrr} — expect to invest more pipeline time before getting a live conversation.",
    "Response rate at {rrr} means outreach will take persistence. Don't invest heavily until you get a reply.",
    "{rrr} recruiter response rate — this candidate doesn't respond to most outreach. Budget extra touchpoints.",
    "Low responsiveness ({rrr} response rate). Factor this into your pipeline — they may be worth it, but they're not easy to reach.",
    "Recruiter response rate: {rrr}. Historical pattern says most messages go unanswered. Try a warm intro instead of cold outreach.",
]

OVERQUALIFIED_PHRASES = [
    "At {yoe} years, this is a very senior profile. Confirm they're comfortable with founding-team IC scope, not looking for a VP title.",
    "{yoe} years of experience — senior enough to want a leadership role. Make sure the founding-team IC framing matches their expectations.",
    "With {yoe} years, they may expect to manage people, not write code. Align on role scope before investing interview time.",
    "{yoe}-year veteran — impressive depth, but the risk is they want director-level scope that a Series A company can't offer yet.",
]

LONG_NOTICE_PHRASES = [
    "{notice}-day notice is the main friction point. Ask early whether they'd negotiate a buyout or early release.",
    "{notice}-day notice period — not a dealbreaker but plan the pipeline with this lead time in mind.",
    "Notice period: {notice} days. That's {weeks} weeks before a start date. Worth front-loading the offer timeline.",
    "The {notice}-day notice is longer than ideal. If they're the right candidate, start the conversation about buyout options.",
    "{notice}d notice. Ask whether their current employer would release them early — some do for the right departure terms.",
]

FLAT_ARC_PHRASES = [
    "Career is solid but scope hasn't expanded much across roles. A capable executor who may need the right environment to stretch.",
    "Steady career but without clear upward movement in scope or title. Not a concern for execution, but growth mindset is worth probing.",
    "Consistent performer based on tenure, but the trajectory is flat. The question is whether they're coasting or selectively stable.",
    "Career scope has stayed roughly constant across roles. Good for reliable delivery; open question on whether they'll drive 0-to-1 ambiguity.",
]

# ── BOTTOM LINE PHRASE BANKS ─────────────────────────────────────────────────

TOP5_CLOSINGS = [
    "Top of the list for good reasons — prioritize outreach.",
    "This is a first-call candidate. Move fast before someone else does.",
    "Strongest profile in the pool on the dimensions that matter. Reach out today.",
    "Priority outreach — this combination of career depth and availability is rare in the pool.",
    "Top-tier match. The fundamentals are strong; the interview is about fit, not filtering.",
]

TOP15_CLOSINGS = [
    "Strong shortlist candidate. Move to screening.",
    "Clear shortlist material — worth a screening call this week.",
    "Solid on the core requirements. Schedule a screen to validate depth.",
    "Among the best in the pool on technical substance. Advance to first round.",
    "Shortlist-worthy — the profile holds up on the dimensions that matter most.",
]

MID_ACTIVE_CLOSINGS = [
    "Worth a first-round call — the engagement signals are there.",
    "Engagement signals are positive — move to screening while they're active.",
    "Active and responsive — good candidate to pipeline while the top tier is in process.",
    "Behavioral signals say they'll respond. Worth a call to evaluate in person.",
]

MID_PASSIVE_CLOSINGS = [
    "Solid profile but queue behind the top tier unless they thin out.",
    "Good fundamentals, lower urgency. Pipeline as backup to the primary shortlist.",
    "Worth keeping warm. Not the first call, but don't lose track of them.",
    "Technically qualified but not differentiating enough to jump the queue. Hold.",
]

LOWER_TIER_CLOSINGS = [
    "Below the primary shortlist threshold on current signals.",
    "Not in the top tier on composite score — would need a strong interview to override.",
    "Below shortlist cut. If the top candidates don't close, revisit.",
    "Outside the core shortlist. Profile has potential but doesn't lead on key dimensions.",
    "Ranked here because of specific gaps — see above. Reconsider if requirements shift.",
]

BORDERLINE_CLOSINGS = [
    "Borderline — include if the top tier doesn't close or if they shine in screening.",
    "On the edge of the shortlist. A strong reference or interview performance could bump them up.",
    "Marginal on composite. Worth keeping on the long list but not a priority outreach.",
    "Just outside the core shortlist. If the pool thins, they move up.",
]

# ── GITHUB PHRASES ────────────────────────────────────────────────────────────

GITHUB_PHRASES = [
    "Active GitHub (score {score}/100) — ships code outside work hours.",
    "GitHub activity at {score}/100 — evidence of building beyond the day job.",
    "GitHub score: {score}/100. Open-source activity adds credibility to the technical claims.",
    "Builds in the open: GitHub score {score}/100. A signal you don't get from resumes alone.",
]

# ── HIDDEN GEM PHRASES ────────────────────────────────────────────────────────

HIDDEN_GEM_PHRASES = [
    "Hidden gem: career substance is strong despite a thin skills section. Likely a builder who doesn't self-promote.",
    "Profile under-sells them — career history is much stronger than the skills list would suggest.",
    "The skills section is modest but the career trajectory tells a different story. Worth a deeper look.",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 9999


def _months_desc(months: int) -> str:
    """Convert months to a human-readable tenure description."""
    if months >= 24:
        yrs = months // 12
        mo = months % 12
        return f"{yrs}yr {mo}mo" if mo else f"{yrs}yr"
    return f"{months}mo"


def _pick_strongest_skill(candidate: dict) -> Optional[dict]:
    """Find the single most credible AI/ML skill — used to anchor a specific claim."""
    AI_SKILL_KEYWORDS = {
        "embedding", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "bm25", "learning to rank", "ltr", "rerank",
        "sentence-transformer", "sentence transformer", "bge", "e5",
        "rag", "retrieval", "dense retrieval", "vector search", "vector db",
        "nlp", "bert", "transformer", "llm", "fine-tun", "lora", "qlora",
        "recommendation", "ranking", "ndcg", "mrr",
    }
    skills = candidate.get("skills", [])
    best = None
    best_score = -1
    for s in skills:
        name = (s.get("name", "") or "").lower()
        if not any(kw in name for kw in AI_SKILL_KEYWORDS):
            continue
        endorse = int(s.get("endorsements", 0) or 0)
        dur = int(s.get("duration_months", 0) or 0)
        prof = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}.get(
            s.get("proficiency", "beginner"), 1
        )
        score = endorse * 2 + dur + prof * 8
        if score > best_score:
            best_score = score
            best = s
    return best


def _career_arc_type(career: list) -> str:
    """
    Read the career chronologically and classify the arc.
    Returns one of: 'ascending', 'flat', 'consulting_to_product',
                    'product_only', 'consulting_heavy', 'lateral_move'
    """
    CONSULTING = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "tech mahindra", "hcl", "mphasis", "hexaware",
    }
    SENIORITY_LEVELS = {
        "junior": 1, "associate": 1,
        "": 2, "engineer": 2, "developer": 2, "analyst": 2,
        "senior": 3, "lead": 3, "specialist": 3,
        "staff": 4, "principal": 4, "architect": 4,
        "head": 5, "director": 5, "vp": 5,
    }

    if not career:
        return "unknown"

    ordered = list(reversed(career))

    consulting_roles = sum(
        1 for ch in ordered
        if any(cf in (ch.get("company", "") or "").lower() for cf in CONSULTING)
    )
    total = len(ordered)

    if consulting_roles == total:
        return "consulting_heavy"
    if consulting_roles >= 1 and consulting_roles < total:
        early_consulting = any(
            cf in (ordered[0].get("company", "") or "").lower()
            for cf in CONSULTING
        )
        late_product = not any(
            cf in (ordered[-1].get("company", "") or "").lower()
            for cf in CONSULTING
        )
        if early_consulting and late_product:
            return "consulting_to_product"

    if consulting_roles == 0:
        first_title = (ordered[0].get("title", "") or "").lower()
        last_title = (ordered[-1].get("title", "") or "").lower()
        first_level = max(
            (v for k, v in SENIORITY_LEVELS.items() if k in first_title), default=2
        )
        last_level = max(
            (v for k, v in SENIORITY_LEVELS.items() if k in last_title), default=2
        )
        if last_level > first_level:
            return "ascending"
        elif last_level == first_level and total >= 3:
            return "flat"
        return "product_only"

    return "lateral_move"


def _most_recent_relevant_role(career: list) -> Optional[dict]:
    """Return the most recent career entry that looks like an AI/ML/search role."""
    AI_TITLE_KWS = {
        "ml", "machine learning", "ai ", "nlp", "search", "retrieval",
        "recommendation", "ranking", "data scientist", "applied scientist",
        "research engineer",
    }
    for ch in career:
        title = (ch.get("title", "") or "").lower()
        if any(kw in title for kw in AI_TITLE_KWS):
            return ch
    return career[0] if career else None


def _notice_narrative(cid: str, notice_days: int, open_to_work: bool, days_inactive: int) -> str:
    """
    Build an availability narrative that interprets signals, not just lists them.
    Uses cid for phrase variation.
    """
    if days_inactive > 180:
        opts = [
            f"Last login was {days_inactive} days ago — passive candidate. Outreach needs to be warm and specific, not a mass ping.",
            f"{days_inactive}d since last active — this is someone who's not looking. Warm intro required, cold outreach will bounce.",
            f"Inactive for {days_inactive} days. Not monitoring the platform. Needs a personal referral or compelling direct message.",
        ]
        return _pick(cid, opts, "avail")
    elif days_inactive > 60:
        if open_to_work:
            opts = [
                f"Open-to-work flag is on but activity dropped off ({days_inactive}d since last active) — possibly exploring quietly.",
                f"Flagged as open to work but hasn't logged in for {days_inactive}d. May have accepted elsewhere or gone passive.",
                f"Open-to-work is set despite {days_inactive}d of inactivity — could be a stale flag or a quiet explorer.",
            ]
            return _pick(cid, opts, "avail")
        opts = [
            f"Not showing active signals ({days_inactive}d since last login). Worth a reach-out but temper response expectations.",
            f"Last active {days_inactive}d ago — not an active job seeker. Outreach may take multiple touchpoints.",
            f"{days_inactive} days inactive. Reachability is uncertain — budget extra follow-up time.",
        ]
        return _pick(cid, opts, "avail")
    elif notice_days <= 15:
        opts = [
            f"Immediate availability — {notice_days}-day notice. Rare at this level.",
            f"{notice_days}-day notice — essentially available now. Move fast.",
            f"Can start in {notice_days} days. At this seniority, that's unusually quick.",
        ]
        return _pick(cid, opts, "avail")
    elif notice_days <= 30:
        avail = "actively looking" if open_to_work else "available"
        opts = [
            f"{avail.capitalize()}, {notice_days}-day notice. Can move fast.",
            f"{notice_days}d notice and {avail}. Timeline works for an aggressive hiring process.",
            f"Ready in {notice_days} days. {'Looking actively — high intent.' if open_to_work else 'Not flagged as actively looking, but timeline is short.'}",
        ]
        return _pick(cid, opts, "avail")
    elif notice_days <= 60:
        otw = " Open to work." if open_to_work else ""
        opts = [
            f"{notice_days}-day notice — manageable timeline.{otw}",
            f"Notice period: {notice_days}d. Standard and workable.{otw}",
            f"Can start in ~{notice_days} days. No friction on availability.{otw}",
        ]
        return _pick(cid, opts, "avail")
    elif notice_days <= 90:
        otw = " Open to work." if open_to_work else ""
        opts = [
            f"{notice_days}-day notice. Not a blocker, but plan the pipeline accordingly.{otw}",
            f"Notice: {notice_days}d — plan around it. Not ideal but typical for India at this level.{otw}",
            f"{notice_days}d notice. Start the conversation early so timeline doesn't bottleneck the hire.{otw}",
        ]
        return _pick(cid, opts, "avail")
    else:
        opts = [
            f"{notice_days}-day notice is on the long side. Worth asking if they'd negotiate a buyout.",
            f"{notice_days}d notice — longer than ideal. Explore whether early release is possible.",
            f"Notice period: {notice_days} days. Plan around it or ask about buyout options upfront.",
        ]
        return _pick(cid, opts, "avail")


def _detect_title_inflation(career: list) -> Optional[str]:
    """Detect title inflation. Returns the inflated title string if found, None otherwise."""
    SENIOR_MARKERS = ["lead", "staff", "principal", "head", "director", "senior"]
    MANAGEMENT_WORDS = ["managed", "hired", "mentored", "grew the team", "people manager", "reports to me"]
    IC_WORDS = ["implemented", "built", "wrote", "developed", "created", "designed the"]

    for ch in career[:2]:
        title = (ch.get("title", "") or "").lower()
        desc = (ch.get("description", "") or "").lower()
        if not any(sm in title for sm in SENIOR_MARKERS):
            continue
        has_mgmt = any(mw in desc for mw in MANAGEMENT_WORDS)
        has_ic = any(iw in desc for iw in IC_WORDS)
        if has_ic and not has_mgmt:
            return ch.get("title", "Senior role")
    return None


def _get_consulting_companies(career: list):
    """Return list of consulting company names found in career, and the arc type detail."""
    CONSULTING = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "tech mahindra", "hcl", "mphasis", "hexaware",
    }
    found = []
    for ch in career:
        co = (ch.get("company", "") or "").lower()
        if any(cf in co for cf in CONSULTING):
            found.append(ch.get("company", ""))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NARRATIVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, scores: dict, rank: int) -> str:
    """
    Generate Shadow Recruiter narrative — thinking out loud, not a data dump.

    Every phrase is selected deterministically from phrase banks based on candidate_id,
    so no two candidates share the same wording unless they trigger the same concern
    AND hash to the same phrase index (statistically unlikely with 5+ options per bank).
    """
    cid = str(candidate.get("candidate_id", "UNKNOWN"))
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    rs = candidate.get("redrob_signals", {})

    title = str(profile.get("current_title", "Unknown") or "Unknown")
    company = str(profile.get("current_company", "Unknown") or "Unknown")
    yoe = float(profile.get("years_of_experience", 0) or 0)

    notice = int(rs.get("notice_period_days", 90) or 90)
    open_to_work = bool(rs.get("open_to_work_flag", False))
    rrr = float(rs.get("recruiter_response_rate", 0.3) or 0.3)
    days_inactive = _days_since(str(rs.get("last_active_date", "2020-01-01") or "2020-01-01"))
    github = float(rs.get("github_activity_score", -1) or -1)

    arc_type = _career_arc_type(career)
    relevant_role = _most_recent_relevant_role(career)
    strongest_skill = _pick_strongest_skill(candidate)

    career_score = scores.get("career_substance", 0)
    skill_score = scores.get("skill_credibility", 0)
    avail_score = scores.get("behavioral_availability", 0)
    star_score = scores.get("star_predictor", 0)
    is_hidden_gem = scores.get("hidden_gem", False)

    # ── Honeypot: short rejection ─────────────────────────────────────────────
    if scores.get("honeypot"):
        return (
            f"Profile flagged as invalid: {scores.get('honeypot_reason', 'data inconsistency')}."
            " Excluded from ranking."
        )

    # ── Component A: Opening hook ─────────────────────────────────────────────
    def opening():
        if arc_type == "ascending" and career_score >= 0.5:
            first = career[-1] if career else {}
            first_title = first.get("title", "engineer")
            opts = [
                f"{yoe:.0f} years, clear upward arc: started as {first_title} and now {title} at {company}.",
                f"Grew from {first_title} to {title} at {company} over {yoe:.0f} years — the career trajectory is distinctly upward.",
                f"Career progression from {first_title} to {title} ({yoe:.0f}yr). Currently at {company} — ascending arc.",
            ]
            return _pick(cid, opts, "open")
        elif arc_type == "product_only" and career_score >= 0.4:
            cos = list(dict.fromkeys(ch.get("company") for ch in career if ch.get("company")))
            co_str = ", ".join(str(c) for c in cos[:3])
            opts = [
                f"{yoe:.0f} years entirely at product companies ({co_str}) — no services stint.",
                f"Pure product background: {co_str}. {yoe:.0f} years, zero consulting.",
                f"Product-only career across {co_str} ({yoe:.0f}yr). That's rare in this pool.",
            ]
            return _pick(cid, opts, "open")
        elif arc_type == "consulting_to_product":
            last_co = career[0].get("company", company) if career else company
            opts = [
                f"Started at a services firm, now {title} at {last_co}. {yoe:.0f} years — the move to product is notable.",
                f"Consulting-to-product transition: now {title} at {last_co} ({yoe:.0f}yr total). The direction matters.",
                f"{yoe:.0f}-year career that started in consulting and landed at {last_co} as {title}. Self-directed transition.",
            ]
            return _pick(cid, opts, "open")
        elif relevant_role:
            dur = int(relevant_role.get("duration_months", 0) or 0)
            rel_title = relevant_role.get("title", "?")
            rel_co = relevant_role.get("company", "?")
            opts = [
                f"{yoe:.0f}-year {title} at {company}. Most relevant role: {rel_title} at {rel_co} ({_months_desc(dur)}).",
                f"Currently {title} at {company} ({yoe:.0f}yr). Relevant experience: {rel_title} at {rel_co} for {_months_desc(dur)}.",
                f"{title} at {company}, {yoe:.0f} years total. Closest JD-match role: {rel_title} at {rel_co} ({_months_desc(dur)}).",
            ]
            return _pick(cid, opts, "open")
        else:
            opts = [
                f"{yoe:.0f}-year {title} at {company}.",
                f"Currently {title} at {company} with {yoe:.0f} years of experience.",
                f"{title}, {company} ({yoe:.0f}yr).",
            ]
            return _pick(cid, opts, "open")

    # ── Component B: Evidence read ────────────────────────────────────────────
    def evidence():
        if strongest_skill:
            name = strongest_skill.get("name", "")
            endorse = int(strongest_skill.get("endorsements", 0) or 0)
            dur = int(strongest_skill.get("duration_months", 0) or 0)
            prof = strongest_skill.get("proficiency", "intermediate")
            vals = {"name": name, "endorse": endorse, "dur": dur, "prof": prof}

            if endorse >= 30 and dur >= 24:
                return _pick(cid, DEEP_SKILL_PHRASES, "evid").format(**vals)
            elif endorse >= 15:
                return _pick(cid, DECENT_SKILL_PHRASES, "evid").format(**vals)
            elif dur >= 24:
                return _pick(cid, DURATION_ONLY_PHRASES, "evid").format(**vals)
            else:
                return _pick(cid, WEAK_SKILL_PHRASES, "evid").format(**vals)

        if career_score >= 0.6:
            return _pick(cid, CAREER_STRONG_PHRASES, "evid")
        elif career_score >= 0.35:
            return _pick(cid, CAREER_MODERATE_PHRASES, "evid")
        # Weak/no evidence — composed entirely from candidate data
        n_skills = len(skills)
        n_career = len(career)
        top_skill = skills[0].get("name", "unspecified") if skills else "none"
        latest = career[0] if career else {}
        latest_role = latest.get("title", title)
        latest_co = latest.get("company", company)

        idx = _pick_idx(cid, 4, "evid")
        if idx == 0:
            return f"{title} at {company}, {yoe:.0f}yr, {n_skills} skills (strongest: {top_skill}). Career under {latest_role} at {latest_co} lacks ML production signals."
        elif idx == 1:
            return f"{n_career} roles over {yoe:.0f} years as {title}. Top skill: {top_skill}. The {company} trajectory shows no retrieval or ranking work."
        elif idx == 2:
            return f"Currently {latest_role} at {latest_co}. {n_skills} skills on profile, {top_skill} being the strongest. Missing the AI/ML career thread this JD requires."
        else:
            return f"{yoe:.0f}yr {title} ({company}), {n_career} career entries, best skill: {top_skill}. No evidence of production ML, ranking, or retrieval systems work."

    # ── Component C: Concern ──────────────────────────────────────────────────
    def concern():
        # Title inflation
        inflated_title = _detect_title_inflation(career)
        if inflated_title:
            return _pick(cid, TITLE_INFLATION_PHRASES, "concern").format(title=inflated_title)

        # Consulting background
        consulting_cos = _get_consulting_companies(career)
        if arc_type == "consulting_heavy" and consulting_cos:
            co_str = ", ".join(str(c) for c in list(dict.fromkeys(consulting_cos))[:2])
            return _pick(cid, CONSULTING_HEAVY_PHRASES, "concern").format(companies=co_str)
        if arc_type == "consulting_to_product" and consulting_cos:
            early_co = consulting_cos[-1] if consulting_cos else "a services firm"
            return _pick(cid, CONSULTING_TRANSITION_PHRASES, "concern").format(early_co=early_co)

        # Thin skills
        if skill_score < 0.20 and career_score >= 0.35:
            return _pick(cid, THIN_SKILLS_PHRASES, "concern")

        # Ghost
        if rrr < 0.2:
            rrr_str = f"{rrr:.0%}"
            return _pick(cid, GHOST_PHRASES, "concern").format(rrr=rrr_str)

        # Overqualified
        if yoe > 12 and career_score >= 0.5:
            return _pick(cid, OVERQUALIFIED_PHRASES, "concern").format(yoe=f"{yoe:.0f}")

        # Long notice
        if notice > 90:
            weeks = notice // 7
            return _pick(cid, LONG_NOTICE_PHRASES, "concern").format(notice=notice, weeks=weeks)

        # Flat arc
        if star_score < 0.3 and career_score >= 0.4:
            return _pick(cid, FLAT_ARC_PHRASES, "concern")

        return None

    # ── Component D: Availability ─────────────────────────────────────────────
    def availability():
        return _notice_narrative(cid, notice, open_to_work, days_inactive)

    # ── Component E: Bottom line ──────────────────────────────────────────────
    def bottom_line():
        comp = scores.get("composite", 0)
        if rank <= 5:
            idx = _pick_idx(cid, 5, "close")
            if idx == 0: return f"Rank {rank} of 100 (composite {comp:.3f}). Prioritize outreach."
            elif idx == 1: return f"Top {rank} for a reason — move fast."
            elif idx == 2: return f"At {comp:.3f} composite, this is a first-call candidate."
            elif idx == 3: return f"Rank {rank}: best-in-pool. The interview is about fit, not filtering."
            else: return f"Composite {comp:.3f} puts them at rank {rank}. Reach out today."
        elif rank <= 15:
            idx = _pick_idx(cid, 5, "close")
            if idx == 0: return f"Rank {rank} (composite {comp:.3f}). Strong shortlist — move to screening."
            elif idx == 1: return f"At rank {rank}, clear shortlist material. Schedule a screen."
            elif idx == 2: return f"Composite {comp:.3f}, rank {rank}. Solid on core requirements."
            elif idx == 3: return f"Rank {rank} of 100. Among the best on technical substance."
            else: return f"Shortlist-worthy at rank {rank} ({comp:.3f}). Advance to first round."
        elif rank <= 30:
            idx = _pick_idx(cid, 4, "close")
            if avail_score >= 0.7:
                if idx == 0: return f"Rank {rank} with strong engagement signals. Worth a call."
                elif idx == 1: return f"At rank {rank} ({comp:.3f}), the availability signals favor action."
                elif idx == 2: return f"Rank {rank}: responsive candidate. Pipeline while the top tier processes."
                else: return f"Composite {comp:.3f}, rank {rank}. Active and reachable."
            else:
                if idx == 0: return f"Rank {rank} ({comp:.3f}). Queue behind top tier unless they thin out."
                elif idx == 1: return f"At rank {rank}, solid but not urgent. Keep warm."
                elif idx == 2: return f"Rank {rank}: technically qualified, not differentiating enough to jump the queue."
                else: return f"Composite {comp:.3f} puts them at rank {rank}. Backup shortlist."
        elif rank <= 60:
            idx = _pick_idx(cid, 4, "close")
            if idx == 0: return f"Rank {rank} ({comp:.3f}). Borderline — include if top tier doesn't close."
            elif idx == 1: return f"At rank {rank}, a strong interview could change the picture."
            elif idx == 2: return f"Rank {rank}: marginal on composite ({comp:.3f}). Long list, not priority."
            else: return f"Just outside the core shortlist at rank {rank}."
        else:
            idx = _pick_idx(cid, 5, "close")
            if idx == 0: return f"Rank {rank} ({comp:.3f}). Below the primary shortlist."
            elif idx == 1: return f"At rank {rank}, would need a strong interview to override the data."
            elif idx == 2: return f"Rank {rank}: below shortlist cut. Revisit if top candidates don't close."
            elif idx == 3: return f"Composite {comp:.3f} places them at rank {rank}. Gaps noted above."
            else: return f"Ranked {rank} of 100 ({comp:.3f}). Reconsider if requirements shift."

    # ── Assemble narrative ────────────────────────────────────────────────────

    parts = []

    # TOP TIER (1-15): endorsement + evidence + one probe
    if rank <= 15:
        parts.append(opening())
        parts.append(evidence())
        con = concern()
        if con:
            # Vary the concern prefix
            prefix = _pick(cid, [
                "One thing to probe:",
                "Flag for screening:",
                "Question to ask:",
                "Worth investigating:",
                "Point of clarification:",
            ], "prefix")
            parts.append(f"{prefix} {con}")
        if days_inactive > 60 or notice > 90 or notice <= 15:
            parts.append(availability())
        if github >= 50:
            parts.append(_pick(cid, GITHUB_PHRASES, "gh").format(score=f"{github:.0f}"))
        if is_hidden_gem:
            parts.append(_pick(cid, HIDDEN_GEM_PHRASES, "gem"))
        parts.append(bottom_line())

    # MID TIER (16-50): balanced read, honest gap
    elif rank <= 50:
        parts.append(opening())
        parts.append(evidence())
        con = concern()
        if con:
            gap_prefix = _pick(cid, ["Gap:", "Concern:", "Watch out:", "Open question:", "Risk:"], "gappfx")
            parts.append(f"{gap_prefix} {con}")
        parts.append(availability())
        if is_hidden_gem:
            parts.append(_pick(cid, HIDDEN_GEM_PHRASES, "gem"))
        parts.append(bottom_line())

    # LOWER TIER (51-100): honest, direct, not harsh
    else:
        parts.append(opening())
        # Compose tier reason from candidate-specific data — never use static phrases
        cs_pct = f"{career_score:.0%}"
        sk_pct = f"{skill_score:.0%}"
        av_pct = f"{avail_score:.0%}"
        st_pct = f"{star_score:.2f}"

        if career_score < 0.15:
            idx = _pick_idx(cid, 3, "tier")
            if idx == 0:
                parts.append(f"Career substance at {cs_pct} is below the shortlist threshold for {title}. ML production work not evident.")
            elif idx == 1:
                parts.append(f"Scored {cs_pct} on career substance — insufficient AI/ML signal from the {company} trajectory to rank higher.")
            else:
                parts.append(f"With career substance at {cs_pct} and skills at {sk_pct}, the profile doesn't clear the ML threshold.")
        elif skill_score < 0.15:
            idx = _pick_idx(cid, 3, "tier")
            if idx == 0:
                parts.append(f"Skills credibility at {sk_pct} is the gap — career ({cs_pct}) shows AI interest but endorsements and duration don't back it up.")
            elif idx == 1:
                parts.append(f"Career substance is {cs_pct} but skill corroboration only {sk_pct}. Needs a technical screen to validate depth.")
            else:
                parts.append(f"AI/ML career signal exists ({cs_pct}) but skills scored {sk_pct} on credibility. Unverified claims hold this profile back.")
        elif avail_score < 0.4:
            parts.append(
                f"Availability drags the composite: {days_inactive}d inactive, {rrr:.0%} response rate, availability score {av_pct}. Technically {cs_pct} on career substance but unreachable."
            )
        else:
            # The default bucket — most lower-tier candidates land here
            # Use the actual composite dimensions to explain WHY they're here, not a generic phrase
            # Find their weakest dimension
            dims = [
                ("career substance", career_score),
                ("skill credibility", skill_score),
                ("experience quality", scores.get("experience_quality", 0)),
                ("availability", avail_score),
                ("career trajectory", star_score),
            ]
            dims.sort(key=lambda x: x[1])
            weakest_name, weakest_val = dims[0]
            second_name, second_val = dims[1]

            idx = _pick_idx(cid, 4, "tier")
            comp = scores.get("composite", 0)
            if idx == 0:
                parts.append(f"At rank {rank} ({comp:.3f}), {weakest_name} ({weakest_val:.0%}) and {second_name} ({second_val:.0%}) are the limiting factors for this {title} at {company}.")
            elif idx == 1:
                parts.append(f"Rank {rank} ({comp:.3f}): {weakest_name} at {weakest_val:.0%} holds back an otherwise competent {title} from {company}.")
            elif idx == 2:
                parts.append(f"Ranked {rank} with composite {comp:.3f}. The {weakest_name} dimension ({weakest_val:.0%}) is the gap — {second_name} at {second_val:.0%} doesn't compensate.")
            else:
                parts.append(f"Composite {comp:.3f} (rank {rank}): {cs_pct} career, {sk_pct} skills, {st_pct} trajectory — none individually disqualifying, but collectively below the top-50 cut.")
        parts.append(bottom_line())

    # Join with spaces
    narrative = " ".join(p.rstrip(".") + "." for p in parts if p)

    # Enforce 300-char limit
    if len(narrative) > 300:
        narrative = narrative[:297] + "..."

    return narrative
