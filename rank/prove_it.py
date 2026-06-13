"""
prove_it  –  cross-reference a candidate's claims against profile evidence.

Produces 3-5 claim assessments per candidate, each tagged as
corroborated / unverified / contradicted.
"""

import re

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OWNERSHIP_VERBS = ("led", "built", "architected", "founded", "designed",
                    "created", "owned")

_LEADERSHIP_TITLES = ("lead", "staff", "head", "principal", "director",
                      "vp", "cto", "ceo", "founder", "co-founder",
                      "chief", "manager", "senior")


def _safe_str(value):
    """Return a lowered string or empty string for None / non-string."""
    if value is None:
        return ""
    return str(value).lower()


def _safe_int(value, default=0):
    """Return an int, falling back to *default* for None / bad types."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    """Return a float, falling back to *default* for None / bad types."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _career_descriptions(candidate):
    """Yield (index, lowered-description) for every career entry."""
    history = candidate.get("career_history") or []
    for idx, entry in enumerate(history):
        desc = _safe_str(entry.get("description"))
        if desc:
            yield idx, desc


def _career_titles(candidate):
    """Yield (index, lowered-title) for every career entry."""
    history = candidate.get("career_history") or []
    for idx, entry in enumerate(history):
        title = _safe_str(entry.get("title"))
        if title:
            yield idx, title


def _keyword_in_texts(keyword, texts):
    """
    Check if *keyword* (lowered) appears in any of the given texts.
    Returns the first matching index or -1.
    Uses word-boundary matching so 'java' won't match 'javascript'.
    """
    pattern = re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")
    for idx, text in texts:
        if pattern.search(text):
            return idx
    return -1


# ---------------------------------------------------------------------------
# Claim generators
# ---------------------------------------------------------------------------

def _expert_skill_claims(candidate):
    """
    Top-2 most-endorsed expert/advanced skills → claim assessments.
    """
    skills = candidate.get("skills") or []
    expert_skills = []
    for sk in skills:
        prof = _safe_str(sk.get("proficiency"))
        if prof in ("expert", "advanced"):
            endorsements = _safe_int(sk.get("endorsements"))
            expert_skills.append((endorsements, sk))

    # Sort descending by endorsements, pick top 2
    expert_skills.sort(key=lambda x: x[0], reverse=True)
    top_skills = [pair[1] for pair in expert_skills[:2]]

    results = []
    descs = list(_career_descriptions(candidate))

    for sk in top_skills:
        name = sk.get("name") or "unknown"
        proficiency = _safe_str(sk.get("proficiency"))
        endorsements = _safe_int(sk.get("endorsements"))
        duration = _safe_int(sk.get("duration_months"))

        claim = "{prof}-level {name}".format(prof=proficiency.capitalize(),
                                             name=name)

        # Rule: duration_months == 0 but proficiency is 'expert' → contradicted
        if proficiency == "expert" and duration == 0:
            results.append({
                "claim": claim,
                "verdict": "contradicted",
                "evidence": (
                    "Skill '{name}' listed as expert but has 0 months "
                    "duration recorded".format(name=name)
                ),
                "source": "skills"
            })
            continue

        # Rule: keyword appears in any career description → corroborated
        match_idx = _keyword_in_texts(name, descs)
        if match_idx >= 0:
            results.append({
                "claim": claim,
                "verdict": "corroborated",
                "evidence": (
                    "'{name}' mentioned in career_history[{idx}] "
                    "description".format(name=name, idx=match_idx)
                ),
                "source": "career_history[{idx}]".format(idx=match_idx)
            })
            continue

        # Rule: >= 15 endorsements AND >= 24 months → corroborated
        if endorsements >= 15 and duration >= 24:
            results.append({
                "claim": claim,
                "verdict": "corroborated",
                "evidence": (
                    "{end} endorsements and {dur} months of experience "
                    "with '{name}'".format(end=endorsements, dur=duration,
                                           name=name)
                ),
                "source": "skills"
            })
            continue

        # Fallback → unverified
        parts = []
        if endorsements < 15:
            parts.append(
                "only {end} endorsements (need 15+)".format(end=endorsements)
            )
        if duration < 24:
            parts.append(
                "only {dur} months duration (need 24+)".format(dur=duration)
            )
        detail = "; ".join(parts) if parts else "no supporting evidence found"
        results.append({
            "claim": claim,
            "verdict": "unverified",
            "evidence": (
                "'{name}' not found in career descriptions; "
                "{detail}".format(name=name, detail=detail)
            ),
            "source": "skills"
        })

    return results


def _ownership_claims(candidate):
    """
    Scan summary + headline for ownership verbs → 1 strongest claim.
    """
    profile = candidate.get("profile") or {}
    summary = _safe_str(profile.get("summary"))
    headline = _safe_str(profile.get("headline"))
    combined = summary + " " + headline

    # Find all ownership verbs present
    found = []
    for verb in _OWNERSHIP_VERBS:
        pattern = re.compile(r"\b" + re.escape(verb) + r"\b")
        match = pattern.search(combined)
        if match:
            # Capture a snippet around the match for the claim text
            start = max(0, match.start() - 30)
            end = min(len(combined), match.end() + 40)
            snippet = combined[start:end].strip()
            found.append((verb, snippet))

    if not found:
        return []

    # Pick the first (strongest = first in our priority-ordered tuple)
    verb, snippet = found[0]
    claim = "Ownership claim: '...{snippet}...'".format(snippet=snippet)

    descs = list(_career_descriptions(candidate))
    titles = list(_career_titles(candidate))

    # Check career descriptions for matching ownership language
    desc_match = _keyword_in_texts(verb, descs)
    if desc_match >= 0:
        return [{
            "claim": claim,
            "verdict": "corroborated",
            "evidence": (
                "Verb '{verb}' also found in career_history[{idx}] "
                "description".format(verb=verb, idx=desc_match)
            ),
            "source": "career_history[{idx}]".format(idx=desc_match)
        }]

    # Check career titles for leadership-level titles
    for idx, title in titles:
        for lt in _LEADERSHIP_TITLES:
            if lt in title:
                return [{
                    "claim": claim,
                    "verdict": "corroborated",
                    "evidence": (
                        "Title '{title}' in career_history[{idx}] suggests "
                        "ownership-level role".format(
                            title=title, idx=idx
                        )
                    ),
                    "source": "career_history[{idx}]".format(idx=idx)
                }]

    # No supporting evidence
    return [{
        "claim": claim,
        "verdict": "unverified",
        "evidence": (
            "Ownership verb '{verb}' in profile summary/headline but no "
            "matching language or leadership title in career history".format(
                verb=verb
            )
        ),
        "source": "profile.summary"
    }]


def _yoe_claim(candidate):
    """
    Compare stated years_of_experience against sum of career durations.
    """
    profile = candidate.get("profile") or {}
    stated_yoe = _safe_float(profile.get("years_of_experience"))
    if stated_yoe == 0.0 and profile.get("years_of_experience") is None:
        # No stated YoE → skip
        return []

    history = candidate.get("career_history") or []
    total_months = 0
    for entry in history:
        total_months += _safe_int(entry.get("duration_months"))

    computed_yoe = total_months / 12.0 if total_months > 0 else 0.0
    delta = abs(stated_yoe - computed_yoe)

    claim = "States {yoe} years of experience".format(
        yoe=("{:.1f}".format(stated_yoe)).rstrip("0").rstrip(".")
    )

    if delta > 1.5:
        direction = "exceeds" if stated_yoe > computed_yoe else "is less than"
        return [{
            "claim": claim,
            "verdict": "contradicted",
            "evidence": (
                "Stated YoE ({stated}) {dir} career timeline total "
                "({computed} years) by {delta} years".format(
                    stated="{:.1f}".format(stated_yoe),
                    dir=direction,
                    computed="{:.1f}".format(computed_yoe),
                    delta="{:.1f}".format(delta)
                )
            ),
            "source": "profile.years_of_experience vs career_history"
        }]

    return [{
        "claim": claim,
        "verdict": "corroborated",
        "evidence": (
            "Career timeline totals {computed} years, within 1.5yr of "
            "stated {stated} years".format(
                computed="{:.1f}".format(computed_yoe),
                stated="{:.1f}".format(stated_yoe)
            )
        ),
        "source": "profile.years_of_experience vs career_history"
    }]


def _current_title_claim(candidate):
    """
    Does profile.current_title match career_history[0].title?
    """
    profile = candidate.get("profile") or {}
    current_title = _safe_str(profile.get("current_title"))
    if not current_title:
        return []

    history = candidate.get("career_history") or []
    if not history:
        return [{
            "claim": "Current title: '{title}'".format(
                title=profile.get("current_title", "")
            ),
            "verdict": "unverified",
            "evidence": "No career history entries to verify against",
            "source": "profile.current_title"
        }]

    latest_title = _safe_str(history[0].get("title"))
    if not latest_title:
        return [{
            "claim": "Current title: '{title}'".format(
                title=profile.get("current_title", "")
            ),
            "verdict": "unverified",
            "evidence": "Latest career history entry has no title",
            "source": "career_history[0]"
        }]

    # Check for significant match: either substring containment or
    # high word overlap
    if current_title == latest_title:
        verdict = "corroborated"
        evidence = (
            "Profile current_title exactly matches career_history[0].title"
        )
    elif current_title in latest_title or latest_title in current_title:
        verdict = "corroborated"
        evidence = (
            "Profile current_title ('{ct}') is a substring match with "
            "career_history[0].title ('{lt}')".format(
                ct=current_title, lt=latest_title
            )
        )
    else:
        # Word-overlap heuristic: if >= 50% of words match → corroborated
        ct_words = set(current_title.split())
        lt_words = set(latest_title.split())
        if ct_words and lt_words:
            overlap = len(ct_words & lt_words)
            max_len = max(len(ct_words), len(lt_words))
            ratio = overlap / max_len if max_len else 0.0
            if ratio >= 0.5:
                verdict = "corroborated"
                evidence = (
                    "Profile current_title ('{ct}') shares significant "
                    "word overlap with career_history[0].title "
                    "('{lt}')".format(ct=current_title, lt=latest_title)
                )
            else:
                verdict = "unverified"
                evidence = (
                    "Profile current_title ('{ct}') differs significantly "
                    "from career_history[0].title ('{lt}')".format(
                        ct=current_title, lt=latest_title
                    )
                )
        else:
            verdict = "unverified"
            evidence = (
                "Could not compare current_title ('{ct}') with "
                "career_history[0].title ('{lt}')".format(
                    ct=current_title, lt=latest_title
                )
            )

    return [{
        "claim": "Current title: '{title}'".format(
            title=profile.get("current_title", "")
        ),
        "verdict": verdict,
        "evidence": evidence,
        "source": "career_history[0]"
    }]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_prove_it(candidate):
    """
    Returns a list of claim assessments (3-5 per candidate):
    [
      {
        "claim": "Led the ML platform team at Flipkart",
        "verdict": "corroborated",   # or "unverified" or "contradicted"
        "evidence": "Career history shows 'Staff ML Engineer' title at "
                    "Flipkart for 2yr 4mo",
        "source": "career_history[1]"
      },
      ...
    ]
    """
    if not candidate or not isinstance(candidate, dict):
        return []

    results = []
    results.extend(_expert_skill_claims(candidate))
    results.extend(_ownership_claims(candidate))
    results.extend(_yoe_claim(candidate))
    results.extend(_current_title_claim(candidate))

    return results


def prove_it_summary(results):
    """
    Returns a one-line trust signal for embedding in reasoning strings.

    Examples:
      - 'Profile claims corroborated by career evidence.'
      - 'One claim unverified: expert-level FAISS not supported by career
         descriptions.'
      - '2 contradictions found: stated YoE exceeds career timeline by
         3 years.'
    """
    if not results:
        return "No claims could be extracted for verification."

    counts = {"corroborated": 0, "unverified": 0, "contradicted": 0}
    for r in results:
        verdict = (r.get("verdict") or "unverified").lower()
        if verdict in counts:
            counts[verdict] += 1
        else:
            counts["unverified"] += 1

    contradicted = counts["contradicted"]
    unverified = counts["unverified"]
    corroborated = counts["corroborated"]

    # Prioritise the most alarming signal
    if contradicted > 0:
        # Find first contradicted claim for detail
        detail_claim = None
        for r in results:
            if (r.get("verdict") or "").lower() == "contradicted":
                detail_claim = r
                break
        detail = ""
        if detail_claim:
            detail = ": {ev}".format(ev=detail_claim.get("evidence", ""))
        noun = "contradiction" if contradicted == 1 else "contradictions"
        return "{n} {noun} found{detail}.".format(
            n=contradicted, noun=noun, detail=detail
        )

    if unverified > 0:
        detail_claim = None
        for r in results:
            if (r.get("verdict") or "").lower() == "unverified":
                detail_claim = r
                break
        detail = ""
        if detail_claim:
            detail = ": {cl}".format(cl=detail_claim.get("claim", ""))
        if unverified == 1:
            return "One claim unverified{detail}.".format(detail=detail)
        return "{n} claims unverified{detail}.".format(
            n=unverified, detail=detail
        )

    # All corroborated
    return "Profile claims corroborated by career evidence."
