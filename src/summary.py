"""Writes the short plain-language note a mentor sees on the Student 360 page.
Rule-based text built from the student's own numbers, no external API."""

GOOD, WEAK = 70, 50


def _level(score):
    if score >= GOOD:
        return "strong"
    if score >= WEAK:
        return "average"
    return "weak"


def student_summary(s, cohort_mean, pillars, actions):
    first = s["name"].split()[0]
    gap = s["success_score"] - cohort_mean["success_score"]
    where = "above" if gap >= 0 else "below"
    lines = [f"{first} has a Success Score of {s['success_score']:.0f}, {abs(gap):.0f} points {where} the campus "
             f"average, and is flagged {s['risk_flag'].lower()} risk."]

    diffs = {p: s[p] - cohort_mean[p] for p in pillars}
    best = max(diffs, key=diffs.get)
    worst = min(diffs, key=diffs.get)
    if diffs[worst] < -5:
        lines.append(f"The biggest drag is {worst.lower()} ({s[worst]:.0f}/100, {_level(s[worst])}), "
                     f"while {best.lower()} is the relative strength ({s[best]:.0f}/100).")
    else:
        lines.append(f"No pillar is far below average; {best.lower()} is the clearest strength ({s[best]:.0f}/100).")

    change = s["attendance_change_3m"]
    if change <= -10:
        lines.append(f"Attendance has dropped {abs(change):.0f} points in three months, which usually shows up in marks "
                     f"a few weeks later.")
    elif change >= 5:
        lines.append(f"Attendance is up {change:.0f} points over three months, a good sign.")

    acad, plac = s.get("academic_risk_probability"), s.get("placement_risk_probability")
    if acad is not None:
        lines.append(f"The early-warning models put the chance of a poor academic outcome at {acad:.0f}% and of a poor "
                     f"placement outcome at {plac:.0f}%.")
        if s["risk_flag"] == "Low" and max(acad, plac) >= 50:
            lines.append("No rule has fired yet, so this student would be missed by a threshold-based report.")
        elif acad < 35 and plac >= 60:
            lines.append("Academically this student is fine; the gap is placement readiness, which is where the effort "
                         "should go.")

    steps = [actions[p] for p in sorted(diffs, key=diffs.get)[:2] if diffs[p] < 0]
    return " ".join(lines), steps
