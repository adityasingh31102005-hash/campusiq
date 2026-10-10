"""CampusIQ pipeline: load 8 raw sources -> clean -> integrate -> Success Score -> risk -> segments.
Everything the dashboard shows comes from build().
"""
from pathlib import Path
import numpy as np
import pandas as pd

import risk_model

DATA = Path(__file__).resolve().parent.parent / "data"

# Pillar weights (sum = 100). Explained in docs/success_score_note.md
WEIGHTS = {"Academic": 25, "Attendance": 20, "Placement": 20, "LMS": 15,
           "Skills": 10, "Engagement": 5, "Feedback": 5}
PILLARS = list(WEIGHTS)

SOURCES = {
    "academics": "Academic", "subject_marks": "Academic", "attendance": "Attendance",
    "lms_activity": "LMS", "engagement": "Engagement", "placement": "Placement",
    "skills": "Skills", "feedback": "Feedback",
}


def clean_id(s: pd.Series) -> pd.Series:
    """'stu-0012', ' STU0012 ', 'stu_0012' -> 'STU0012'"""
    digits = s.astype(str).str.extract(r"(\d+)")[0]
    return "STU" + digits.str.zfill(4)


REQUIRED_COLUMNS = {
    "academics": ["student_id", "cgpa", "backlogs"],
    "subject_marks": ["student_id", "subject", "marks"],
    "attendance": ["student_id", "subject", "classes_held", "classes_attended"],
    "lms_activity": ["student_id", "logins_per_week", "assignments_assigned", "assignments_submitted", "last_login_days_ago"],
    "engagement": ["student_id", "events_attended", "clubs_joined", "hackathons", "certifications"],
    "placement": ["student_id", "aptitude_score", "coding_score", "mock_interviews_taken", "mock_interview_avg"],
    "skills": ["student_id", "technical_score", "soft_skill_score"],
    "feedback": ["student_id", "satisfaction_1to5", "faculty_feedback_1to5"],
}


def missing_columns(name, df):
    return [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]


def _load(name, report, uploads):
    df = uploads[name].copy() if name in uploads else pd.read_csv(DATA / f"{name}.csv")
    raw = len(df)
    fixed = clean_id(df["student_id"])
    n_bad = int((fixed != df["student_id"]).sum())
    df["student_id"] = fixed
    if n_bad:
        report.append((name, "Inconsistent student ID format", n_bad, "Standardised to STU####"))
    return df, raw


def _dedupe(df, name, report, keys=("student_id",)):
    n = int(df.duplicated(list(keys)).sum())
    if n:
        report.append((name, "Duplicate records", n, "Kept latest record"))
    return df.drop_duplicates(list(keys), keep="last")


def _num(df, cols, name, report):
    for c in cols:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n = int(df[c].isna().sum() - before)
        if n:
            report.append((name, f"Non-numeric text in '{c}'", n, "Converted to missing"))
    return df


def compute_pillars(d: pd.DataFrame) -> pd.DataFrame:
    """Turn integrated raw indicators into seven 0-100 pillar scores + the Success Score.
    Works on the whole cohort or on a single edited row (what-if simulator)."""
    d = d.copy()
    backlog_score = np.clip(100 - 25 * d["backlogs"], 0, 100)
    d["Academic"] = 0.5 * d["cgpa"] * 10 + 0.3 * d["avg_marks"] + 0.2 * backlog_score
    d["Attendance"] = np.clip((d["attendance_pct"] - 40) / 55, 0, 1) * 100
    recency = np.clip((30 - d["last_login_days_ago"]) / 27, 0, 1) * 100
    d["LMS"] = (0.5 * d["assignment_completion"] + 0.3 * np.clip(d["logins_per_week"] / 10, 0, 1) * 100
                + 0.2 * recency)
    has_mock = d["mock_interviews_taken"] > 0
    d["Placement"] = np.where(
        has_mock,
        0.35 * d["aptitude_score"] + 0.35 * d["coding_score"] + 0.30 * d["mock_interview_avg"].fillna(0),
        0.5 * d["aptitude_score"] + 0.5 * d["coding_score"])
    d["Skills"] = 0.6 * d["technical_score"] + 0.4 * d["soft_skill_score"]
    d["Engagement"] = np.clip(12 * d["events_attended"] + 15 * d["clubs_joined"] + 20 * d["hackathons"]
                              + 15 * d["certifications"], 0, 100)
    d["Feedback"] = ((d["satisfaction_1to5"] - 1) / 4 * 50 + (d["faculty_feedback_1to5"] - 1) / 4 * 50)
    d[PILLARS] = d[PILLARS].clip(0, 100).round(1)
    d["success_score"] = sum(d[p] * w / 100 for p, w in WEIGHTS.items()).round(1)
    return d


TRIGGERS = {
    "Attendance below 65%": lambda d: d["attendance_pct"] < 65,
    "2 or more backlogs": lambda d: d["backlogs"] >= 2,
    "CGPA below 5.5": lambda d: d["cgpa"] < 5.5,
    "Under half of assignments submitted": lambda d: d["assignment_completion"] < 50,
    "No LMS login for 14+ days": lambda d: d["last_login_days_ago"] >= 14,
    "Attendance fell 10+ points in 3 months": lambda d: d["attendance_change_3m"] <= -10,
}

ACTIONS = {
    "Academic": "Assign a subject tutor and schedule remedial classes for weakest subjects",
    "Attendance": "Mentor meeting this week; send attendance alert to student and parent",
    "LMS": "Set an assignment catch-up plan with weekly check-ins",
    "Placement": "Enrol in aptitude + coding bootcamp and book 2 mock interviews",
    "Skills": "Recommend a targeted certification / soft-skills workshop",
    "Engagement": "Connect with a club or peer group matching their interests",
    "Feedback": "One-to-one counsellor conversation to understand concerns",
}


def flag_risk(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    hits = pd.DataFrame({k: f(d) for k, f in TRIGGERS.items()}, index=d.index)
    d["trigger_count"] = hits.sum(axis=1)
    d["risk_reasons"] = hits.apply(lambda r: "; ".join(r.index[r.values]), axis=1)
    high = (d["success_score"] < 45) | (d["trigger_count"] >= 2)
    med = (d["success_score"] < 58) | (d["trigger_count"] == 1)
    d["risk_flag"] = np.select([high, med], ["High", "Medium"], "Low")
    return d


def segment(d: pd.DataFrame) -> pd.Series:
    conds = [
        (d["attendance_change_3m"] <= -10) & (d["Academic"] >= 50),
        d["Academic"] < 50,
        (d["Academic"] >= 68) & (d["Placement"] < 52),
        (d["Placement"] >= 65) & (d["Academic"] < 62),
        d["success_score"] >= 72,
        (d["Academic"] >= 62) & (d["Engagement"] < 30),
    ]
    names = ["Sudden disengagers", "Academically struggling", "Strong academics, low placement readiness",
             "Job-ready but low grades", "All-round achievers", "Quiet performers"]
    return pd.Series(np.select(conds, names, "Steady middle"), index=d.index)


SEGMENT_ACTION = {
    "Sudden disengagers": "Were doing fine, now slipping fast. Immediate mentor outreach before grades drop.",
    "Academically struggling": "Remedial classes, peer tutoring and backlog-clearing plan.",
    "Strong academics, low placement readiness": "Aptitude/coding bootcamps and mock interviews; they have the grades, not the practice.",
    "Job-ready but low grades": "Protect eligibility: targeted help on CGPA cut-off subjects.",
    "All-round achievers": "Nominate as peer mentors; offer advanced projects and competitions.",
    "Quiet performers": "Good grades but invisible on campus. Nudge towards clubs, events and teamwork.",
    "Steady middle": "Light-touch monitoring; group workshops to move them up a tier.",
}


def build(uploads=None):
    """uploads: {source_name: DataFrame} to replace any source file with one the admin uploaded."""
    uploads = uploads or {}
    report, raw_rows = [], {}
    students = pd.read_csv(DATA / "students.csv")
    ids = set(students["student_id"])
    raw_rows["students"] = len(students)
    t = {}
    for name in SOURCES:
        t[name], raw_rows[name] = _load(name, report, uploads)

    ac = _dedupe(_num(t["academics"], ["cgpa", "backlogs"], "academics", report), "academics", report)
    pct = ac["cgpa"] > 10
    if pct.any():
        report.append(("academics", "CGPA entered as percentage", int(pct.sum()), "Converted to 10-point scale (/9.5)"))
        ac.loc[pct, "cgpa"] = (ac.loc[pct, "cgpa"] / 9.5).round(2)

    sm = _num(t["subject_marks"], ["marks"], "subject_marks", report)
    at = t["attendance"]
    over = at["classes_attended"] > at["classes_held"]
    if over.any():
        report.append(("attendance", "Classes attended exceeds classes held", int(over.sum()), "Capped at classes held"))
        at.loc[over, "classes_attended"] = at.loc[over, "classes_held"]
    at["attendance_pct"] = (at["classes_attended"] / at["classes_held"] * 100).round(1)
    subj = sm.merge(at, on=["student_id", "subject"], how="outer")

    lms = _dedupe(_num(t["lms_activity"], ["logins_per_week"], "lms_activity", report), "lms_activity", report)
    lms["assignment_completion"] = (lms["assignments_submitted"] / lms["assignments_assigned"] * 100).round(1)
    pl = _dedupe(_num(t["placement"], ["aptitude_score", "coding_score"], "placement", report), "placement", report)
    sk = _num(t["skills"], ["technical_score"], "skills", report)
    en, fb = t["engagement"], t["feedback"]

    att_overall = at.groupby("student_id").agg(h=("classes_held", "sum"), a=("classes_attended", "sum"))
    att_overall["attendance_pct"] = (att_overall["a"] / att_overall["h"] * 100).round(1)

    m = students.copy()
    parts = {
        "academics": ac, "subject_marks": sm.groupby("student_id")["marks"].mean().round(1).rename("avg_marks").reset_index(),
        "attendance": att_overall[["attendance_pct"]].reset_index(),
        "lms_activity": lms.drop(columns=["assignments_assigned", "assignments_submitted"]),
        "engagement": en, "placement": pl, "skills": sk, "feedback": fb,
    }
    present = pd.DataFrame(index=m["student_id"])
    for name, df in parts.items():
        present[name] = present.index.isin(df["student_id"])
        m = m.merge(df, on="student_id", how="left")
    m["sources_available"] = present.sum(axis=1).values
    m["missing_sources"] = present.apply(lambda r: ", ".join(r.index[~r.values]), axis=1).values
    for name in parts:
        n = int((~present[name]).sum())
        if n:
            report.append((name, "Students with no record in this source", n, "Imputed with department-year median, flagged"))

    num_cols = [c for c in m.select_dtypes("number").columns if c not in ("year", "sources_available", "mock_interview_avg")]
    n_missing = int(m[num_cols].isna().sum().sum())
    grp_med = m.groupby(["department", "year"])[num_cols].transform("median")
    m[num_cols] = m[num_cols].fillna(grp_med).fillna(m[num_cols].median())
    report.append(("all sources", "Missing values after integration", n_missing, "Filled with department-year median"))

    trends = pd.read_csv(DATA / "monthly_trends.csv")
    wide = trends.pivot(index="student_id", columns="month", values="attendance_pct")
    change = (wide.iloc[:, -1] - wide.iloc[:, -4]).round(1).rename("attendance_change_3m")
    m = m.merge(change, on="student_id", how="left")

    m = flag_risk(compute_pillars(m))
    models = risk_model.train()
    m["academic_risk_probability"] = risk_model.predict(models["academic"]["model"], m)
    m["placement_risk_probability"] = risk_model.predict(models["placement"]["model"], m)
    m["risk_probability"] = m[["academic_risk_probability", "placement_risk_probability"]].max(axis=1)
    # students the rules call Low but at least one model is worried about
    m["watchlist"] = (m["risk_flag"] == "Low") & (m["risk_probability"] >= 50)
    m["ai_risk_type"] = np.select(
        [(m.academic_risk_probability >= 50) & (m.placement_risk_probability >= 50),
         m.academic_risk_probability >= 50, m.placement_risk_probability >= 50],
        ["Academic + placement", "Academic", "Placement"], "None")
    m["segment"] = segment(m)
    m["weakest_pillar"] = m[PILLARS].sub(m[PILLARS].mean()).idxmin(axis=1)
    m["recommended_action"] = m["weakest_pillar"].map(ACTIONS)

    report_df = pd.DataFrame(report, columns=["Source", "Issue found", "Records", "Action taken"])
    src = pd.DataFrame({"Source": list(raw_rows), "Raw rows": list(raw_rows.values())})
    src["Students covered"] = [len(students)] + [int(present[n].sum()) for n in SOURCES]
    src["Coverage %"] = (src["Students covered"] / len(students) * 100).round(1)
    return {"master": m, "report": report_df, "sources": src, "trends": trends, "subjects": subj, "models": models}


def explain(row: pd.Series, cohort_mean: pd.Series) -> pd.DataFrame:
    """Points each pillar adds to this student's score, and the gap vs the cohort average."""
    out = pd.DataFrame({
        "Pillar": PILLARS,
        "Pillar score": [row[p] for p in PILLARS],
        "Weight": [WEIGHTS[p] for p in PILLARS],
    })
    out["Points earned"] = (out["Pillar score"] * out["Weight"] / 100).round(1)
    out["vs cohort avg"] = ((out["Pillar score"] - cohort_mean[PILLARS].values) * out["Weight"] / 100).round(1)
    return out


if __name__ == "__main__":
    r = build()
    m = r["master"]
    print(m["success_score"].describe().round(1))
    print(m["risk_flag"].value_counts())
    print(m["segment"].value_counts())
    print(r["report"].to_string())
    print(r["sources"].to_string())
    for k, v in r["models"].items():
        print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v["metrics"].items() if a != "outcome"})
    print(m["ai_risk_type"].value_counts().to_dict())
    print("watchlist", int(m["watchlist"].sum()), "| avg prob by flag", m.groupby("risk_flag")["risk_probability"].mean().round(0).to_dict())
    print(m[PILLARS].mean().round(1))
    print(m[PILLARS + ["success_score"]].corr()["success_score"].round(2))
