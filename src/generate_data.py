"""Generates the synthetic campus dataset: 8 separate source files, deliberately messy,
so the integration and cleaning pipeline has real work to do.
Run:  python src/generate_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

DEPTS = {
    "CSE": ["Data Structures", "DBMS", "Operating Systems", "Mathematics", "Communication Skills"],
    "IT": ["Web Technologies", "DBMS", "Computer Networks", "Mathematics", "Communication Skills"],
    "AIML": ["Machine Learning", "Python Programming", "Statistics", "Mathematics", "Communication Skills"],
    "ECE": ["Digital Electronics", "Signals & Systems", "Microprocessors", "Mathematics", "Communication Skills"],
    "ME": ["Thermodynamics", "Fluid Mechanics", "Manufacturing", "Mathematics", "Communication Skills"],
    "CE": ["Structural Analysis", "Surveying", "Geotechnical Engg", "Mathematics", "Communication Skills"],
}
FIRST = ["Aarav", "Vivaan", "Aditya", "Diya", "Ananya", "Ishaan", "Kabir", "Meera", "Riya", "Arjun", "Sneha",
         "Rohan", "Kavya", "Tanvi", "Harsh", "Neha", "Yash", "Pooja", "Karan", "Simran", "Dev", "Nikhil",
         "Priya", "Aisha", "Manav", "Sanya", "Rahul", "Tara", "Varun", "Zoya"]
LAST = ["Sharma", "Verma", "Singh", "Gupta", "Patel", "Reddy", "Nair", "Iyer", "Khan", "Das", "Mehta",
        "Joshi", "Kapoor", "Bose", "Chopra", "Yadav", "Mishra", "Pillai", "Gill", "Saxena"]
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]


def clip(x, lo=0, hi=100):
    return np.clip(x, lo, hi)


ids = np.array([f"STU{i:04d}" for i in range(1, N + 1)])
dept = rng.choice(list(DEPTS), N, p=[.28, .18, .16, .16, .12, .10])
year = rng.choice([1, 2, 3, 4], N, p=[.28, .26, .24, .22])

# Hidden traits that drive everything: ability, engagement, career-readiness
a = rng.normal(size=N)
e = 0.5 * a + np.sqrt(0.75) * rng.normal(size=N)
c = 0.35 * a + 0.30 * e + 0.85 * rng.normal(size=N)
declining = rng.random(N) < 0.10  # students who are slipping this semester

students = pd.DataFrame({
    "student_id": ids,
    "name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(N)],
    "department": dept, "year": year,
    "section": rng.choice(list("ABCD"), N),
    "mentor": [f"Prof. {rng.choice(LAST)}" for _ in range(N)],
})

academics = pd.DataFrame({
    "student_id": ids,
    "cgpa": np.round(clip(7.0 + 1.1 * a + rng.normal(0, .4, N), 3.5, 10), 2),
    "backlogs": clip(rng.poisson(np.exp(-1.2 - 0.9 * a)), 0, 8),
})

att_now = clip(78 + 9 * e + 3 * a + rng.normal(0, 5, N) - 12 * declining, 30, 100)
rows = []
for i in range(N):
    for s in DEPTS[dept[i]]:
        held = int(rng.integers(38, 49))
        p = clip(att_now[i] + rng.normal(0, 6), 5, 100) / 100
        rows.append((ids[i], s, round(float(clip(62 + 12 * a[i] + rng.normal(0, 8))), 1),
                     held, int(rng.binomial(held, p))))
subjects = pd.DataFrame(rows, columns=["student_id", "subject", "marks", "classes_held", "classes_attended"])
subject_marks = subjects[["student_id", "subject", "marks"]].copy()
attendance = subjects[["student_id", "subject", "classes_held", "classes_attended"]].copy()

lms = pd.DataFrame({
    "student_id": ids,
    "logins_per_week": np.round(clip(6 + 2.5 * e + rng.normal(0, 1.5, N) - 2.5 * declining, 0, 20), 1),
    "assignments_assigned": 20,
    "assignments_submitted": rng.binomial(20, 1 / (1 + np.exp(-(1.2 + 1.0 * e + 0.4 * a - 0.9 * declining)))),
    "last_login_days_ago": clip(rng.exponential(np.exp(1.0 - 0.8 * e + 1.0 * declining)), 0, 45).astype(int),
})

engagement = pd.DataFrame({
    "student_id": ids,
    "events_attended": rng.poisson(np.exp(0.8 + 0.5 * e)),
    "clubs_joined": clip(rng.poisson(np.exp(-0.3 + 0.4 * e)), 0, 4),
    "hackathons": rng.poisson(np.exp(-0.7 + 0.4 * e + 0.3 * c)),
    "certifications": rng.poisson(np.exp(-0.2 + 0.5 * c)),
})

mocks = rng.poisson(np.exp(0.3 + 0.5 * c) * np.array([0.1, 0.4, 1.0, 1.6])[year - 1])
placement = pd.DataFrame({
    "student_id": ids,
    "aptitude_score": np.round(clip(58 + 13 * c + 4 * a + rng.normal(0, 7, N)), 1),
    "coding_score": np.round(clip(52 + 16 * c + rng.normal(0, 9, N)), 1),
    "mock_interviews_taken": mocks,
    "mock_interview_avg": np.where(mocks > 0, np.round(clip(55 + 14 * c + rng.normal(0, 8, N)), 1), np.nan),
})

skills = pd.DataFrame({
    "student_id": ids,
    "technical_score": np.round(clip(58 + 13 * c + 5 * a + rng.normal(0, 7, N)), 1),
    "soft_skill_score": np.round(clip(62 + 10 * e + rng.normal(0, 8, N)), 1),
})

feedback = pd.DataFrame({
    "student_id": ids,
    "satisfaction_1to5": clip(np.round(3.6 + 0.5 * e + rng.normal(0, .6, N)), 1, 5).astype(int),
    "faculty_feedback_1to5": np.round(clip(3.5 + 0.6 * a + 0.3 * e + rng.normal(0, .5, N), 1, 5), 1),
})

rows = []
for k, m in enumerate(MONTHS):
    back = len(MONTHS) - 1 - k  # months before now
    rows.append(pd.DataFrame({
        "student_id": ids, "month": m,
        "attendance_pct": np.round(clip(att_now + declining * 4.5 * back + rng.normal(0, 3, N), 20, 100), 1),
        "lms_logins": np.round(clip(lms.logins_per_week + declining * 1.0 * back + rng.normal(0, 1, N), 0, 20), 1),
        "test_score": np.round(clip(62 + 12 * a + declining * 2.5 * back + rng.normal(0, 6, N)), 1),
    }))
monthly = pd.concat(rows, ignore_index=True)


# ---------- make it messy, like real exports from 7 different campus systems ----------
def messy_ids(df, frac=0.06):
    df = df.copy()
    idx = rng.choice(df.index, int(len(df) * frac), replace=False)
    styles = [lambda s: s.lower(), lambda s: f" {s} ", lambda s: s.replace("STU", "STU-"),
              lambda s: s.replace("STU", "stu_")]
    df["student_id"] = df["student_id"].astype(object)
    for i in idx:
        df.at[i, "student_id"] = styles[rng.integers(len(styles))](df.at[i, "student_id"])
    return df


def drop_rows(df, frac):
    return df.drop(rng.choice(df.index, int(len(df) * frac), replace=False))


def add_dupes(df, frac=0.015):
    return pd.concat([df, df.sample(int(len(df) * frac), random_state=7)], ignore_index=True)


def blank(df, cols, frac=0.02):
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype(object)
        df.loc[rng.choice(df.index, int(len(df) * frac), replace=False), col] = np.nan
    return df


# academics: some CGPAs entered as percentage, a few blanks
pct = rng.choice(academics.index, 24, replace=False)
academics.loc[pct, "cgpa"] = np.round(academics.loc[pct, "cgpa"] * 9.5, 1)
academics = add_dupes(messy_ids(blank(academics, ["cgpa"], 0.01)))

# marks: text values where a number should be
subject_marks["marks"] = subject_marks["marks"].astype(object)
subject_marks.loc[rng.choice(subject_marks.index, 60, replace=False), "marks"] = rng.choice(["AB", "NA", "absent"], 60)
subject_marks = messy_ids(subject_marks, 0.03)

# attendance: attended > held in a few rows
bad = rng.choice(attendance.index, 40, replace=False)
attendance.loc[bad, "classes_attended"] = attendance.loc[bad, "classes_held"] + rng.integers(1, 6, 40)
attendance = messy_ids(attendance, 0.03)

lms = add_dupes(messy_ids(drop_rows(blank(lms, ["logins_per_week"]), 0.025)))
engagement = messy_ids(drop_rows(engagement, 0.03))
placement = add_dupes(messy_ids(drop_rows(blank(placement, ["aptitude_score", "coding_score"]), 0.03)))
skills = messy_ids(drop_rows(blank(skills, ["technical_score"]), 0.02))
feedback = messy_ids(drop_rows(feedback, 0.06))

for name, df in {"students": students, "academics": academics, "subject_marks": subject_marks,
                 "attendance": attendance, "lms_activity": lms, "engagement": engagement,
                 "placement": placement, "skills": skills, "feedback": feedback,
                 "monthly_trends": monthly}.items():
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:15s} {len(df):6d} rows")


# ---------- last year's cohort, where we already know how the year ended ----------
# Used only to train the early-warning model. poor_outcome = 1 means the student
# failed a course, dropped out or needed formal academic intervention.
H = 1500
ha = rng.normal(size=H)
he = 0.5 * ha + np.sqrt(0.75) * rng.normal(size=H)
hc = 0.35 * ha + 0.30 * he + 0.85 * rng.normal(size=H)
hd = rng.random(H) < 0.10

history = pd.DataFrame({
    "cgpa": np.round(clip(7.0 + 1.1 * ha + rng.normal(0, .4, H), 3.5, 10), 2),
    "backlogs": clip(rng.poisson(np.exp(-1.2 - 0.9 * ha)), 0, 8),
    "avg_marks": np.round(clip(62 + 12 * ha + rng.normal(0, 4, H)), 1),
    "attendance_pct": np.round(clip(78 + 9 * he + 3 * ha + rng.normal(0, 5, H) - 12 * hd, 30, 100), 1),
    "attendance_change_3m": np.round(-13.5 * hd + rng.normal(0, 4, H), 1),
    "assignment_completion": rng.binomial(20, 1 / (1 + np.exp(-(1.2 + 1.0 * he + 0.4 * ha - 0.9 * hd)))) * 5.0,
    "logins_per_week": np.round(clip(6 + 2.5 * he + rng.normal(0, 1.5, H) - 2.5 * hd, 0, 20), 1),
    "last_login_days_ago": clip(rng.exponential(np.exp(1.0 - 0.8 * he + 1.0 * hd)), 0, 45).astype(int),
    "aptitude_score": np.round(clip(58 + 13 * hc + 4 * ha + rng.normal(0, 7, H)), 1),
    "coding_score": np.round(clip(52 + 16 * hc + rng.normal(0, 9, H)), 1),
    "technical_score": np.round(clip(58 + 13 * hc + 5 * ha + rng.normal(0, 7, H)), 1),
    "soft_skill_score": np.round(clip(62 + 10 * he + rng.normal(0, 8, H)), 1),
    "events_attended": rng.poisson(np.exp(0.8 + 0.5 * he)),
    "satisfaction_1to5": clip(np.round(3.6 + 0.5 * he + rng.normal(0, .6, H)), 1, 5).astype(int),
})
# two outcomes, because a student can be academically fine and still not get placed
p_acad = 1 / (1 + np.exp(-(-2.3 - 1.2 * ha - 1.0 * he - 0.3 * hc + 1.4 * hd + rng.normal(0, .5, H))))
p_plac = 1 / (1 + np.exp(-(-0.9 - 1.3 * hc - 0.5 * ha - 0.3 * he + rng.normal(0, .6, H))))
history["poor_academic_outcome"] = (rng.random(H) < p_acad).astype(int)
history["poor_placement_outcome"] = (rng.random(H) < p_plac).astype(int)
history.to_csv(OUT / "historical_cohort.csv", index=False)
print(f"{'historical_cohort':15s} {len(history):6d} rows, {history.poor_academic_outcome.mean():.0%} poor academic, "
      f"{history.poor_placement_outcome.mean():.0%} poor placement")
