"""Early-warning model: learns from last year's cohort which students ended the year badly,
then gives every current student a probability of the same thing happening."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"

FEATURES = ["cgpa", "backlogs", "avg_marks", "attendance_pct", "attendance_change_3m",
            "assignment_completion", "logins_per_week", "last_login_days_ago", "aptitude_score",
            "coding_score", "technical_score", "soft_skill_score", "events_attended", "satisfaction_1to5"]

LABELS = {
    "cgpa": "CGPA", "backlogs": "Backlogs", "avg_marks": "Average marks", "attendance_pct": "Attendance %",
    "attendance_change_3m": "Attendance change (3 months)", "assignment_completion": "Assignments submitted %",
    "logins_per_week": "LMS logins per week", "last_login_days_ago": "Days since last LMS login",
    "aptitude_score": "Aptitude score", "coding_score": "Coding score", "technical_score": "Technical skills",
    "soft_skill_score": "Soft skills", "events_attended": "Events attended", "satisfaction_1to5": "Student satisfaction",
}


TARGETS = {
    "academic": ("poor_academic_outcome", "failed a course, dropped out or needed formal academic intervention"),
    "placement": ("poor_placement_outcome", "finished the year without a job offer or placement-eligible readiness"),
}


def _fit(hist, target):
    X, y = hist[FEATURES], hist[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    # logistic regression on purpose: the coefficients can be shown to faculty as reasons
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]
    pred = prob >= 0.5
    metrics = {
        "trained_on": len(X_tr), "tested_on": len(X_te), "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y_te, prob)),
        "recall": float(recall_score(y_te, pred)), "precision": float(precision_score(y_te, pred)),
    }
    coef = model.named_steps["logisticregression"].coef_[0]
    drivers = (pd.DataFrame({"Indicator": [LABELS[f] for f in FEATURES], "feature": FEATURES, "Effect": coef.round(2)})
               .sort_values("Effect"))
    return model, metrics, drivers


def train():
    """One model per outcome the brief asks about: academic risk and placement risk."""
    hist = pd.read_csv(DATA / "historical_cohort.csv")
    out = {}
    for kind, (target, description) in TARGETS.items():
        model, metrics, drivers = _fit(hist, target)
        metrics["outcome"] = description
        out[kind] = {"model": model, "metrics": metrics, "drivers": drivers}
    return out


def predict(model, df):
    return (model.predict_proba(df[FEATURES])[:, 1] * 100).round(1)


def top_reasons(model, row, n=3):
    """Which indicators push this one student's probability up the most."""
    scaler = model.named_steps["standardscaler"]
    coef = model.named_steps["logisticregression"].coef_[0]
    z = (row[FEATURES].astype(float).values - scaler.mean_) / scaler.scale_
    push = pd.Series(z * coef, index=FEATURES).sort_values(ascending=False)
    return [(LABELS[f], row[f]) for f in push.index[:n] if push[f] > 0]
