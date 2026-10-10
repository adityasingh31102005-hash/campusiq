"""CampusIQ - AI-Powered Student Analytics & Success Platform (HacXLerate 2026, KPMG challenge)
Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import (ACTIONS, PILLARS, REQUIRED_COLUMNS, SEGMENT_ACTION, TRIGGERS, WEIGHTS, build,  # noqa: E402
                      compute_pillars, explain, flag_risk, missing_columns)
import risk_model  # noqa: E402
from summary import student_summary  # noqa: E402

st.set_page_config(page_title="CampusIQ | Student Success Platform", page_icon="🎓", layout="wide")
RISK_COLORS = {"High": "#d64545", "Medium": "#e8a33d", "Low": "#3a9d6e"}
RISK_ORDER = ["High", "Medium", "Low"]


@st.cache_data(show_spinner="Integrating campus data...")
def load():
    return build()


uploads = st.session_state.setdefault("uploads", {})
R = build(uploads) if uploads else load()
full = R["master"]

# ------------------------------ sidebar ------------------------------
st.sidebar.title("🎓 CampusIQ")
st.sidebar.caption("Student Analytics & Success Platform")
page = st.sidebar.radio("View", ["Overview", "At-Risk Students", "Student 360", "AI Early Warning", "Segments",
                                 "Trends & Insights", "Data Integration", "How the Score Works"])
st.sidebar.divider()
depts = st.sidebar.multiselect("Department", sorted(full["department"].unique()))
years = st.sidebar.multiselect("Year", sorted(full["year"].unique()))
df = full
if depts:
    df = df[df["department"].isin(depts)]
if years:
    df = df[df["year"].isin(years)]
st.sidebar.caption(f"{len(df):,} of {len(full):,} students in view")
if uploads:
    st.sidebar.info("Using uploaded data for: " + ", ".join(uploads))
if df.empty:
    st.warning("No students match these filters.")
    st.stop()


def risk_pie(d):
    c = d["risk_flag"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
    c.columns = ["Risk", "Students"]
    fig = px.pie(c, names="Risk", values="Students", hole=0.55, color="Risk", color_discrete_map=RISK_COLORS)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10), height=300)
    return fig


# ------------------------------ pages ------------------------------
if page == "Overview":
    st.title("Campus Overview")
    st.caption("One view of every student, built from 8 integrated campus data sources.")
    high, med = (df["risk_flag"] == "High").sum(), (df["risk_flag"] == "Medium").sum()
    k = st.columns(5)
    k[0].metric("Students", f"{len(df):,}")
    k[1].metric("Avg Success Score", f"{df['success_score'].mean():.1f}",
                f"{df['success_score'].mean() - full['success_score'].mean():+.1f} vs campus")
    k[2].metric("High risk", f"{high}", f"{high / len(df):.0%} of students", delta_color="off")
    k[3].metric("Medium risk", f"{med}", f"{med / len(df):.0%} of students", delta_color="off")
    k[4].metric("Avg attendance", f"{df['attendance_pct'].mean():.1f}%")

    a, b = st.columns([1, 2])
    with a:
        st.subheader("Risk split")
        st.plotly_chart(risk_pie(df), width="stretch")
    with b:
        st.subheader("Success Score distribution")
        fig = px.histogram(df, x="success_score", color="risk_flag", nbins=40, color_discrete_map=RISK_COLORS,
                           category_orders={"risk_flag": RISK_ORDER},
                           labels={"success_score": "Success Score", "risk_flag": "Risk"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=300, bargap=0.05)
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Where is risk concentrated?")
        g = (df.assign(high=df["risk_flag"] == "High").groupby(["department", "year"])["high"].mean() * 100).round(0)
        heat = g.unstack("year")
        fig = px.imshow(heat, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                        labels=dict(x="Year", y="Department", color="% high risk"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Pillar health by department")
        p = df.groupby("department")[PILLARS].mean().round(0)
        fig = px.imshow(p, text_auto=True, color_continuous_scale="RdYlGn", zmin=30, zmax=90, aspect="auto",
                        labels=dict(color="Avg score"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, width="stretch")

    st.subheader("What needs attention")
    worst = g.idxmax()
    weakest = df[PILLARS].mean().idxmin()
    slip = (df["attendance_change_3m"] <= -10).sum()
    gap = (df["segment"] == "Strong academics, low placement readiness").sum()
    st.markdown(
        f"- **{worst[0]} Year {worst[1]}** has the highest share of high-risk students (**{g.max():.0f}%**).\n"
        f"- **{weakest}** is the weakest pillar in this view (avg **{df[weakest].mean():.0f}/100**).\n"
        f"- **{slip} students** lost 10+ attendance points in the last 3 months: early warning before grades fall.\n"
        f"- **{gap} students** have strong academics but low placement readiness: quick win for the placement cell.\n"
        f"- **{int(df['watchlist'].sum())} students** look fine by the rules but the early-warning model puts them above 50% risk.")

elif page == "At-Risk Students":
    st.title("At-Risk Students")
    st.caption("Prioritised intervention list, sorted worst first. Every flag comes with its reasons and a suggested action.")
    c = st.columns(3)
    levels = c[0].multiselect("Risk level", RISK_ORDER, default=["High"])
    reason = c[1].selectbox("Reason", ["Any"] + list(TRIGGERS))
    mentor = c[2].selectbox("Mentor", ["All"] + sorted(df["mentor"].unique()))
    v = df[df["risk_flag"].isin(levels)]
    if reason != "Any":
        v = v[v["risk_reasons"].str.contains(reason, regex=False)]
    if mentor != "All":
        v = v[v["mentor"] == mentor]
    v = v.sort_values("success_score")
    st.metric("Students in this list", len(v))
    cols = ["student_id", "name", "department", "year", "mentor", "success_score", "risk_flag",
            "academic_risk_probability", "placement_risk_probability", "risk_reasons", "weakest_pillar",
            "recommended_action"]
    st.dataframe(v[cols], width="stretch", hide_index=True, height=460, column_config={
        "success_score": st.column_config.ProgressColumn("Success Score", min_value=0, max_value=100, format="%.1f"),
        "academic_risk_probability": st.column_config.NumberColumn("Academic risk %", format="%.0f%%"),
        "placement_risk_probability": st.column_config.NumberColumn("Placement risk %", format="%.0f%%")})
    st.download_button("Download this list (CSV)", v[cols].to_csv(index=False), "at_risk_students.csv", "text/csv")
    st.subheader("Most common reasons in this list")
    rc = pd.Series({k: v["risk_reasons"].str.contains(k, regex=False).sum() for k in TRIGGERS}).sort_values()
    fig = px.bar(rc, orientation="h", labels={"value": "Students", "index": ""})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=10, b=10), height=300)
    st.plotly_chart(fig, width="stretch")

elif page == "Student 360":
    st.title("Student 360")
    opts = df.sort_values("success_score")
    labels = dict(zip(opts["student_id"], opts["student_id"] + " | " + opts["name"] + " | " + opts["risk_flag"] + " risk"))
    sid = st.selectbox("Select a student (sorted by lowest score first)", list(labels), format_func=labels.get)
    s = full.set_index("student_id").loc[sid]
    st.subheader(f"{s['name']}  ·  {s['department']} Year {s['year']}, Section {s['section']}  ·  Mentor: {s['mentor']}")
    st.caption(f"Segment: {s['segment']}")
    k = st.columns(4)
    k[0].metric("Success Score", f"{s['success_score']:.1f}", f"{s['success_score'] - full['success_score'].mean():+.1f} vs campus avg")
    k[1].metric("Academic risk", f"{s['academic_risk_probability']:.0f}%")
    k[2].metric("Placement risk", f"{s['placement_risk_probability']:.0f}%")
    k[3].metric("Data sources", f"{s['sources_available']}/8")
    if s["risk_reasons"]:
        st.error("**Why flagged:** " + s["risk_reasons"])
    note, next_steps = student_summary(s, full[PILLARS + ["success_score"]].mean(), PILLARS, ACTIONS)
    st.markdown("**Mentor summary** (written from this student's own numbers)")
    st.write(note)
    for kind in ("academic", "placement"):
        if s[f"{kind}_risk_probability"] >= 40:
            why = risk_model.top_reasons(R["models"][kind]["model"], s)
            if why:
                st.caption(f"Model's main reasons for {kind} risk: "
                           + ", ".join(f"{label} = {value:g}" for label, value in why))
    if s["missing_sources"]:
        st.info(f"No record in: {s['missing_sources']}. Those values were estimated from the department-year median.")

    ex = explain(s, full[PILLARS].mean())
    a, b = st.columns(2)
    with a:
        st.markdown("**What drives this score** (points gained or lost vs the average student)")
        ex["Effect"] = ex["vs cohort avg"].apply(lambda x: "Pulls score down" if x < 0 else "Lifts score up")
        fig = px.bar(ex.sort_values("vs cohort avg"), x="vs cohort avg", y="Pillar", orientation="h", color="Effect",
                     color_discrete_map={"Pulls score down": "#d64545", "Lifts score up": "#3a9d6e"}, text="vs cohort avg")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=330, xaxis_title="Points vs campus average", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Score breakdown**")
        st.dataframe(ex[["Pillar", "Pillar score", "Weight", "Points earned"]], hide_index=True, width="stretch")
        st.caption(f"Points earned add up to the Success Score: {ex['Points earned'].sum():.1f}")

    a, b = st.columns(2)
    with a:
        st.markdown("**6-month trend**")
        t = R["trends"][R["trends"]["student_id"] == sid].melt(["student_id", "month"], var_name="Metric")
        t = t[t["Metric"] != "lms_logins"]
        fig = px.line(t, x="month", y="value", color="Metric", markers=True)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=300, yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Subject-wise marks and attendance**")
        sub = R["subjects"][R["subjects"]["student_id"] == sid][["subject", "marks", "attendance_pct"]]
        st.dataframe(sub, hide_index=True, width="stretch")

    st.subheader("Recommended interventions")
    weak = ex.sort_values("vs cohort avg").head(3)
    for _, r in weak[weak["vs cohort avg"] < 0].iterrows():
        st.markdown(f"- **{r['Pillar']}** ({r['Pillar score']:.0f}/100): {ACTIONS[r['Pillar']]}")
    if (weak["vs cohort avg"] >= 0).all():
        st.markdown("- No pillar is below campus average. " + SEGMENT_ACTION[s["segment"]])

    st.subheader("What-if simulator")
    st.caption("Test an intervention before committing to it: how would the score and risk flag change?")
    w = st.columns(4)
    sim = s.copy()
    sim["attendance_pct"] = w[0].slider("Attendance %", 0, 100, int(s["attendance_pct"]))
    sim["assignment_completion"] = w[1].slider("Assignments submitted %", 0, 100, int(s["assignment_completion"]))
    sim["backlogs"] = w[2].slider("Backlogs", 0, 8, int(s["backlogs"]))
    sim["coding_score"] = w[3].slider("Coding score", 0, 100, int(s["coding_score"]))
    one = pd.DataFrame([sim])
    if sim["attendance_pct"] > s["attendance_pct"]:
        one["attendance_change_3m"] = 0
    out = flag_risk(compute_pillars(one)).iloc[0]
    new_acad = risk_model.predict(R["models"]["academic"]["model"], one)[0]
    new_plac = risk_model.predict(R["models"]["placement"]["model"], one)[0]
    r = st.columns(4)
    r[2].metric("Projected academic risk", f"{new_acad:.0f}%", f"{new_acad - s['academic_risk_probability']:+.0f} pts", delta_color="inverse")
    r[3].metric("Projected placement risk", f"{new_plac:.0f}%", f"{new_plac - s['placement_risk_probability']:+.0f} pts", delta_color="inverse")
    r[0].metric("Projected Success Score", f"{out['success_score']:.1f}", f"{out['success_score'] - s['success_score']:+.1f}")
    r[1].metric("Projected risk flag", out["risk_flag"], f"was {s['risk_flag']}", delta_color="off")

elif page == "AI Early Warning":
    st.title("AI Early Warning")
    st.caption("Two machine-learning models trained on last year's cohort, where the year-end outcome is already "
               "known. The brief asks for students at risk of poor academic performance and of poor placement "
               "outcomes, and those are different groups, so there is one model for each.")
    acc = {k: R["models"][k]["metrics"] for k in ("academic", "placement")}
    k = st.columns(4)
    k[0].metric("Past students trained on", f"{acc['academic']['trained_on']:,}")
    k[1].metric("Academic model AUC", f"{acc['academic']['auc']:.2f}", f"catches {acc['academic']['recall']:.0%} of cases", delta_color="off")
    k[2].metric("Placement model AUC", f"{acc['placement']['auc']:.2f}", f"catches {acc['placement']['recall']:.0%} of cases", delta_color="off")
    k[3].metric("On the watchlist", int(df["watchlist"].sum()))
    st.caption(f"Both tested on {acc['academic']['tested_on']} students the models never saw. An AUC of 0.5 would be guessing "
               f"and 1.0 would be perfect. The models point faculty at who to look at first; they do not decide anything.")

    a, b = st.columns(2)
    with a:
        st.subheader("Who is at risk of what")
        counts = df["ai_risk_type"].value_counts().reindex(
            ["Academic + placement", "Academic", "Placement", "None"]).fillna(0).reset_index()
        counts.columns = ["Risk type", "Students"]
        fig = px.bar(counts, x="Students", y="Risk type", orientation="h", text="Students",
                     color="Risk type", color_discrete_map={"Academic + placement": "#d64545", "Academic": "#e8a33d",
                                                            "Placement": "#4c78a8", "None": "#b8bec7"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=330, showlegend=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("A student can be heading for a good degree and still be unprepared for placement. Those two groups "
                   "need completely different help, which a single risk score would hide.")
    with b:
        st.subheader("Academic risk vs placement risk")
        fig = px.scatter(df, x="academic_risk_probability", y="placement_risk_probability", color="risk_flag",
                         color_discrete_map=RISK_COLORS, category_orders={"risk_flag": RISK_ORDER}, opacity=0.6,
                         hover_data=["student_id", "name", "department"],
                         labels={"academic_risk_probability": "Academic risk %",
                                 "placement_risk_probability": "Placement risk %", "risk_flag": "Rule-based flag"})
        fig.add_hline(y=50, line_dash="dot", line_color="#888888")
        fig.add_vline(x=50, line_dash="dot", line_color="#888888")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=330, legend=dict(orientation="h", y=-0.3, title=""))
        st.plotly_chart(fig, width="stretch")

    st.subheader("What each model learned")
    which = st.radio("Model", ["academic", "placement"], horizontal=True,
                     format_func=lambda x: f"{x.title()} risk", key="driver_model")
    st.caption("Outcome being predicted: " + acc[which]["outcome"] + f". {acc[which]['base_rate']:.0%} of last year's "
               f"cohort ended up in this group.")
    d = R["models"][which]["drivers"].copy()
    d["Direction"] = d["Effect"].apply(lambda x: "Raises risk" if x > 0 else "Lowers risk")
    fig = px.bar(d, x="Effect", y="Indicator", orientation="h", color="Direction",
                 color_discrete_map={"Raises risk": "#d64545", "Lowers risk": "#3a9d6e"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=420, yaxis_title="",
                      xaxis_title="Effect on risk when this indicator goes up")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Watchlist: students the rules miss")
    st.caption("Low risk by the rules, but a model gives them a 50%+ chance of a poor outcome. These are the earliest catches.")
    w = df[df["watchlist"]].sort_values("risk_probability", ascending=False)
    cols = ["student_id", "name", "department", "year", "mentor", "success_score", "ai_risk_type",
            "academic_risk_probability", "placement_risk_probability", "recommended_action"]
    if w.empty:
        st.success("No students on the watchlist in this view.")
    else:
        st.dataframe(w[cols], hide_index=True, width="stretch", height=320, column_config={
            "academic_risk_probability": st.column_config.NumberColumn("Academic risk %", format="%.0f%%"),
            "placement_risk_probability": st.column_config.NumberColumn("Placement risk %", format="%.0f%%")})
        st.download_button("Download watchlist (CSV)", w[cols].to_csv(index=False), "watchlist.csv", "text/csv")

elif page == "Segments":
    st.title("Student Segments")
    st.caption("Groups that need different interventions, not one-size-fits-all.")
    seg = df.groupby("segment").agg(Students=("student_id", "count"), Avg_score=("success_score", "mean"),
                                    High_risk=("risk_flag", lambda x: (x == "High").mean() * 100)).round(1).reset_index()
    seg["Suggested action"] = seg["segment"].map(SEGMENT_ACTION)
    seg = seg.sort_values("Students", ascending=False)
    a, b = st.columns([1, 1])
    with a:
        fig = px.bar(seg.sort_values("Students"), x="Students", y="segment", orientation="h", text="Students")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=380, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.scatter(df, x="Academic", y="Placement", color="segment", opacity=0.6, hover_data=["student_id", "name"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=380, legend=dict(orientation="h", y=-0.25, title=""))
        st.plotly_chart(fig, width="stretch")
    st.dataframe(seg.rename(columns={"segment": "Segment", "Avg_score": "Avg score", "High_risk": "% high risk"}),
                 hide_index=True, width="stretch")
    pick = st.selectbox("See students in a segment", seg["segment"])
    members = df[df["segment"] == pick].sort_values("success_score")
    cols = ["student_id", "name", "department", "year", "success_score", "risk_flag"] + PILLARS
    st.dataframe(members[cols], hide_index=True, width="stretch", height=320)
    st.download_button("Download segment (CSV)", members[cols].to_csv(index=False), "segment.csv", "text/csv")

elif page == "Trends & Insights":
    st.title("Trends & Insights")
    t = R["trends"].merge(df[["student_id", "risk_flag", "department"]], on="student_id")
    metric = st.radio("Metric", ["attendance_pct", "lms_logins", "test_score"], horizontal=True,
                      format_func={"attendance_pct": "Attendance %", "lms_logins": "LMS logins / week",
                                   "test_score": "Test score"}.get)
    a, b = st.columns(2)
    with a:
        st.markdown("**By risk level**: at-risk students show the drop months before the semester ends")
        g = t.groupby(["month", "risk_flag"])[metric].mean().reset_index()
        fig = px.line(g, x="month", y=metric, color="risk_flag", markers=True, color_discrete_map=RISK_COLORS,
                      category_orders={"risk_flag": RISK_ORDER})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=340, xaxis_title="", yaxis_title="", legend_title="Risk")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**By department**")
        g = t.groupby(["month", "department"])[metric].mean().reset_index()
        fig = px.line(g, x="month", y=metric, color="department", markers=True)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=340, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Insights only visible after combining sources")
    a, b = st.columns(2)
    with a:
        st.markdown("**Attendance + LMS together predict marks better than either alone**")
        fig = px.scatter(df, x="attendance_pct", y="assignment_completion", color="avg_marks",
                         color_continuous_scale="RdYlGn", opacity=0.7,
                         labels={"attendance_pct": "Attendance %", "assignment_completion": "Assignments submitted %",
                                 "avg_marks": "Avg marks"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10), height=340)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Which indicators move with the Success Score**")
        ind = ["cgpa", "attendance_pct", "assignment_completion", "logins_per_week", "aptitude_score", "coding_score",
               "technical_score", "soft_skill_score", "events_attended", "satisfaction_1to5", "backlogs"]
        corr = df[ind].corrwith(df["success_score"]).sort_values().round(2)
        fig = px.bar(corr, orientation="h", labels={"value": "Correlation with Success Score", "index": ""})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=10, b=10), height=340)
        st.plotly_chart(fig, width="stretch")
    lo = df[(df["attendance_pct"] < 70) & (df["assignment_completion"] < 60)]["avg_marks"].mean()
    hi = df[(df["attendance_pct"] >= 80) & (df["assignment_completion"] >= 80)]["avg_marks"].mean()
    st.info(f"Students low on both attendance and assignments average **{lo:.0f}** marks, against **{hi:.0f}** for "
            f"students high on both. Neither the attendance register nor the LMS shows this gap on its own.")

elif page == "Data Integration":
    st.title("Data Integration")
    st.caption("8 separate campus systems, cleaned and joined into one student record.")
    k = st.columns(4)
    k[0].metric("Sources integrated", len(R["sources"]))
    k[1].metric("Raw rows processed", f"{R['sources']['Raw rows'].sum():,}")
    k[2].metric("Data issues fixed", f"{R['report']['Records'].sum():,}")
    k[3].metric("Students with all 8 sources", f"{(full['sources_available'] == 8).mean():.0%}")
    with st.expander("Upload your own data", expanded=bool(uploads)):
        st.write("Replace any source with an export from your own campus system. The file is cleaned and "
                 "integrated the same way, and every page updates.")
        c1, c2 = st.columns([1, 2])
        target = c1.selectbox("Source to replace", list(REQUIRED_COLUMNS), key="upload_target")
        c1.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS[target]))
        up = c2.file_uploader("CSV file", type="csv", key="upload_file")
        if up is not None and c2.button("Use this file"):
            new = pd.read_csv(up)
            gone = missing_columns(target, new)
            if gone:
                c2.error("This file is missing columns: " + ", ".join(gone))
            else:
                uploads[target] = new
                st.rerun()
        if uploads and st.button("Reset to sample data"):
            uploads.clear()
            st.rerun()
    st.subheader("Source coverage")
    st.dataframe(R["sources"], hide_index=True, width="stretch", column_config={
        "Coverage %": st.column_config.ProgressColumn("Coverage %", min_value=0, max_value=100, format="%.1f%%")})
    st.subheader("Cleaning log")
    st.caption("Every fix the pipeline made, so nothing is changed silently.")
    st.dataframe(R["report"], hide_index=True, width="stretch")
    st.subheader("Integrated student record (sample)")
    st.dataframe(full.head(50), hide_index=True, width="stretch")
    st.download_button("Download full integrated dataset (CSV)", full.to_csv(index=False), "integrated_students.csv", "text/csv")

else:
    st.title("How the Success Score Works")
    st.markdown(
        "The **Student Success Score (0-100)** combines seven pillars. Each pillar is scored 0-100 from raw "
        "indicators, then multiplied by its weight. The logic is fully transparent: no black box.")
    w = pd.DataFrame({"Pillar": PILLARS, "Weight %": list(WEIGHTS.values()), "Indicators and formula": [
        "50% CGPA (x10) + 30% average subject marks + 20% backlog score (100 minus 25 per backlog)",
        "Overall attendance, scaled so 40% = 0 and 95% = 100",
        "35% aptitude + 35% coding + 30% mock interview average (50/50 aptitude and coding if no mocks yet)",
        "50% assignments submitted + 30% weekly logins (10+/week = full) + 20% login recency",
        "60% technical assessment + 40% soft-skill assessment",
        "Events, clubs, hackathons and certifications, capped at 100",
        "50% student satisfaction + 50% faculty feedback, rescaled from 1-5",
    ]})
    a, b = st.columns([1, 2])
    with a:
        fig = px.pie(w, names="Pillar", values="Weight %", hole=0.5)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=False)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.dataframe(w, hide_index=True, width="stretch")
    st.subheader("Risk flag rules")
    st.markdown(
        "A student is flagged from the score **and** from early-warning triggers, so a student with a decent "
        "overall score but one serious problem is not missed.\n\n"
        "- **High:** Success Score below 45, or 2+ triggers\n"
        "- **Medium:** Success Score below 58, or exactly 1 trigger\n"
        "- **Low:** everyone else\n\n**Triggers:**\n" + "\n".join(f"- {t}" for t in TRIGGERS))
    st.subheader("Where the AI comes in")
    st.markdown(
        "The Success Score and flags above are transparent rules. Alongside them, two logistic-regression models "
        "trained on last year's cohort estimate each student's chance of a poor **academic** outcome and of a poor "
        "**placement** outcome. Rules explain *what is wrong today*; the models warn about *who is likely to slip "
        "next*, and separate the two kinds of risk the brief asks for. See the **AI Early Warning** page.")
    st.subheader("Why these weights")
    st.markdown(
        "- **Academic (25) and Attendance (20)** are the strongest, earliest signals of a student falling behind.\n"
        "- **Placement (20)** reflects what success means to students and the institution at graduation.\n"
        "- **LMS (15)** is a leading indicator: activity drops weeks before marks do.\n"
        "- **Skills, Engagement, Feedback (20 together)** round out the picture without letting soft signals dominate.")
