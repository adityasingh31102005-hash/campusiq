# CampusIQ: AI-Powered Student Analytics & Success Platform

HacXLerate 2026, Round 1. Problem statement: **AI-Powered Student Analytics and Success Platform (KPMG in India)**.

CampusIQ joins eight separate campus data sources into one record per student, computes a transparent **Student Success Score**, flags at-risk students with the reasons, and tells faculty what to do next.

**Live demo:** _add your deployment link here_

## What it does

| Requirement | Where in the app |
|---|---|
| Data integration across sources | **Data Integration** page: source coverage, cleaning log, integrated dataset download |
| Student Success Score | Every page; formula on **How the Score Works** |
| Risk flags | **At-Risk Students**: prioritised list with reasons and recommended action |
| Key trends | **Trends & Insights**: 6-month attendance, LMS and test trends by risk level and department |
| Relevant student insights | **Overview** "What needs attention" and **Student 360** |
| Bonus: Student segmentation | **Segments** page: 7 segments, each with a targeted intervention |
| Bonus: Explainable score | **Student 360**: which pillars lift or pull down each student's score, and why they were flagged |
| At-risk: academic and placement | **AI Early Warning**: two models trained on last year's cohort, one per outcome the brief names, plus a watchlist of students the rules miss |
| Mentor summary | **Student 360**: an auto-generated plain-language note on each student with next steps |
| Bring your own data | **Data Integration**: upload a CSV to replace any source; it is validated, cleaned and integrated |
| What-if simulator | **Student 360**: test an intervention and see the projected score, risk flag and risk probability |

## Data

`src/generate_data.py` creates a synthetic campus of 1,200 students across 6 departments, as 8 source files plus a monthly trend file, covering every suggested category:

| File | Category | Fields |
|---|---|---|
| `students.csv` | Master | department, year, section, mentor |
| `academics.csv`, `subject_marks.csv` | Academic | CGPA, backlogs, subject marks |
| `attendance.csv` | Attendance | subject-wise classes held and attended |
| `lms_activity.csv` | LMS | logins per week, assignments submitted, last login |
| `engagement.csv` | Engagement | events, clubs, hackathons, certifications |
| `placement.csv` | Placement | aptitude, coding, mock interviews |
| `skills.csv` | Skills | technical and soft-skill assessments |
| `feedback.csv` | Feedback | student satisfaction, faculty feedback |
| `monthly_trends.csv` | Trends | 6 months of attendance, LMS logins, test scores |
| `historical_cohort.csv` | Model training | Last year's 1,500 students with their known academic and placement outcomes |

The files are deliberately messy, like real exports: inconsistent ID formats (`stu-0012`, ` STU0012 `), duplicate rows, CGPA typed as a percentage, text such as `AB` in marks, attendance above classes held, and students missing from some systems. The pipeline fixes each issue and logs it.

## How the score works

Seven pillars, each scored 0-100, combined with fixed weights:

Academic 25 · Attendance 20 · Placement 20 · LMS 15 · Skills 10 · Engagement 5 · Feedback 5

Risk flag: **High** if the score is below 45 or 2+ early-warning triggers fire; **Medium** if below 58 or 1 trigger; otherwise **Low**. Full detail in [`docs/success_score_note.md`](docs/success_score_note.md).

## Where the AI is

The score and flags are transparent rules, so faculty can check them. Next to them, two logistic-regression models are trained on last year's cohort (1,500 students whose year-end outcome is known) and give every current student a probability of:

- a poor **academic** outcome (failed course, dropout, formal intervention) — AUC 0.87 on held-out students
- a poor **placement** outcome (no offer or not placement-ready) — AUC 0.85

Keeping them separate matters: a student can be heading for a good degree and still be unprepared for placement, and those two groups need different help. Students who are Low risk by the rules but above 50% by either model go on a watchlist, which is where the earliest catches are.

## Tech stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit

```
app.py                  Streamlit dashboard (8 views)
src/generate_data.py    Synthetic multi-source data generator
src/pipeline.py         Cleaning, integration, scoring, risk flags, segments, explainability
src/risk_model.py       Early-warning models (logistic regression, one per outcome)
src/summary.py          Auto-generated mentor summary
data/                   Source CSV files
docs/                   Success Score note
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the data: `python src/generate_data.py`

## Notes for evaluators

No login is needed. Use the sidebar to switch views and filter by department and year. On **Student 360**, the student list is sorted with the lowest scores first, so the first few students show the risk reasons, explanation chart and simulator best.

All data is synthetic. No real student information is used.
