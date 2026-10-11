# CampusIQ

A student success dashboard built for HacXLerate 2026, Round 1, for the KPMG problem statement
"AI-Powered Student Analytics and Success Platform".

**Live demo:** https://campusiq-74yntvtpyggp7bsmmshwa8.streamlit.app

## The problem we went after

A college already records everything it needs to spot a struggling student. Attendance is in one
register, marks in another, LMS logins somewhere else, placement practice in a fourth place. Nobody
reads them together, so the warning only arrives with the result sheet, when the semester is over.

CampusIQ pulls those sources into one record per student, scores them, flags who is in trouble, and
says why. A mentor opens one page instead of chasing five systems.

## What it does

The dashboard has eight views. **Overview** gives the campus picture: how many students are at risk,
which department and year they sit in, which pillar is weakest. **At-Risk Students** is the working
list, sorted worst first, with the reasons each flag fired and a suggested action, exportable to CSV
because the next step happens in a mentor meeting.

**Student 360** is where you land on one student: the triggers that flagged them, a chart of which
pillars pull their score up or down, a short written summary, their six-month trend, subject-wise
marks, and a what-if simulator that projects what an intervention would actually change.

**AI Early Warning** holds the two prediction models. **Segments** groups students into seven types
that each need different help. **Trends & Insights** shows what only the combined data reveals.
**Data Integration** shows every cleaning fix we made and lets a college upload its own CSV in place
of ours. **How the Score Works** publishes the entire formula.

## The data

We generate a synthetic campus of 1,200 students across six departments, because no college hands
out real student records for a hackathon. The generator writes eight source files covering every
category the brief listed: academics, subject marks, attendance, LMS activity, engagement,
placement, skills and feedback, plus six months of monthly trends and last year's graduating cohort
with their known outcomes.

We made the files messy on purpose, the way real exports are. Student IDs appear four different ways
(`STU0012`, `stu-0012`, `stu_0012`, with stray spaces). Some rows are duplicated. A few CGPAs were
typed as percentages. Marks columns contain `AB` and `absent`. Some attendance records claim more
classes attended than were held. Whole students are missing from some systems.

The pipeline fixes each of these and writes what it did to the cleaning log on screen, so nothing
changes silently. Students missing from a source get the median for their department and year, and
their record says which sources were missing.

## The Student Success Score

Seven pillars, each scored 0-100 from raw indicators, then weighted:

- Academic, 25 percent, from CGPA, average marks and backlogs
- Attendance, 20 percent
- Placement, 20 percent, from aptitude, coding and mock interviews
- LMS, 15 percent, from assignments, login frequency and recency
- Skills, 10 percent
- Engagement, 5 percent
- Feedback, 5 percent

A student is High risk below 45 or with two or more early-warning triggers, Medium below 58 or with
one trigger, Low otherwise. The six triggers are attendance under 65 percent, two or more backlogs,
CGPA under 5.5, fewer than half the assignments submitted, no LMS login in 14 days, and attendance
down ten points over three months.

Weights and triggers are in `src/pipeline.py` and explained in `docs/success_score_note.md`. A
college can disagree with a weight and change one line.

## Where the AI is

The score is deliberately rule-based so faculty can check it. The prediction sits next to it.

We train two logistic regressions on last year's cohort, where the outcome is already known. One
predicts a poor academic outcome, a failed course or dropout or formal intervention, and reaches an
AUC of 0.87 on students it never saw. The other predicts a poor placement outcome and reaches 0.85.

Two models rather than one, because they are not the same students. A student can be heading for a
good degree and still be nowhere near placement-ready, and those two groups need completely
different help.

We chose logistic regression over something stronger on purpose. The coefficients can be shown to
faculty as the reason behind each probability, and a college has to be able to defend a flag to the
student it is about.

The sharpest result: 143 students are Low risk by every rule, yet a model puts them above 50
percent. A threshold report would miss all of them. They get their own watchlist.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the data: `python src/generate_data.py`

## What is in here

```
app.py                  the dashboard, all eight views
src/pipeline.py         cleaning, integration, scoring, risk flags, segments
src/risk_model.py       the two early-warning models
src/summary.py          the written mentor summary
src/generate_data.py    the synthetic campus
data/                   the source CSV files
docs/                   Success Score note and slides
```

Built with Python, pandas, scikit-learn, Plotly and Streamlit.

## For evaluators

No login needed. Start on Overview, then open Student 360, where the list is sorted lowest score
first, so the first student shows everything the page can do.

Deliverables: the dashboard is the live link above, the Success Score note is
`docs/success_score_note.md`, and the slides are `docs/CampusIQ_Deck.pdf`.

All data is synthetic. No real student information is used anywhere in this project.
