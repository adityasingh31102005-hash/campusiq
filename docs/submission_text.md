# Submission form text

## Project title
CampusIQ: Student Success Analytics and Early Warning Platform

## Project summary (3 lines)
CampusIQ brings eight disconnected campus data sources into one record per student and turns them into a Student Success Score, a risk flag with its reasons, and a suggested next step.
Two machine-learning models trained on last year's cohort separately predict academic risk and placement risk, catching students no rule-based threshold would flag.
Faculty get an interactive dashboard that answers who needs help, why, and what to do about it.

## Solution description (~150 words)
Colleges already record attendance, marks, LMS activity, placement practice, extracurriculars and feedback, but each sits in its own system, so nothing adds up to a warning until results are published.

CampusIQ integrates eight source files into one student view, handling the problems real exports carry: inconsistent ID formats, duplicate rows, CGPA entered as a percentage, text in numeric columns, attendance exceeding classes held, and students missing from whole systems. Every fix is logged on screen.

From that unified view it computes a Student Success Score from seven weighted pillars, flags risk using the score plus six early-warning triggers, and groups students into seven segments that each need a different intervention. Alongside the rules, two logistic-regression models trained on last year's cohort predict poor academic outcomes (AUC 0.87) and poor placement outcomes (AUC 0.85), surfacing 143 students the rules call low risk.

Every flag shows its drivers, so faculty can check the reasoning before acting.

## Notes for evaluators
- No login is required. Use the sidebar to switch views and filter by department and year.
- Start on **Overview**, then open **Student 360**: the student list is sorted lowest score first, so the first few students show the risk reasons, the score drivers, the auto-written mentor summary and the what-if simulator.
- **AI Early Warning** holds the two predictive models, their accuracy on held-out students, what each one learned, and the watchlist.
- **Data Integration** shows source coverage and the full cleaning log, and accepts an uploaded CSV in place of any source file.
- **How the Score Works** publishes the complete scoring formula and risk rules.
- All data is synthetic. No real student records are used.
