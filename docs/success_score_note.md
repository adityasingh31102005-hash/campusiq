# Student Success Score: short note

## Purpose

One number (0-100) per student that tells faculty and administrators how well the student is positioned to succeed, built from every data source the campus holds, and simple enough that anyone can check how it was calculated.

## Indicators used

| Pillar | Weight | Indicators | How the pillar is scored (0-100) |
|---|---|---|---|
| Academic | 25% | CGPA, average subject marks, backlogs | 50% CGPA x 10 + 30% average marks + 20% backlog score (100 minus 25 per backlog) |
| Attendance | 20% | Subject-wise classes held and attended | Overall attendance %, scaled so 40% = 0 and 95% = 100 |
| Placement | 20% | Aptitude score, coding score, mock interviews | 35% aptitude + 35% coding + 30% mock interview average. If no mocks taken yet: 50% aptitude + 50% coding |
| LMS | 15% | Weekly logins, assignments submitted, days since last login | 50% assignment completion + 30% logins (10 per week = full marks) + 20% recency (3 days = full, 30 days = 0) |
| Skills | 10% | Technical assessment, soft-skill assessment | 60% technical + 40% soft skills |
| Engagement | 5% | Events, clubs, hackathons, certifications | 12 per event + 15 per club + 20 per hackathon + 15 per certification, capped at 100 |
| Feedback | 5% | Student satisfaction, faculty feedback | Both rescaled from 1-5 to 0-100, averaged |

## How they are combined

**Success Score = sum of (pillar score x pillar weight)**

Example: a student with Academic 80, Attendance 70, Placement 50, LMS 60, Skills 55, Engagement 40, Feedback 75 scores
20.0 + 14.0 + 10.0 + 9.0 + 5.5 + 2.0 + 3.75 = **64.3**.

## Why these weights

- Academic and Attendance carry the most weight (45% together) because they are the strongest and earliest signs of a student falling behind.
- Placement is 20% because employability is what success means at graduation, for the student and the institution.
- LMS activity is 15% because it is a leading indicator: logins and submissions fall weeks before marks do.
- Skills, Engagement and Feedback (20% together) complete the picture without letting softer signals dominate.

## Risk flag

The flag uses the score and six early-warning triggers, so a student with an acceptable overall score but one serious problem is still caught.

- **High:** score below 45, or two or more triggers
- **Medium:** score below 58, or exactly one trigger
- **Low:** everyone else

Triggers: attendance below 65%; 2 or more backlogs; CGPA below 5.5; under half of assignments submitted; no LMS login for 14+ days; attendance down 10+ points in 3 months.

## AI early-warning probability

Separately from the score, two logistic-regression models are trained on last year's cohort, where the year-end outcome is already known. Both use the same 14 indicators:

| Model | Predicts | AUC on held-out students |
|---|---|---|
| Academic risk | Failed course, dropout or formal academic intervention | 0.87 |
| Placement risk | No offer or not placement-ready by year end | 0.85 |

The brief asks for students at risk of poor academic performance *and* of poor placement outcomes, and these are genuinely different groups, so they are modelled and displayed separately.

Logistic regression was chosen over a stronger black-box model on purpose: its coefficients can be shown to faculty as the reasons behind each probability, which a college has to be able to defend.

The score tells faculty what is wrong today; the probabilities warn who is likely to slip next. Students who are Low risk by the rules but above 50% by either model are placed on a watchlist.

## Explainability

Because the score is a weighted sum, each student's score breaks down exactly into points earned per pillar. The dashboard shows, for every student, how many points each pillar adds or removes compared with the campus average, plus the triggers that caused the risk flag.

## Handling missing data

If a student has no record in a source, the missing values are filled with the median of their department and year, and the student record shows which sources were missing so the score is read with that in mind.

## Segmentation

Students are grouped by rule into seven segments that each need a different intervention: Sudden disengagers, Academically struggling, Strong academics with low placement readiness, Job-ready but low grades, All-round achievers, Quiet performers, and Steady middle.
