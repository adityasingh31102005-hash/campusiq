# Demo walkthrough and Q&A prep

## 3-minute demo order

1. **Overview** (30s). "A college records all of this already, but in separate systems. This is 1,200 students in one view. 211 are high risk, and risk is concentrated in CE Year 2."
2. **At-Risk Students** (30s). "Sorted worst first. Every row carries the reasons it was flagged and a suggested action. A mentor can filter to only their own students and export the list."
3. **Student 360** (60s). Pick the first student. "Here is why he was flagged: six triggers fired. This chart shows which pillars pull his score down: attendance costs him 13 points against the campus average. This summary is written automatically. And here is the what-if: raise attendance to 95% and the score moves from 19 to 39, with academic risk falling."
4. **AI Early Warning** (45s). "The rules describe today. These two models predict the year end, trained on last year's cohort where we know the outcome. Academic risk and placement risk are separate because they are different groups. 143 students are low risk by every rule but above 50% by a model: those are the catches a threshold report misses."
5. **Data Integration** (15s). "Eight sources, every cleaning fix logged, and a college can upload its own CSV in place of ours."

## Questions to expect

**Why synthetic data?**
No college can release real student records for a hackathon. The generator reproduces the structure and the mess of real exports deliberately, and the dashboard accepts an uploaded CSV so the same pipeline runs on real data unchanged.

**How did you choose the weights?**
Academic and attendance carry the most weight because they are the earliest and strongest signals of a student falling behind. Placement is 20% because that is what success means at graduation. LMS is a leading indicator: activity drops weeks before marks do. The rest round out the picture without letting soft signals dominate. All of it is published on the How the Score Works page, so a college can argue with a weight and change it.

**Why logistic regression and not something stronger?**
Because a college has to be able to defend a flag to a student. The coefficients are shown directly as the reasons behind each probability. A gradient-boosted model would gain a little accuracy and lose the explanation, which is a bad trade here.

**Is 0.87 AUC good?**
0.5 is guessing, 1.0 is perfect. It is measured on 375 students the model never saw. Precision is lower than recall on purpose: this is a prompt to look closer, not a verdict, so missing an at-risk student costs more than a false alarm.

**What is actually AI here?**
Two trained models, plus explainability on top of them. The score itself is rules, deliberately, so faculty can check it. The brief asks to go beyond a dashboard and convert data into decision intelligence, and the split between what is wrong today and who slips next is how we did that.

**What would you build next?**
Live ERP and LMS connectors, retraining each semester as real outcomes arrive, a mentor workflow that records what intervention was tried and whether it worked, and fairness checks by department and background before any college acts on a flag.
