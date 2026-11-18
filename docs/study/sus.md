# SUS questionnaire (System Usability Scale)

Asked once per session, after the "Thank you!" screen and before the debriefing (about 2 minutes). Standard wording
(Brooke 1996); do not change it, so the score can be compared with other studies. The participant answers in a Google Form,
in a new tab on the study Mac after the interface shows "Thank you!".

**Instruction to the participant:** "Please answer these 10 short questions about the program you just used.
Choose one answer per line; go with your first impression."

Each item: 1 = Strongly disagree, 2, 3, 4, 5 = Strongly agree.

| # | Statement |
|---|-----------|
| 1 | I think that I would like to use this system frequently. |
| 2 | I found the system unnecessarily complex. |
| 3 | I thought the system was easy to use. |
| 4 | I think that I would need the support of a technical person to be able to use this system. |
| 5 | I found the various functions in this system were well integrated. |
| 6 | I thought there was too much inconsistency in this system. |
| 7 | I would imagine that most people would learn to use this system very quickly. |
| 8 | I found the system very cumbersome to use. |
| 9 | I felt very confident using the system. |
| 10 | I needed to learn a lot of things before I could get going with this system. |

The form also has a field **Participant ID** (filled in by the experimenter, e.g. P01) and an optional comment box
("Anything else about the program?"). No names.

## Storing the answers

In the form, each statement is a "Linear scale" question from 1 to 5 (labels: Strongly disagree / Strongly
agree), all required, in this order. The experimenter types the ID before handing over.

After each session (or once at the end, from the form's response sheet), make sure
`data/study/study_kits/sus.csv` has one line per participant (not in git; back it up with the sessions):

```
participant,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
P01,4,2,4,1,4,2,5,2,4,1
```

Comments go in the session log, not in the CSV.

## Score

`scripts/analyze_study.py` computes it (`study_sus.csv`, `study_sus_summary.csv`): odd items count answer − 1,
even items 5 − answer, the sum times 2.5, so 0–100. Only the analyzed participants are included. The score is
reported descriptively (mean with 95 % bootstrap interval, median); it is one score for the whole interface, not
split by condition. Around 68 is commonly read as an average score (Sauro 2011).
