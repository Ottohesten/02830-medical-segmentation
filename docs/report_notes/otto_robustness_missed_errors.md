# Report notes: Robustness, missed errors, the locked test on fresh scans

| | |
|---|---|
| **Owner** | Otto |
| **Branch** | `otto/robustness` (merged), `otto/missed-errors` |
| **Where the details are** | [docs/tasks/g1_robustness.md](../tasks/g1_robustness.md), [docs/tasks/g1_missed_errors.md](../tasks/g1_missed_errors.md), `results/kits_robustness_*/`, `results/kits_missed_*/` (on the branch) |
| **Last updated** | *(date)* |

How to fill this in: [README.md](README.md). Notes, not report text. Every number with its interval and file path.

## 1. What was done

*(The question in one sentence, then the design in about 5 lines: data, method, what was compared with what, what
was fixed before the results were seen.)*

## 2. Results

*(One line per result: the number, its interval, the file it comes from, and whether it is locked/fixed beforehand
or exploratory.)*

## 3. What it means, and what it does not show

*(Your reading of the results in plain words. Which conclusions do the numbers support, and which do they not?)*

## 4. Limitations

*(What could make the result wrong or narrower than it looks: sample size, one machine, one seed, simulated instead
of real, choices that could have gone another way.)*

## 5. Decisions and rejected alternatives

*(Date, what was chosen, what was rejected, why. Only the ones a reader of the report or an examiner would ask
about.)*

## 6. What went wrong, and surprises

*(Problems, dead ends and things that turned out differently than expected, and what was learned from them.)*

## 7. Figures and tables for the report

*(Path of each candidate figure or table, one line on what it shows, and which one you would pick if there is room
for only one.)*

## 8. Top 3 points for the report

1.
2.
3.

## 9. Contribution

*(What you did yourself: code, experiments, analyses, documents. Short list.)*

## 10. Specific to this part

- Robustness: how the degradation and its strength were chosen, and what happens to the queue when scans get worse.
- Missed errors: which kinds of confident errors the score cannot see, and what was tried to catch them.
- The locked test on fresh scans: the improvement criterion that was fixed beforehand, and why neither candidate
  met it. What the test says about how well the G1 score carries over to new scans.
