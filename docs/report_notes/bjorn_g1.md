# Report notes: G1 ranking and heatmap, the review interface, study design

| | |
|---|---|
| **Owner** | Bjørn |
| **Branch** | `main` |
| **Where the details are** | [PROJECT_LOG.md](../../PROJECT_LOG.md), `results/kits100/`, `ui/`, [docs/study/](../study/) |
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

- Why the dataset changed from MSD Spleen to KiTS23, and why the 3 mm vs. 6 mm disagreement stands in for an ensemble.
- What locking the choices on the development set cost and gave (e.g. the entropy variant that did better on test).
- What the exploratory 3 mm vs. 6 mm comparison showed (grid effect near the edge vs. real differences).
- How the review interface changed after the first hands-on tests, and why.
