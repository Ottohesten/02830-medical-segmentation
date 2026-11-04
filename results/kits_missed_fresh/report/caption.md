# Figure caption — fresh ranking test

**Fresh-pool test of two predeclared ranking comparisons.** On the next 100 KiTS23 cases of the original
seed-42 draw, 10 cases have 3 mm Dice < 0.83. **A:** Review-AUC differences to locked G1 (0.921) with
95% paired case-bootstrap intervals (2,000 resamples, seed 42). The selected combination averages the
G1, missing-side and hull-deficit suspicion ranks; far disagreement counts disagreement more than 6 mm
from the 3 mm edge, normalized by predicted volume. Both intervals include zero, so neither meets the
primary improvement criterion. These are named per-comparison intervals, without a familywise claim.
**B:** Expected bad-case capture within 20 reviews, using fair random breaking of score ties: G1 and
the combination find 9/10; far disagreement finds 10/10. Bars are descriptive fixed-budget points.

The combination has a small positive prespecified secondary quality-AUC difference (+0.001075,
unadjusted 95% CI [0.000079, 0.001878]); it does not replace the primary result. Quality curves simulate
perfect corrections and do not measure actual human correction time. All scores were frozen before GT;
both questions ran in one outcome stage. The two previously used G3 smoke cases remain included, as
disclosed in the locked protocol. Original released models and unchanged G1 source results were used.

Source values: `figure_source_data.csv`; full results: `../fresh_summary.md`. PDF and PNG contain aggregate
statistics only, with no CT, label-map or model-output imagery.
