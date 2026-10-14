"""Analysis of the user study (G2): does the uncertainty heatmap make human correction more efficient?

This module carries out the analysis plan in docs/study/analysis_plan.md (locked before the first participant).
If the two ever disagree, the plan is what was promised; the code must be fixed.

Per corrected study scan (from sessions/<participant>/<case>_mask.nii.gz and _log.json):
- Dice before: the model's mask against the ground truth (what the participant started from);
- Dice after:  the participant's saved mask against the ground truth;
- minutes:     the time on the scan, from the log (until "Done", or the time limit);
- gain per minute = (Dice after - Dice before) / minutes. This is the PRIMARY metric ("Dice per unit of human
  effort"). With the time limit it stays defined: the gain is measured at "Done" or when the time ran out.
  Scans where the time ran out and scans the participant did not change (gain 0) are kept as they are.

Per participant: the mean over their scans WITH the heatmap and over their scans WITHOUT it (3 + 3). The
comparison is a Wilcoxon signed-rank test on the per-participant differences (with - without), two-sided.
Pairing per participant takes out that some people are simply faster or more careful than others. Because every
scan is corrected with the heatmap by half of the participants and without it by the other half, differences
between scans cancel out as well (see study.py for the balancing).

Hypotheses (see HYPOTHESES): H1 (primary) the gain per minute is higher with the heatmap; H2 (key secondary) the
raw NASA-TLX workload is lower with it. H2 is only tested if H1 is supported ("fixed sequence"). Testing in a fixed
order keeps the overall chance of a false positive at alpha without making the primary test stricter. Every other
comparison is exploratory: its p-value is reported, but no conclusion is drawn from it.

Which data count: only real participant ids (P01, ...). Test ids (T01, ...) and practice scans are left out.
Only COMPLETE sessions count (every study scan saved); other participants are listed in study_exclusions.csv.
All participants must have seen the heatmap shown the same way (study.heatmap_display, saved by the server in
every scan log); the analysis stops if the display differs between scans, because then "with heatmap" would mean
different things for different participants.
The ground truth is used here, in the evaluation, and never shown to the participants in the study scans.

Simulated sessions (made only by the tests, logs with "simulated": true) are never mixed with real ones: the
output file names start with SIMULATED_, every table has simulated = True, and they cannot be written to results/.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from segreview.config import REPO_ROOT
from segreview.data import list_cases, load_organ_mask
from segreview.heatmap_display import check_display, display_label
from segreview.metrics import dice
from segreview.study import TLX_SCALES, WITH, WITHOUT, assignment, id_pattern, id_prefixes, load_manifest, study_dir

OUTCOMES = ["gain_per_min", "dice_after", "dice_gain", "minutes"]     # per participant and condition
PRIMARY = "gain_per_min"
# The hypotheses of the analysis plan: outcome -> (name, role, expected sign of the difference with - without).
HYPOTHESES = {"gain_per_min": ("H1", "primary", +1), "raw_tlx": ("H2", "key secondary", -1)}
EXACT_MAX_PAIRS = 24          # up to this many non-zero differences the p-value uses all 2^n sign patterns
RANDOM_SIGN_PATTERNS = 2 ** 20   # above that: this many random sign patterns (seeded)
SIMULATED_PREFIX = "SIMULATED_"


def participant_folders(study: dict, sessions_dir: Path) -> list[Path]:
    """Session folders of real participants only (P01, ...), sorted. Test ids and anything else are skipped."""
    pattern = id_pattern(id_prefixes(study)[0])
    if not sessions_dir.is_dir():
        return []
    return sorted(f for f in sessions_dir.iterdir() if f.is_dir() and pattern.fullmatch(f.name))


class TruthCache:
    """Ground-truth organ masks of the study scans, cut to the same slices as the viewer files (loaded once each)."""

    def __init__(self, study: dict, source: dict):
        self.facts = load_manifest(study)["facts"]["scans"]
        self.cases = {c.case_id: c for c in list_cases(source)}
        self.source = source
        self.cache = {}

    def get(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if case_id not in self.cache:
            truth, ignore = load_organ_mask(self.source, self.cases[case_id])
            z0, z1 = self.facts[case_id]["crop_z"]
            self.cache[case_id] = (truth[:, :, z0:z1], ignore[:, :, z0:z1])
        return self.cache[case_id]


def _mask(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj) > 0


def simulated_flags(study: dict, sessions_dir: Path) -> set[bool]:
    """Which kinds of sessions the folder holds: {False} real, {True} simulated, both = mixed (not allowed)."""
    flags = set()
    for folder in participant_folders(study, sessions_dir):
        for path in list(folder.glob("*_log.json")) + list(folder.glob("tlx_*.json")):
            flags.add(bool(json.loads(path.read_text()).get("simulated", False)))
    return flags


def scan_table(study: dict, source: dict, sessions_dir: Path, truths: TruthCache | None = None) -> pd.DataFrame:
    """One row per corrected study scan of a real participant (see the module docstring for the columns).

    Also: timeout (the time ran out), changed_voxels (voxels where the saved mask differs from the model's mask)
    and unchanged (the participant saved the model's mask as it was).
    The condition in each log is checked against the balancing plan, so a mix-up cannot go unnoticed.
    truths: a TruthCache to reuse (optional; the ground truth is then loaded only once per scan).
    """
    root = study_dir(study)
    manifest = load_manifest(study)
    truths = truths or TruthCache(study, source)
    rows = []
    for folder in participant_folders(study, sessions_dir):
        pid = folder.name
        plan = {a["case_id"]: a for a in assignment(pid, manifest["study_scans"], id_prefixes(study)[0])}
        for log_path in sorted(folder.glob("*_log.json")):
            log = json.loads(log_path.read_text())
            case_id = log_path.name.removesuffix("_log.json")
            if case_id not in plan:
                raise ValueError(f"{log_path}: {case_id} is not a study scan")
            if log["condition"] != plan[case_id]["condition"]:
                raise ValueError(f"{log_path}: condition {log['condition']} differs from the plan "
                                 f"({plan[case_id]['condition']})")
            if "heatmap_display" not in log:
                raise ValueError(f"{log_path}: no heatmap_display in the log (saved by an older server version?)")
            truth, ignore = truths.get(case_id)
            before = _mask(root / "scans" / case_id / "mask.nii.gz")
            after = _mask(folder / f"{case_id}_mask.nii.gz")
            if after.shape != truth.shape or before.shape != truth.shape:
                raise ValueError(f"{pid}/{case_id}: mask shape {after.shape} differs from the ground truth {truth.shape}")
            minutes = log["duration_s"] / 60.0
            d_before, d_after = dice(before, truth, ignore), dice(after, truth, ignore)
            changed = int((after != before).sum())
            rows.append({"participant": pid, "case_id": case_id, "condition": log["condition"],
                         "position": plan[case_id]["position"], "reason": log["reason"],
                         "timeout": log["reason"] == "timeout", "duration_s": log["duration_s"], "minutes": minutes,
                         "dice_before": d_before, "dice_after": d_after, "dice_gain": d_after - d_before,
                         "gain_per_min": (d_after - d_before) / minutes,
                         "changed_voxels": changed, "unchanged": changed == 0,
                         "strokes": log.get("strokes"), "undo_count": log.get("undo_count"),
                         "heatmap_visible_s": log.get("heatmap_visible_s"),
                         "heatmap_display": display_label(check_display(log["heatmap_display"])),
                         "simulated": bool(log.get("simulated", False))})
    return pd.DataFrame(rows)


def mark_complete(scans: pd.DataFrame, n_study_scans: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only complete sessions: a participant counts if every study scan was saved (both conditions in full).

    A participant who stopped early is left out (the session guide replaces them by the next participant with the
    same id). Leaving out only their missing scans would compare means over different numbers of scans.
    Output: (scans with an 'included' column, exclusions table: participant, n_scans, reason).
    """
    counts = scans.groupby("participant").size()
    incomplete = counts[counts != n_study_scans]
    scans = scans.assign(included=~scans.participant.isin(incomplete.index))
    exclusions = pd.DataFrame({"participant": incomplete.index, "n_scans": incomplete.values,
                               "reason": [f"incomplete session: {n} of {n_study_scans} study scans saved"
                                          for n in incomplete.values]})
    return scans, exclusions


def participant_table(scans: pd.DataFrame) -> pd.DataFrame:
    """Per participant: mean of each outcome over the scans WITH and WITHOUT the heatmap, and the difference.

    Columns: participant, heatmap_first (the heatmap block came first), n_with, n_without, and for each outcome
    <outcome>_with, <outcome>_without, <outcome>_diff (= with - without; positive = better with the heatmap for
    gains and Dice, slower for minutes).
    """
    rows = []
    for pid, g in scans.groupby("participant"):
        row = {"participant": pid, "heatmap_first": bool(g.sort_values("position").condition.iloc[0] == WITH),
               "n_with": int((g.condition == WITH).sum()), "n_without": int((g.condition == WITHOUT).sum())}
        for o in OUTCOMES:
            w, wo = g.loc[g.condition == WITH, o].mean(), g.loc[g.condition == WITHOUT, o].mean()
            row.update({f"{o}_with": w, f"{o}_without": wo, f"{o}_diff": w - wo})
        rows.append(row)
    return pd.DataFrame(rows)


def signed_rank_p(nonzero: np.ndarray, seed: int = 0) -> tuple[float, str]:
    """Two-sided p-value of the Wilcoxon signed-rank test, by flipping signs.

    The statistic is the sum of the ranks of the positive differences (ranks of |difference|, ties get their
    average rank). If the heatmap makes no difference, each participant's difference is as likely to be positive
    as negative, so every pattern of signs on the ranks is equally likely. The p-value is the share of sign
    patterns whose statistic lies at least as far from its middle value as the observed one.
    Up to EXACT_MAX_PAIRS differences all 2^n patterns are used (exact; without ties this is the same p-value as
    the textbook exact test); above that RANDOM_SIGN_PATTERNS random patterns with a fixed seed.

    Input: the non-zero differences, seed (only used above EXACT_MAX_PAIRS).
    Output: (p-value, a short description of the method).
    """
    ranks = stats.rankdata(np.abs(nonzero))
    n, middle = len(ranks), ranks.sum() / 2
    observed = abs(ranks[nonzero > 0].sum() - middle)
    exact = n <= EXACT_MAX_PAIRS
    total = 2 ** n if exact else RANDOM_SIGN_PATTERNS
    rng = np.random.default_rng(seed)
    extreme, done, chunk = 0, 0, 2 ** 16
    while done < total:
        size = min(chunk, total - done)
        if exact:     # the bits of the numbers done ... done + size - 1 are all sign patterns, one per row
            signs = (np.arange(done, done + size, dtype=np.int64)[:, None] >> np.arange(n)) & 1
        else:
            signs = rng.integers(0, 2, size=(size, n))
        # Ranks are whole or half numbers, so the sums are exact; the small tolerance only guards the comparison.
        extreme += int(np.sum(np.abs(signs @ ranks - middle) >= observed - 1e-9))
        done += size
    method = f"exact sign-flip ({total} patterns)" if exact else f"random sign-flip ({total} patterns, seeded)"
    return extreme / total, method


def wilcoxon_paired(diff: np.ndarray, n_bootstrap: int = 10000, ci_level: float = 0.95, seed: int = 0) -> dict:
    """Wilcoxon signed-rank test of paired differences against 0 (two-sided), with effect sizes.

    Zero differences are left out of the test (the usual Wilcoxon convention), but not out of the mean.
    - r_rank_biserial: the matched-pairs rank-biserial correlation r = (sum of ranks of positive differences - sum
      of ranks of negative ones) / (sum of all ranks): +1 = every participant higher with the heatmap, -1 = every
      one lower, 0 = balanced. Its sign gives the direction of the effect.
    - mean_diff with a percentile bootstrap confidence interval (ci_low, ci_high): participants are drawn with
      replacement n_bootstrap times, and the interval holds the middle ci_level of the resampled means.
    With 5 non-zero pairs the smallest possible p is 2/32, so p < 0.05 needs at least 6.

    Input: array of differences (with - without), one per participant; bootstrap settings; seed.
    Output: dict with n (non-zero pairs), n_participants, mean_diff, ci_low, ci_high, median_diff, statistic
    (the smaller of the two rank sums), p, p_method, r_rank_biserial.
    """
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    nonzero = diff[diff != 0]
    out = {"n": int(len(nonzero)), "n_participants": int(len(diff)),
           "mean_diff": float(np.mean(diff)) if len(diff) else np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "median_diff": float(np.median(diff)) if len(diff) else np.nan,
           "statistic": np.nan, "p": np.nan, "p_method": "", "r_rank_biserial": np.nan}
    if len(diff) > 1:
        rng = np.random.default_rng(seed)
        means = diff[rng.integers(0, len(diff), size=(n_bootstrap, len(diff)))].mean(axis=1)
        out["ci_low"], out["ci_high"] = (float(q) for q in np.quantile(means, [(1 - ci_level) / 2, (1 + ci_level) / 2]))
    if len(nonzero) == 0:
        return out
    ranks = stats.rankdata(np.abs(nonzero))
    plus, minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    out["r_rank_biserial"] = float((plus - minus) / ranks.sum())
    out["statistic"] = float(min(plus, minus))
    out["p"], out["p_method"] = signed_rank_p(nonzero, seed)
    return out


def conclude(tests: pd.DataFrame, alpha: float) -> pd.Series:
    """The conclusion for each row of the tests table, by the rules of the analysis plan.

    H1: 'supported' if p < alpha and the effect goes the expected way (sign of r_rank_biserial), 'opposite
    direction' if p < alpha the other way, else 'not supported'. H2 is only tested if H1 is supported (fixed
    sequence); otherwise 'not tested (H1 not supported)'. Exploratory rows get 'exploratory': no conclusion.
    """
    def decide(row) -> str:
        if not row.hypothesis:
            return "exploratory"
        if np.isnan(row.p) or row.p >= alpha:
            return "not supported"
        return "supported" if np.sign(row.r_rank_biserial) == row.expected_sign else "opposite direction"

    out = tests.apply(decide, axis=1) if len(tests) else pd.Series(dtype=str)
    h1 = out[tests.hypothesis == "H1"]
    if (tests.hypothesis == "H2").any() and not (len(h1) and h1.iloc[0] == "supported"):
        out[tests.hypothesis == "H2"] = "not tested (H1 not supported)"
    return out


def paired_tests(participants: pd.DataFrame, tlx: pd.DataFrame | None = None, alpha: float = 0.05,
                 n_bootstrap: int = 10000, ci_level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Wilcoxon tests with vs without the heatmap, one row per outcome (only participants with both conditions).

    Rows: gain per minute (H1), Dice after, Dice gain, minutes, and raw TLX (H2) if it was asked after each block.
    Columns: outcome, hypothesis, role, expected_sign, mean_with, mean_without, the columns of wilcoxon_paired,
    and conclusion (see conclude()).
    """
    rows = []
    both = participants[(participants.n_with > 0) & (participants.n_without > 0)]
    boot = {"n_bootstrap": n_bootstrap, "ci_level": ci_level, "seed": seed}

    def row(outcome, with_values, without_values):
        name, role, sign = HYPOTHESES.get(outcome, ("", "exploratory", 0))
        return {"outcome": outcome, "hypothesis": name, "role": role, "expected_sign": sign,
                "mean_with": with_values.mean(), "mean_without": without_values.mean(),
                **wilcoxon_paired((with_values - without_values).to_numpy(), **boot)}

    for o in OUTCOMES:
        rows.append(row(o, both[f"{o}_with"], both[f"{o}_without"]))
    if tlx is not None and len(tlx) and {WITH, WITHOUT} <= set(tlx.which):
        wide = tlx.pivot(index="participant", columns="which", values="raw_tlx").dropna(subset=[WITH, WITHOUT])
        rows.append(row("raw_tlx", wide[WITH], wide[WITHOUT]))
    tests = pd.DataFrame(rows)
    tests["conclusion"] = conclude(tests, alpha)
    return tests


def tlx_table(study: dict, sessions_dir: Path) -> pd.DataFrame:
    """All NASA-TLX answers of real participants: participant, which (session or condition), the six scales, raw_tlx."""
    rows = []
    for folder in participant_folders(study, sessions_dir):
        for path in sorted(folder.glob("tlx_*.json")):
            r = json.loads(path.read_text())
            rows.append({"participant": folder.name, "which": r["which"], **{k: r["scales"][k] for k in TLX_SCALES},
                         "raw_tlx": r["raw_tlx"], "simulated": bool(r.get("simulated", False))})
    return pd.DataFrame(rows)


def condition_table(scans: pd.DataFrame, tlx: pd.DataFrame) -> pd.DataFrame:
    """Descriptive numbers per condition (exploratory): how many scans timed out or were left unchanged, the median
    time, how long the heatmap was switched on, and the median of each NASA-TLX scale (if asked per block)."""
    rows = []
    for cond in (WITH, WITHOUT):
        g = scans[scans.condition == cond]
        row = {"condition": cond, "n_scans": len(g), "n_timeout": int(g.timeout.sum()),
               "n_unchanged": int(g.unchanged.sum()), "median_minutes": g.minutes.median(),
               "median_heatmap_visible_share": (g.heatmap_visible_s / g.duration_s).median() if cond == WITH else np.nan}
        t = tlx[tlx.which == cond] if len(tlx) else tlx
        for k in TLX_SCALES + ["raw_tlx"]:
            row[f"median_{k}"] = t[k].median() if len(t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def order_table(participants: pd.DataFrame) -> pd.DataFrame:
    """Exploratory check of order effects: the primary difference (with - without) for participants who had the
    heatmap block first vs second, with a two-sided Mann-Whitney U test between the two groups. If the heatmap
    only helped in one order (e.g. because of learning), the two groups would differ."""
    diff = f"{PRIMARY}_diff"
    first, second = participants[participants.heatmap_first][diff], participants[~participants.heatmap_first][diff]
    p = stats.mannwhitneyu(first, second, alternative="two-sided").pvalue if len(first) and len(second) else np.nan
    return pd.DataFrame([{"heatmap_first": order, "n": len(v), "mean_diff": v.mean(), "median_diff": v.median(),
                          "mannwhitney_p": p} for order, v in ((True, first), (False, second))])


def plot_paired(participants: pd.DataFrame, outcome: str, ylabel: str, path: Path, title: str) -> None:
    """One line per participant from WITHOUT to WITH the heatmap, so every pair is visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for _, r in participants.iterrows():
        ax.plot([0, 1], [r[f"{outcome}_without"], r[f"{outcome}_with"]], "-o", color="#4a6fa5", alpha=.7, ms=4)
    ax.set_xticks([0, 1], ["without heatmap", "with heatmap"])
    ax.set_xlim(-.3, 1.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze(study: dict, source: dict, sessions_dir: Path, out_dir: Path,
            truths: TruthCache | None = None) -> dict[str, pd.DataFrame] | None:
    """Run the whole analysis on one sessions folder and write the tables and a figure to out_dir.

    Output files (all starting with SIMULATED_ if the sessions are simulated): study_scans.csv (every saved study
    scan, with 'included'), study_exclusions.csv, study_participants.csv, study_tests.csv (H1, H2, exploratory),
    study_conditions.csv, study_order.csv, study_tlx.csv (if any) and figures/study_gain_per_min.png.
    Settings: study.analysis (alpha, n_bootstrap, ci_level) and the study seed.
    Returns the tables, or None if there are no participant sessions yet.
    """
    flags = simulated_flags(study, sessions_dir)
    if len(flags) > 1:
        raise ValueError("Real and simulated sessions are mixed in one folder; keep simulated sessions separate.")
    simulated = flags == {True}
    if simulated and Path(out_dir).resolve().is_relative_to((REPO_ROOT / "results").resolve()):
        raise ValueError("Simulated sessions must never be written to results/.")
    scans = scan_table(study, source, sessions_dir, truths)
    if scans.empty:
        return None
    displays = sorted(scans.heatmap_display.unique())
    if len(displays) > 1:
        raise ValueError(f"The participants saw the heatmap in different ways {displays}; the comparison needs one "
                         "locked study.heatmap_display. See study_scans for who saw what.")
    scans, exclusions = mark_complete(scans, len(load_manifest(study)["study_scans"]))
    included = scans[scans.included]
    tlx = tlx_table(study, sessions_dir)
    if len(tlx):
        tlx = tlx[tlx.participant.isin(set(included.participant))]
    settings = study["analysis"]
    participants = participant_table(included)
    tests = (paired_tests(participants, tlx, settings["alpha"], settings["n_bootstrap"], settings["ci_level"],
                          study["seed"]) if len(participants) else pd.DataFrame())
    tables = {"scans": scans, "exclusions": exclusions, "participants": participants, "tests": tests,
              "conditions": condition_table(included, tlx),
              "order": order_table(participants) if len(participants) else pd.DataFrame(), "tlx": tlx}
    if len(tests):
        tests.insert(0, "heatmap_display", displays[0])
    for table in tables.values():
        if "simulated" not in table:
            table.insert(0, "simulated", simulated)
    prefix = SIMULATED_PREFIX if simulated else ""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name != "tlx" or len(table):
            table.to_csv(out_dir / f"{prefix}study_{name}.csv", index=False)
    if len(participants):
        title = f"{'SIMULATED DATA - ' if simulated else ''}{len(participants)} participants"
        plot_paired(participants, PRIMARY, "Dice gain per minute",
                    out_dir / "figures" / f"{prefix}study_gain_per_min.png", title)
    return tables
