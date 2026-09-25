"""Analysis of the user study (G2): does the uncertainty heatmap make human correction more efficient?

Per corrected study scan (from sessions/<participant>/<case>_mask.nii.gz and _log.json):
- Dice before: the model's mask against the ground truth (what the participant started from);
- Dice after:  the participant's saved mask against the ground truth;
- minutes:     the time on the scan, from the log (until "Done", or the time limit);
- gain per minute = (Dice after - Dice before) / minutes. This is the PRIMARY metric ("Dice per unit of human
  effort"). With the time limit it stays defined: the gain is measured at "Done" or when the time ran out.

Per participant: the mean over their scans WITH the heatmap and over their scans WITHOUT it (3 + 3). The
comparison is a Wilcoxon signed-rank test on the per-participant differences (with - without), two-sided.
Pairing per participant takes out that some people are simply faster or more careful than others. Because every
scan is corrected with the heatmap by half of the participants and without it by the other half, differences
between scans cancel out as well (see study.py for the balancing).

NASA-TLX (Raw TLX, 0-100, higher = more workload): reported per participant. If it was collected after each block
(study.tlx = after_each_block), the raw TLX with vs without the heatmap is compared the same way.

Which data count: only real participant ids (P01, ...). Test ids (T01, ...) and practice scans are left out.
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
from segreview.metrics import dice
from segreview.study import TLX_SCALES, WITH, WITHOUT, assignment, id_pattern, id_prefixes, load_manifest, study_dir

OUTCOMES = ["gain_per_min", "dice_after", "dice_gain", "minutes"]     # per participant and condition
PRIMARY = "gain_per_min"
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
            truth, ignore = truths.get(case_id)
            before = _mask(root / "scans" / case_id / "mask.nii.gz")
            after = _mask(folder / f"{case_id}_mask.nii.gz")
            if after.shape != truth.shape or before.shape != truth.shape:
                raise ValueError(f"{pid}/{case_id}: mask shape {after.shape} differs from the ground truth {truth.shape}")
            minutes = log["duration_s"] / 60.0
            d_before, d_after = dice(before, truth, ignore), dice(after, truth, ignore)
            rows.append({"participant": pid, "case_id": case_id, "condition": log["condition"],
                         "position": plan[case_id]["position"], "reason": log["reason"],
                         "duration_s": log["duration_s"], "minutes": minutes,
                         "dice_before": d_before, "dice_after": d_after, "dice_gain": d_after - d_before,
                         "gain_per_min": (d_after - d_before) / minutes,
                         "strokes": log.get("strokes"), "undo_count": log.get("undo_count"),
                         "heatmap_visible_s": log.get("heatmap_visible_s"),
                         "simulated": bool(log.get("simulated", False))})
    return pd.DataFrame(rows)


def participant_table(scans: pd.DataFrame) -> pd.DataFrame:
    """Per participant: mean of each outcome over the scans WITH and WITHOUT the heatmap, and the difference.

    Columns: participant, n_with, n_without, and for each outcome <outcome>_with, <outcome>_without,
    <outcome>_diff (= with - without; positive = better with the heatmap for gains and Dice, slower for minutes).
    """
    rows = []
    for pid, g in scans.groupby("participant"):
        row = {"participant": pid, "n_with": int((g.condition == WITH).sum()),
               "n_without": int((g.condition == WITHOUT).sum())}
        for o in OUTCOMES:
            w, wo = g.loc[g.condition == WITH, o].mean(), g.loc[g.condition == WITHOUT, o].mean()
            row.update({f"{o}_with": w, f"{o}_without": wo, f"{o}_diff": w - wo})
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_paired(diff: np.ndarray) -> dict:
    """Wilcoxon signed-rank test of paired differences against 0 (two-sided), with an effect size.

    Zero differences are left out (the usual Wilcoxon convention). The effect size is the matched-pairs
    rank-biserial correlation r = (sum of ranks of positive differences - sum of ranks of negative ones) /
    (sum of all ranks): +1 = every participant better with the heatmap, -1 = every one worse, 0 = balanced.
    With exact p-values, n = 5 non-zero pairs can never reach p < 0.05 (smallest possible p = 2/32).

    Input: array of differences (with - without), one per participant.
    Output: dict with n (non-zero pairs), statistic, p, r_rank_biserial, mean_diff, median_diff.
    """
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    nonzero = diff[diff != 0]
    out = {"n": int(len(nonzero)), "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
           "median_diff": float(np.median(diff)) if len(diff) else np.nan,
           "statistic": np.nan, "p": np.nan, "r_rank_biserial": np.nan}
    if len(nonzero) == 0:
        return out
    ranks = stats.rankdata(np.abs(nonzero))
    out["r_rank_biserial"] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    res = stats.wilcoxon(nonzero, alternative="two-sided")
    out.update(statistic=float(res.statistic), p=float(res.pvalue))
    return out


def paired_tests(participants: pd.DataFrame, tlx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wilcoxon tests with vs without the heatmap, one row per outcome (only participants with both conditions).

    The primary outcome is gain per minute; the others (Dice after, Dice gain, minutes, and raw TLX if it was
    collected per block) are secondary and should be read as exploratory.
    """
    rows = []
    both = participants[(participants.n_with > 0) & (participants.n_without > 0)]
    for o in OUTCOMES:
        rows.append({"outcome": o, "primary": o == PRIMARY, "mean_with": both[f"{o}_with"].mean(),
                     "mean_without": both[f"{o}_without"].mean(), **wilcoxon_paired(both[f"{o}_diff"].to_numpy())})
    if tlx is not None and len(tlx) and {WITH, WITHOUT} <= set(tlx.which):
        wide = tlx.pivot(index="participant", columns="which", values="raw_tlx").dropna(subset=[WITH, WITHOUT])
        rows.append({"outcome": "raw_tlx", "primary": False, "mean_with": wide[WITH].mean(),
                     "mean_without": wide[WITHOUT].mean(), **wilcoxon_paired((wide[WITH] - wide[WITHOUT]).to_numpy())})
    return pd.DataFrame(rows)


def tlx_table(study: dict, sessions_dir: Path) -> pd.DataFrame:
    """All NASA-TLX answers of real participants: participant, which (session or condition), the six scales, raw_tlx."""
    rows = []
    for folder in participant_folders(study, sessions_dir):
        for path in sorted(folder.glob("tlx_*.json")):
            r = json.loads(path.read_text())
            rows.append({"participant": folder.name, "which": r["which"], **{k: r["scales"][k] for k in TLX_SCALES},
                         "raw_tlx": r["raw_tlx"], "simulated": bool(r.get("simulated", False))})
    return pd.DataFrame(rows)


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

    Output files: study_scans.csv, study_participants.csv, study_tests.csv, study_tlx.csv,
    figures/study_gain_per_min.png (all starting with SIMULATED_ if the sessions are simulated).
    Returns the tables, or None if there are no participant sessions yet.
    """
    flags = simulated_flags(study, sessions_dir)
    if len(flags) > 1:
        raise ValueError("Real and simulated sessions are mixed in one folder; keep simulated sessions separate.")
    simulated = flags == {True}
    if simulated and Path(out_dir).resolve().is_relative_to((REPO_ROOT / "results").resolve()):
        raise ValueError("Simulated sessions must never be written to results/.")
    scans = scan_table(study, source, sessions_dir, truths)
    tlx = tlx_table(study, sessions_dir)
    if scans.empty:
        return None
    participants = participant_table(scans)
    tests = paired_tests(participants, tlx)
    for table in (participants, tests):
        table.insert(0, "simulated", simulated)
    prefix = SIMULATED_PREFIX if simulated else ""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scans.to_csv(out_dir / f"{prefix}study_scans.csv", index=False)
    participants.to_csv(out_dir / f"{prefix}study_participants.csv", index=False)
    tests.to_csv(out_dir / f"{prefix}study_tests.csv", index=False)
    if len(tlx):
        tlx.to_csv(out_dir / f"{prefix}study_tlx.csv", index=False)
    title = f"{'SIMULATED DATA - ' if simulated else ''}{len(participants)} participants"
    plot_paired(participants, PRIMARY, "Dice gain per minute", out_dir / "figures" / f"{prefix}study_gain_per_min.png", title)
    return {"scans": scans, "participants": participants, "tests": tests, "tlx": tlx}
