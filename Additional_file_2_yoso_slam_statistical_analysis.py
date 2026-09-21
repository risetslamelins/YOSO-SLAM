#!/usr/bin/env python3
"""
YOSO-SLAM — Reproducible statistical analysis
=============================================
Accompanies: "YOSO-SLAM: One-Shot Panoptic Segmentation for Efficient
Dynamic RGB-D Simultaneous Localization and Mapping"

Reproduces every statistic reported in the revised manuscript:

  (1) Paired Wilcoxon signed-rank tests, YOSO-SLAM vs. each baseline
      (revised Table 6), with Holm correction across the 5 comparisons.
  (2) Bias-corrected-and-accelerated (BCa) bootstrap 95% confidence
      intervals for each method's global mean ATE.
  (3) Paired two-one-sided-tests (TOST) equivalence analysis against a
      pre-specified margin, reported both parametrically (t) and
      nonparametrically (two one-sided Wilcoxon tests).
  (4) Paired Wilcoxon signed-rank tests on per-sequence mean tracking
      time, YOSO-SLAM vs. the other panoptic front-ends (Section 5.9).
  (5) Ablation study (Table 10): aggregate means, percentage changes with
      the formula stated, and Wilcoxon tests vs. the full pipeline (n=10).
  (6) Repeated-run experiment (Tables 7-8): per-sequence mean/SD/CV and
      two-sided Mann-Whitney U tests, n = 4 vs. 4.

Input : ate32_full_precision.csv           — 32 x 6, ATE RMSE (m)
        tracking_time32_full_precision.csv — 32 x 4, mean tracking time (s)
        ablation10_full_precision.csv      — 10 x 12, ATE and tracking time
        repeated_runs_full_precision.csv   — 72 rows, long format
Usage : python yoso_slam_statistical_analysis.py
Requires: numpy, pandas, scipy >= 1.9

Conventions (stated explicitly for reproducibility)
---------------------------------------------------
* Estimand      : per-sequence paired difference in ATE RMSE.
* Alternative   : two-sided.
* Zero handling : scipy default 'wilcox' (zero differences discarded).
                  The Panoptic-SLAM comparison contains 1 zero difference.
* Method        : scipy default (exact for n<=25, else normal approx.).
* Equivalence   : margin fixed a priori on practical grounds (see below),
                  NOT chosen from the observed data.
"""

import numpy as np
import pandas as pd
from scipy import stats

# --- Equivalence margin, pre-specified on practical grounds -----------------
# 0.02 m is below one cell of a standard 5 cm occupancy grid used for indoor
# mobile-robot navigation, and is ~7% of the ATE improvement that dynamic
# filtering delivers over the ORB-SLAM3 baseline (0.386 -> 0.111 m).
# A trajectory difference of this size cannot change navigation behaviour.
EQUIV_MARGIN_M = 0.020

REFERENCE = "YOSO-SLAM"
BASELINES = ["ORB-SLAM3", "ReFusion", "Panoptic-SLAM",
             "YDM-SLAM", "Mask2Former-SLAM"]
SEED = 42


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj


def wilcoxon_table(df):
    print("=" * 78)
    print("(1) PAIRED WILCOXON SIGNED-RANK TESTS  —  revised Table 6")
    print("=" * 78)
    print(f"{'Comparison':<34}{'W':>8}{'p':>12}{'p (Holm)':>11}  sig")
    raw = []
    for m in BASELINES:
        r = stats.wilcoxon(df[REFERENCE], df[m], alternative="two-sided")
        raw.append((m, r.statistic, r.pvalue))
    adj = holm([p for _, _, p in raw])
    for (m, w, p), pa in zip(raw, adj):
        print(f"{'YOSO-SLAM vs. ' + m:<34}{w:>8.1f}{p:>12.6f}{pa:>11.4f}"
              f"  {'YES' if pa < 0.05 else 'no'}")
    print("\nNote: the five comparisons are exploratory; Holm-adjusted values")
    print("are reported so the reader can apply either standard.")


def bootstrap_cis(df, rng):
    print()
    print("=" * 78)
    print("(2) GLOBAL MEAN ATE with BCa BOOTSTRAP 95% CI (n=32 sequences)")
    print("=" * 78)
    print(f"{'Method':<20}{'mean (m)':>10}{'SD':>9}   95% CI (m)")
    for m in df.columns:
        x = df[m].to_numpy()
        ci = stats.bootstrap((x,), np.mean, confidence_level=0.95,
                             n_resamples=20000, method="BCa",
                             random_state=rng).confidence_interval
        print(f"{m:<20}{x.mean():>10.4f}{x.std(ddof=1):>9.4f}"
              f"   [{ci.low:.4f}, {ci.high:.4f}]")
    print("\nThe intervals overlap substantially: the margins separating the")
    print("leading methods are small relative to between-sequence dispersion.")


def tost_paired(d, margin):
    """Paired TOST. Returns (p_t, p_wilcoxon); equivalence if max < alpha."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((d.mean() + margin) / se, n - 1)   # H0: diff <= -margin
    p_upper = stats.t.cdf((d.mean() - margin) / se, n - 1)  # H0: diff >= +margin
    p_t = max(p_lower, p_upper)

    p_w = max(stats.wilcoxon(d + margin, alternative="greater").pvalue,
              stats.wilcoxon(d - margin, alternative="less").pvalue)
    return p_t, p_w


def equivalence(df, rng):
    print()
    print("=" * 78)
    print(f"(3) TOST EQUIVALENCE ANALYSIS  —  margin = \u00B1{EQUIV_MARGIN_M:.3f} m "
          f"(pre-specified)")
    print("=" * 78)
    print(f"{'Comparison':<30}{'mean diff':>11}{'90% CI (m)':>22}"
          f"{'TOST p':>9}{'TOST p':>9}")
    print(f"{'':<30}{'(m)':>11}{'':>22}{'(t)':>9}{'(Wilc)':>9}")
    for m in ["Mask2Former-SLAM", "Panoptic-SLAM", "YDM-SLAM"]:
        d = (df[REFERENCE] - df[m]).to_numpy()
        ci = stats.bootstrap((d,), np.mean, confidence_level=0.90,
                             n_resamples=20000, method="BCa",
                             random_state=rng).confidence_interval
        p_t, p_w = tost_paired(d, EQUIV_MARGIN_M)
        verdict = "equivalent" if max(p_t, p_w) < 0.05 else "not concluded"
        print(f"{'YOSO vs ' + m:<30}{d.mean():>+11.4f}"
              f"   [{ci.low:+.4f}, {ci.high:+.4f}]{p_t:>9.4f}{p_w:>9.4f}"
              f"   {verdict}")
    print()
    print("Equivalence is concluded only where BOTH the parametric and the")
    print("nonparametric TOST reject at alpha = 0.05. Where it is not")
    print("concluded, we report only that no significant difference was")
    print("detected — we do not claim equivalence.")


def tracking_time_tests():
    print()
    print("=" * 78)
    print("(4) PAIRED WILCOXON ON MEAN TRACKING TIME (s/frame), YOSO-SLAM vs. panoptic peers")
    print("=" * 78)
    tt = pd.read_csv("Additional_file_3_tracking_time32_full_precision.csv", index_col=0)
    assert len(tt) == 32
    print(f"{'Comparison':<34}{'W':>6}{'p':>12}{'faster on':>12}{'reduction':>11}")
    for m in ["Panoptic-SLAM", "Mask2Former-SLAM"]:
        r = stats.wilcoxon(tt[REFERENCE], tt[m], alternative="two-sided")
        faster = int((tt[REFERENCE] < tt[m]).sum())
        red = 100 * (1 - tt[REFERENCE].mean() / tt[m].mean())
        print(f"{'YOSO-SLAM vs. ' + m:<34}{r.statistic:>6.0f}{r.pvalue:>12.2e}"
              f"{faster:>9}/32{red:>10.1f}%")


def ablation():
    print()
    print("=" * 78)
    print("(5) ABLATION (Table 10): aggregate means, % change, Wilcoxon vs. full (n=10)")
    print("=" * 78)
    ab = pd.read_csv("Additional_file_4_ablation10_full_precision.csv", index_col=0)
    cfg = ["V0_ORB-SLAM3", "V1_People", "V2_People+Moving", "V3_People+Unknown",
           "Panoptic-SLAM", "YOSO_full"]
    means = {c: ab[f"ATE_{c}"].mean() for c in cfg}
    print(f"{'Config':<20}{'mean ATE':>10}{'mean t':>9}{'p vs full':>11}")
    for c in cfg:
        p = ("--" if c == "YOSO_full" else
             f"{stats.wilcoxon(ab['ATE_YOSO_full'], ab[f'ATE_{c}']).pvalue:.3f}")
        print(f"{c:<20}{means[c]:>10.4f}{ab[f'time_{c}'].mean():>9.4f}{p:>11}")
    pct = lambda a, b: 100 * (means[a] - means[b]) / means[a]
    print("\nPercentage change computed as (mean_before - mean_after) / mean_before:")
    print(f"  V0->V1 {pct('V0_ORB-SLAM3','V1_People'):.1f}%   V1->V2 {pct('V1_People','V2_People+Moving'):.1f}%"
          f"   V1->V3 {pct('V1_People','V3_People+Unknown'):.1f}%   V1->full {pct('V1_People','YOSO_full'):.1f}%")
    r = stats.wilcoxon(ab["ATE_V0_ORB-SLAM3"], ab["ATE_V1_People"])
    print(f"  V0 vs V1: W={r.statistic:.1f} p={r.pvalue:.3f};", end=" ")
    r = stats.wilcoxon(ab["ATE_V2_People+Moving"], ab["ATE_V3_People+Unknown"])
    print(f"V2 vs V3: W={r.statistic:.1f} p={r.pvalue:.3f}")


def repeated_runs():
    print()
    print("=" * 78)
    print("(6) REPEATED RUNS (Tables 7-8): mean/SD/CV and Mann-Whitney U, n=4 vs 4")
    print("=" * 78)
    rr = pd.read_csv("Additional_file_5_repeated_runs_full_precision.csv")
    print(f"{'Sequence':<22}{'YOSO mean':>10}{'CV%':>6}{'p vs YDM':>10}{'p vs Pan':>10}")
    for s in rr["sequence"].unique():
        g = lambda m: rr[(rr.sequence == s) & (rr.method == m)]["ATE_RMSE_m"].to_numpy()
        y = g("YOSO-SLAM")
        p1 = stats.mannwhitneyu(y, g("YDM-SLAM"), alternative="two-sided").pvalue
        p2 = stats.mannwhitneyu(y, g("Panoptic-SLAM"), alternative="two-sided").pvalue
        print(f"{s:<22}{y.mean():>10.4f}{100*y.std(ddof=1)/y.mean():>6.1f}{p1:>10.4f}{p2:>10.4f}")
    print("\nSmallest attainable two-sided p for n=4 vs 4 is 2/70 = 0.0286.")


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("Additional_file_1_ate32_full_precision.csv", index_col=0)
    assert len(df) == 32, f"expected 32 sequences, found {len(df)}"
    assert not df.isna().any().any(), "missing ATE values"

    wilcoxon_table(df)
    bootstrap_cis(df, rng)
    equivalence(df, rng)
    tracking_time_tests()
    ablation()
    repeated_runs()


if __name__ == "__main__":
    main()
