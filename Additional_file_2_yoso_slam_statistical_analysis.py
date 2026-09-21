#!/usr/bin/env python3
"""
YOSO-SLAM -- Reproducible statistical analysis (Additional file 2)
==================================================================
Accompanies: "YOSO-SLAM: One-Shot Panoptic Segmentation for Efficient
Dynamic RGB-D Simultaneous Localization and Mapping"

Reproduces every statistic reported in the manuscript from the released
per-sequence data files (values exactly as recorded from the evo output and
the timing logs; ATE RMSE at four decimal places, tracking times at three):

  (1) Paired Wilcoxon signed-rank tests, YOSO-SLAM vs. each baseline
      (Table 6), with Holm adjustment across the five comparisons,
      matched-pairs rank-biserial effect sizes r_rb = (W+ - W-)/(W+ + W-),
      and BCa bootstrap 95% CIs for the mean paired difference.
  (2) Bias-corrected-and-accelerated (BCa) bootstrap 95% confidence
      intervals for each method's global mean ATE (Table 2), and the ATE
      reduction of each method relative to ORB-SLAM3 (Table 3).
  (3) Paired two one-sided tests (TOST) equivalence analysis against a
      margin fixed on practical grounds, reported both parametrically (t) and
      nonparametrically (two one-sided Wilcoxon tests) (Section 5.6).
  (4) Paired Wilcoxon signed-rank tests on per-sequence mean tracking
      time, YOSO-SLAM vs. the other panoptic front-ends (Section 5.9).
  (5) Ablation study (Table 10): aggregate means, percentage changes with
      the formula stated, and Wilcoxon tests vs. the full pipeline (n=10).
  (6) Repeated-run experiment (Tables 7-8): per-sequence mean/SD/CV and
      two-sided Mann-Whitney U tests, n = 4 vs. 4.

Input files (same directory):
  Additional_file_1_ate32_per_sequence.csv            32 x 6, ATE RMSE (m)
  Additional_file_3_tracking_time32_per_sequence.csv  32 x 4, mean tracking time (s)
  Additional_file_4_ablation10_per_sequence.csv       10 x 12, ATE and tracking time
  Additional_file_5_repeated_runs_per_run.csv    72 rows, long format
Usage:    python Additional_file_2_yoso_slam_statistical_analysis.py
Requires: numpy, pandas, scipy >= 1.9

Conventions (stated explicitly for reproducibility)
---------------------------------------------------
* Estimand      : per-sequence paired difference in ATE RMSE
                  (YOSO-SLAM minus baseline; negative favors YOSO-SLAM).
* Alternative   : two-sided.
* Statistic W   : min(W+, W-), the smaller of the two signed-rank sums
                  (SciPy convention for a two-sided test).
* Zero handling : scipy default 'wilcox' (zero differences discarded).
                  The Panoptic-SLAM comparison contains 1 zero difference.
* Method        : scipy 'auto' (SciPy 1.17): exact null distribution when the
                  differences contain no ties or zeros (ORB-SLAM3, ReFusion),
                  tie-corrected normal approximation otherwise.
* Equivalence   : margin fixed on practical grounds (see below), independently
                  of the observed differences; the smallest margin at which
                  both TOST variants pass is also reported for transparency.
"""

import numpy as np
import pandas as pd
from scipy import stats

# --- Equivalence margin, fixed on practical grounds ------------------------
# 0.02 m is below one cell of a standard 5 cm occupancy grid used for indoor
# mobile-robot navigation, and is ~7% of the ATE improvement that dynamic
# filtering delivers over the ORB-SLAM3 baseline (0.386 -> 0.111 m).
EQUIV_MARGIN_M = 0.020

REFERENCE = "YOSO-SLAM"
BASELINES = ["ORB-SLAM3", "ReFusion", "Panoptic-SLAM", "YDM-SLAM", "Mask2Former-SLAM"]
SEED = 42

F_ATE = "Additional_file_1_ate32_per_sequence.csv"
F_TIME = "Additional_file_3_tracking_time32_per_sequence.csv"
F_ABL = "Additional_file_4_ablation10_per_sequence.csv"
F_RUNS = "Additional_file_5_repeated_runs_per_run.csv"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)  # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (zeros discarded)."""
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    w_plus, w_minus = r[d > 0].sum(), r[d < 0].sum()
    return (w_plus - w_minus) / (w_plus + w_minus), w_plus, w_minus


def wilcoxon_table(df):
    # A dedicated generator keeps the bootstrap draws of sections (2)-(3)
    # identical to those used for the manuscript.
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("(1) PAIRED WILCOXON SIGNED-RANK TESTS  --  Table 6")
    print("=" * 78)
    print(f"{'Baseline':<18}{'W':>6}{'W+':>6}{'W-':>6}{'p':>11}{'p(Holm)':>9}"
          f"{'r_rb':>7}{'mean diff':>11}   95% BCa CI (m)   sig")
    raw = []
    for m in BASELINES:
        r = stats.wilcoxon(df[REFERENCE], df[m], alternative="two-sided")
        d = (df[REFERENCE] - df[m]).to_numpy()
        rb, wp, wm = rank_biserial(d)
        ci = stats.bootstrap((d,), np.mean, confidence_level=0.95,
                             n_resamples=20000, method="BCa",
                             random_state=rng).confidence_interval
        raw.append((m, r.statistic, wp, wm, r.pvalue, rb, d.mean(), ci))
    adj = holm([x[4] for x in raw])
    for (m, w, wp, wm, p, rb, md, ci), pa in zip(raw, adj):
        print(f"{m:<18}{w:>6.0f}{wp:>6.0f}{wm:>6.0f}{p:>11.6f}{pa:>9.4f}"
              f"{rb:>+7.2f}{md:>+11.4f}   [{ci.low:+.4f}, {ci.high:+.4f}]"
              f"   {'YES' if pa < 0.05 else 'no'}")
    print("\nW = min(W+, W-). Differences are YOSO-SLAM minus baseline.")
    print("The five comparisons are exploratory; Holm-adjusted values are")
    print("reported so the reader can apply either standard.")


def reductions(df):
    print()
    print("=" * 78)
    print("(2b) ATE REDUCTION RELATIVE TO ORB-SLAM3 (from the recorded per-sequence values)  --  Table 3")
    print("=" * 78)
    ref = df["ORB-SLAM3"].mean()
    for m in df.columns:
        print(f"{m:<20}{df[m].mean():>10.4f}{100 * (1 - df[m].mean() / ref):>9.1f}%")


def bootstrap_cis(df, rng):
    print()
    print("=" * 78)
    print("(2) GLOBAL MEAN ATE with BCa BOOTSTRAP 95% CI (n=32 sequences)  --  Table 2")
    print("=" * 78)
    print(f"{'Method':<20}{'mean (m)':>10}{'SD':>9}   95% CI (m)")
    for m in df.columns:
        x = df[m].to_numpy()
        ci = stats.bootstrap((x,), np.mean, confidence_level=0.95,
                             n_resamples=20000, method="BCa",
                             random_state=rng).confidence_interval
        print(f"{m:<20}{x.mean():>10.4f}{x.std(ddof=1):>9.4f}"
              f"   [{ci.low:.4f}, {ci.high:.4f}]")


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
    print(f"(3) TOST EQUIVALENCE ANALYSIS  --  margin = +/-{EQUIV_MARGIN_M:.3f} m (fixed on practical grounds)")
    print("=" * 78)
    print(f"{'Comparison':<30}{'mean diff':>11}{'90% CI (m)':>22}{'TOST p':>9}{'TOST p':>9}")
    print(f"{'':<30}{'(m)':>11}{'':>22}{'(t)':>9}{'(Wilc)':>9}")
    for m in ["Mask2Former-SLAM", "Panoptic-SLAM", "YDM-SLAM"]:
        d = (df[REFERENCE] - df[m]).to_numpy()
        ci = stats.bootstrap((d,), np.mean, confidence_level=0.90,
                             n_resamples=20000, method="BCa",
                             random_state=rng).confidence_interval
        p_t, p_w = tost_paired(d, EQUIV_MARGIN_M)
        verdict = "equivalent" if max(p_t, p_w) < 0.05 else "not concluded"
        print(f"{'YOSO vs ' + m:<30}{d.mean():>+11.4f}"
              f"   [{ci.low:+.4f}, {ci.high:+.4f}]{p_t:>9.4f}{p_w:>9.4f}   {verdict}")
        # smallest margin at which BOTH TOST variants reject at alpha = 0.05 (bisection)
        lo, hi = 0.0, 0.2
        for _ in range(40):
            mid = (lo + hi) / 2
            if max(tost_paired(d, mid)) < 0.05: hi = mid
            else: lo = mid
        print(f"{'':<30}   smallest margin passing both TOST variants: {hi:.4f} m")
    print()
    print("Equivalence is concluded only where BOTH the parametric and the")
    print("nonparametric TOST reject at alpha = 0.05. Where it is not concluded,")
    print("the manuscript reports only that no significant difference was detected.")


def tracking_time_tests():
    print()
    print("=" * 78)
    print("(4) PAIRED WILCOXON ON MEAN TRACKING TIME (s/frame)  --  Section 5.9")
    print("=" * 78)
    tt = pd.read_csv(F_TIME, index_col=0)
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
    ab = pd.read_csv(F_ABL, index_col=0)
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
    rr = pd.read_csv(F_RUNS)
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
    df = pd.read_csv(F_ATE, index_col=0)
    assert len(df) == 32, f"expected 32 sequences, found {len(df)}"
    assert not df.isna().any().any(), "missing ATE values"
    wilcoxon_table(df)
    bootstrap_cis(df, rng)
    reductions(df)
    equivalence(df, rng)
    tracking_time_tests()
    ablation()
    repeated_runs()


if __name__ == "__main__":
    main()
