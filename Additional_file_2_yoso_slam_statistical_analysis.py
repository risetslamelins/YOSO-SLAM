#!/usr/bin/env python3
"""
YOSO-SLAM -- Reproducible statistical analysis (Additional file 2)
==================================================================
Accompanies: "YOSO-SLAM: One-Shot Panoptic Segmentation for Efficient
Dynamic RGB-D Simultaneous Localization and Mapping"

Reproduces the benchmark, tracking-time, ablation and repeated-run statistics
reported in the manuscript (Tables 2, 3, 6, 7, 8 and 10 and the tests of
Sections 5.6, 5.9 and 5.11) from the released per-sequence data files. Not
covered: the field statistics of Section 5.12 and the ORB-SLAM3/ReFusion
tracking times and the initialization and total times of Table 9, whose
underlying logs are not part of the released files. The 32 x 6 ATE RMSE values (Additional file 1) and
the single-run values of the repeated-run file (Additional file 5, "Run 1")
were recomputed at full precision from the stored single-run trajectories
with evo (evo_ape tum <groundtruth> <trajectory> -a; SE(3) Umeyama
alignment, no scale, max. time difference 0.01 s) and are released at six
decimal places, the precision of the evo output; where the evo log of the
original run was still available (ORB-SLAM3, YDM-SLAM, ReFusion: 96 values)
the recomputation agrees with it to < 5e-7 m. The ablation ATE values
(Additional file 4) were recomputed in the same way from the stored
trajectories of the ablation runs (V1-V3) and, for V0, Panoptic-SLAM and the
full pipeline, are the single-run values of Additional file 1. Tracking
times (Additional files 3 and 4) are given as recorded by the timing logs:

  (1) Paired Wilcoxon signed-rank tests, YOSO-SLAM vs. each baseline
      (Table 6), with Holm adjustment across the five comparisons,
      matched-pairs rank-biserial effect sizes r_rb = (W+ - W-)/(W+ + W-),
      and BCa bootstrap 95% CIs for the mean paired difference.
  (2) Global mean, SD, median, minimum and maximum ATE of each method with
      bias-corrected-and-accelerated (BCa) bootstrap 95% confidence
      intervals for the mean (Table 2), and the ATE reduction of each method
      relative to ORB-SLAM3 (Table 3).
  (3) Paired two one-sided tests (TOST) equivalence analysis against a
      margin fixed on practical grounds, reported both parametrically (t) and
      nonparametrically (two one-sided Wilcoxon tests) (Section 5.6), with
      sensitivity checks for the YOSO-Mask2Former pair: the t-based 90% CI of
      the mean difference, a scale-free TOST on the log ATE ratio
      (equivalence bounds of +/-10%), and the number of single-sequence
      differences beyond the margin (the equivalence concerns the mean
      paired difference, not individual sequences).
  (4) Paired Wilcoxon signed-rank tests on per-sequence mean tracking
      time, YOSO-SLAM vs. the other panoptic front-ends (Section 5.9); the
      three-decimal times contain tied differences, so SciPy uses the
      tie-corrected normal approximation, and the exact sign-flip p-value is
      also printed for the case W = 0.
  (5) Ablation study (Table 10): aggregate means, percentage changes with
      the formula stated, and Wilcoxon tests vs. the full pipeline (n=10)
      with Holm adjustment across the five comparisons (exploratory).
  (6) Repeated-run experiment (Tables 7-8): per-sequence mean/SD/CV for
      YOSO-SLAM, Mask2Former-SLAM, YDM-SLAM and Panoptic-SLAM, and two-sided
      Mann-Whitney U tests of YOSO-SLAM against each baseline, n = 4 vs. 4.
  (7) Replicated equivalence analysis for the YOSO-Mask2Former pair
      (Table 8 and Section 5.6): per-sequence run-level mean difference with
      Welch 90% confidence interval and TOST against the +/-0.020 m margin,
      the same comparison on a relative scale (geometric-mean ATE ratio with
      a Welch 90% CI on the log run values, bounds of +/-10%), and the pooled
      sequence-level paired TOST over the six sequences.

Input files (same directory):
  Additional_file_1_ate32_per_sequence.csv            32 x 6, ATE RMSE (m), full precision (6 dp)
  Additional_file_3_tracking_time32_per_sequence.csv  32 x 4, mean tracking time (s)
  Additional_file_4_ablation10_per_sequence.csv       10 x 12, ATE and tracking time
  Additional_file_5_repeated_runs_per_run.csv    96 rows, long format (4 methods x 6 sequences x 4 runs)
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
                  At full precision none of the five comparisons contains a
                  zero difference or tied absolute differences.
* Method        : scipy 'auto' (SciPy 1.17): the exact null distribution is
                  therefore used for all five comparisons. (With the earlier
                  four-decimal tabulation, ties and one zero occurred and the
                  tie-corrected normal approximation applied to three of them.)
* Equivalence   : margin chosen on practical grounds (see below) before the
                  TOST was computed; because the per-sequence values had
                  already been reported, the margin was not pre-registered.
                  The smallest margin at which both TOST variants pass is
                  also reported, as a sensitivity analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats

# --- Equivalence margin, chosen on practical grounds (not pre-registered) ---
# 0.02 m is smaller than one cell of an occupancy grid at the 5 cm resolution
# often used for indoor mobile-robot navigation, and is ~7% of the ATE
# improvement that dynamic filtering delivers over the ORB-SLAM3 baseline
# (0.386 -> 0.111 m). Because the margin was not pre-registered, a scale-free
# check (TOST on the log ATE ratio, bounds +/-10%) is also reported.
RATIO_MARGIN = 0.10
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
    print(f"{'Method':<20}{'mean (m)':>10}{'SD':>9}{'median':>9}{'min':>8}{'max':>8}   95% CI (m)")
    for m in df.columns:
        x = df[m].to_numpy()
        ci = stats.bootstrap((x,), np.mean, confidence_level=0.95,
                             n_resamples=20000, method="BCa",
                             random_state=rng).confidence_interval
        print(f"{m:<20}{x.mean():>10.4f}{x.std(ddof=1):>9.4f}{np.median(x):>9.4f}"
              f"{x.min():>8.4f}{x.max():>8.4f}   [{ci.low:.4f}, {ci.high:.4f}]")


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
    # --- sensitivity checks for the YOSO-Mask2Former pair (analytic; no random draws) ---
    d = (df[REFERENCE] - df["Mask2Former-SLAM"]).to_numpy()
    n = len(d); se = d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(0.95, n - 1)
    print()
    print("Sensitivity checks, YOSO vs Mask2Former-SLAM:")
    print(f"  t-based 90% CI of the mean difference: [{d.mean() - tq * se:+.4f}, {d.mean() + tq * se:+.4f}] m")
    lr = np.log(df[REFERENCE].to_numpy()) - np.log(df["Mask2Former-SLAM"].to_numpy())
    se_l = lr.std(ddof=1) / np.sqrt(n); b = np.log(1 + RATIO_MARGIN)
    p_lo = stats.t.sf((lr.mean() + b) / se_l, n - 1)
    p_hi = stats.t.cdf((lr.mean() - b) / se_l, n - 1)
    print(f"  geometric-mean ATE ratio YOSO/M2F = {np.exp(lr.mean()):.3f}, 90% CI "
          f"[{np.exp(lr.mean() - tq * se_l):.3f}, {np.exp(lr.mean() + tq * se_l):.3f}]; "
          f"TOST on the log ratio, bounds 1/{1 + RATIO_MARGIN:.2f}-{1 + RATIO_MARGIN:.2f}: p = {max(p_lo, p_hi):.4f}")
    beyond = int((np.abs(d) > EQUIV_MARGIN_M).sum())
    print(f"  single-sequence differences beyond +/-{EQUIV_MARGIN_M:.3f} m: {beyond} of {n} (largest: "
          f"{d.max():+.4f} m on {df.index[d.argmax()]}, {d.min():+.4f} m on {df.index[d.argmin()]})")
    low = ((df[REFERENCE] < 0.1) & (df["Mask2Former-SLAM"] < 0.1)).to_numpy()
    pooled = np.r_[df[REFERENCE].to_numpy()[low], df["Mask2Former-SLAM"].to_numpy()[low]]
    print(f"  sequences on which both methods stay below 0.1 m: {low.sum()}; median ATE there "
          f"(both methods pooled): {np.median(pooled):.4f} m")
    print("  -> the equivalence concerns the mean paired difference, not individual sequences.")
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
        d = (tt[REFERENCE] - tt[m]).to_numpy()
        ties = len(np.unique(np.abs(d[d != 0]))) < len(d[d != 0])
        if r.statistic == 0 and (d != 0).all():
            print(f"{'':<34}tied |differences|: {ties} (SciPy p is then the tie-corrected normal approximation);"
                  f" exact sign-flip p for W = 0, n = {len(d)}: {2 / 2 ** len(d):.2e}")


def ablation():
    print()
    print("=" * 78)
    print("(5) ABLATION (Table 10): aggregate means, % change, Wilcoxon vs. full (n=10)")
    print("=" * 78)
    ab = pd.read_csv(F_ABL, index_col=0)
    cfg = ["V0_ORB-SLAM3", "V1_People", "V2_People+Moving", "V3_People+Unknown",
           "Panoptic-SLAM", "YOSO_full"]
    means = {c: ab[f"ATE_{c}"].mean() for c in cfg}
    comp = cfg[:-1]
    res = {c: stats.wilcoxon(ab["ATE_YOSO_full"], ab[f"ATE_{c}"]) for c in comp}
    adj = dict(zip(comp, holm([res[c].pvalue for c in comp])))
    print(f"{'Config':<20}{'mean ATE':>10}{'mean t':>9}{'W':>5}{'p vs full':>11}{'p(Holm)':>9}")
    for c in cfg:
        if c == "YOSO_full":
            print(f"{c:<20}{means[c]:>10.4f}{ab[f'time_{c}'].mean():>9.4f}{'--':>5}{'--':>11}{'--':>9}")
        else:
            print(f"{c:<20}{means[c]:>10.4f}{ab[f'time_{c}'].mean():>9.4f}"
                  f"{res[c].statistic:>5.0f}{res[c].pvalue:>11.3f}{adj[c]:>9.3f}")
    print("Exact null distribution (no zero or tied differences at full precision);")
    print("the n=10 comparisons are exploratory.")
    pct = lambda a, b: 100 * (means[a] - means[b]) / means[a]
    print("\nPercentage change computed as (mean_before - mean_after) / mean_before:")
    print(f"  V0->V1 {pct('V0_ORB-SLAM3','V1_People'):.1f}%   V1->V2 {pct('V1_People','V2_People+Moving'):.1f}%"
          f"   V1->V3 {pct('V1_People','V3_People+Unknown'):.1f}%   V1->full {pct('V1_People','YOSO_full'):.1f}%")
    r = stats.wilcoxon(ab["ATE_V0_ORB-SLAM3"], ab["ATE_V1_People"])
    print(f"  V0 vs V1: W={r.statistic:.1f} p={r.pvalue:.3f};", end=" ")
    r = stats.wilcoxon(ab["ATE_V2_People+Moving"], ab["ATE_V3_People+Unknown"])
    print(f"V2 vs V3: W={r.statistic:.1f} p={r.pvalue:.3f} (unadjusted)")
    yc = ["V1_People", "V2_People+Moving", "V3_People+Unknown", "YOSO_full"]
    m = ab[[f"ATE_{c}" for c in yc]]
    m.columns = yc
    best = m.min(axis=1)
    print("  lowest-ATE YOSO configuration per sequence:",
          m.idxmin(axis=1).value_counts().to_dict())
    print("  worst ratio to the best YOSO configuration on any sequence:",
          {c: round(float((m[c] / best).max()), 2) for c in yc})
    v2, full = ab["ATE_V2_People+Moving"], ab["ATE_YOSO_full"]
    print(f"  V2 vs full head-to-head: V2 lower on {(v2 < full).sum()}, full lower on {(full < v2).sum()}")


REP_SEQS = ["Walking XYZ", "Walking Halfsphere", "Synchronous", "Crowd2",
            "Balloon Tracking", "Placing Non-Obs Box"]
REP_METHODS = ["YOSO-SLAM", "Mask2Former-SLAM", "YDM-SLAM", "Panoptic-SLAM"]


def _runs(rr, s, m):
    g = rr[(rr.sequence == s) & (rr.method == m)].sort_values("run")
    assert len(g) == 4, f"expected 4 runs for {s} / {m}, found {len(g)}"
    return g["ATE_RMSE_m"].to_numpy()


def repeated_runs():
    print()
    print("=" * 78)
    print("(6) REPEATED RUNS (Tables 7-8): mean/SD/CV per method and Mann-Whitney U, n=4 vs 4")
    print("=" * 78)
    rr = pd.read_csv(F_RUNS)
    assert len(rr) == 96, f"expected 96 run rows, found {len(rr)}"
    print(f"{'Sequence':<22}{'Method':<18}{'Run 1':>8}{'Run 2':>8}{'Run 3':>8}{'Run 4':>8}{'Mean':>9}{'SD':>8}{'CV%':>7}")
    for s in REP_SEQS:
        for m in REP_METHODS:
            v = _runs(rr, s, m)
            print(f"{s:<22}{m:<18}" + "".join(f"{x:>8.4f}" for x in v)
                  + f"{v.mean():>9.4f}{v.std(ddof=1):>8.4f}{100*v.std(ddof=1)/v.mean():>7.1f}")
    print()
    print(f"{'Sequence':<22}{'p vs YDM':>10}{'p vs Pan':>10}{'p vs M2F':>10}   (two-sided Mann-Whitney U, YOSO-SLAM vs baseline)")
    for s in REP_SEQS:
        y = _runs(rr, s, "YOSO-SLAM")
        ps = [stats.mannwhitneyu(y, _runs(rr, s, m), alternative="two-sided").pvalue
              for m in ("YDM-SLAM", "Panoptic-SLAM", "Mask2Former-SLAM")]
        print(f"{s:<22}" + "".join(f"{p:>10.3f}" for p in ps))
    print("\nSmallest attainable two-sided p for n=4 vs 4 is 2/70 = 0.0286.")


def tost_welch(y, m, margin):
    """Run-level TOST for two independent groups (Welch t). Returns
    (diff, ci_low, ci_high, p_tost, df) with a 90% CI (= two one-sided 95%)."""
    d = y.mean() - m.mean()
    vy, vm = y.var(ddof=1) / len(y), m.var(ddof=1) / len(m)
    se = np.sqrt(vy + vm)
    df = (vy + vm) ** 2 / (vy ** 2 / (len(y) - 1) + vm ** 2 / (len(m) - 1))
    t = stats.t.ppf(0.95, df)
    p_lower = stats.t.sf((d + margin) / se, df)   # H0: diff <= -margin
    p_upper = stats.t.cdf((d - margin) / se, df)  # H0: diff >= +margin
    return d, d - t * se, d + t * se, max(p_lower, p_upper), df


def replicated_equivalence():
    print()
    print("=" * 78)
    print(f"(7) REPLICATED EQUIVALENCE, YOSO-SLAM vs Mask2Former-SLAM  --  margin = +/-{EQUIV_MARGIN_M:.3f} m, n = 4 runs per method")
    print("=" * 78)
    rr = pd.read_csv(F_RUNS)
    print("Per sequence (run level, Welch t): diff = mean(YOSO) - mean(M2F); 90% CI; TOST p;")
    print("equivalence is concluded when the 90% CI lies within the margin (TOST p < 0.05).")
    print(f"{'Sequence':<22}{'YOSO':>8}{'M2F':>8}{'diff':>9}{'90% CI (m)':>22}{'TOST p':>9}{'df':>6}   verdict")
    seq_diffs = []
    for s in REP_SEQS:
        y, m = _runs(rr, s, "YOSO-SLAM"), _runs(rr, s, "Mask2Former-SLAM")
        d, lo, hi, p, df = tost_welch(y, m, EQUIV_MARGIN_M)
        seq_diffs.append(d)
        verdict = "equivalent" if p < 0.05 else ("YOSO lower" if hi < 0 else ("M2F lower" if lo > 0 else "not concluded"))
        print(f"{s:<22}{y.mean():>8.4f}{m.mean():>8.4f}{d:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{p:>9.4f}{df:>6.1f}   {verdict}")
    print()
    print("Relative scale (run level): geometric-mean ATE ratio YOSO/M2F with a Welch 90% CI on the")
    print(f"log run values; 'yes' when the CI lies within the ratio bounds 1/{1 + RATIO_MARGIN:.2f}-{1 + RATIO_MARGIN:.2f}.")
    print(f"{'Sequence':<22}{'ratio':>8}{'90% CI':>18}   within ratio bounds")
    for s in REP_SEQS:
        ly, lm = np.log(_runs(rr, s, "YOSO-SLAM")), np.log(_runs(rr, s, "Mask2Former-SLAM"))
        lr, lo, hi, p, _ = tost_welch(ly, lm, np.log(1 + RATIO_MARGIN))
        print(f"{s:<22}{np.exp(lr):>8.3f}    [{np.exp(lo):.3f}, {np.exp(hi):.3f}]   {'yes' if p < 0.05 else 'no'}")
    d = np.array(seq_diffs)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.95, n - 1)
    p_t, p_w = tost_paired(d, EQUIV_MARGIN_M)
    p_noninf = stats.t.cdf((d.mean() - EQUIV_MARGIN_M) / se, n - 1)
    print()
    print("Pooled over the six sequences (sequence level, paired on the run means):")
    print(f"  mean paired difference {d.mean():+.4f} m, SD {d.std(ddof=1):.4f}, 90% CI [{d.mean()-t*se:+.4f}, {d.mean()+t*se:+.4f}] m")
    print(f"  paired TOST p (t) = {p_t:.4f}; TOST p (Wilcoxon) = {p_w:.4f}; one-sided non-inferiority p (upper) = {p_noninf:.4f}")
    print(f"  two-sided paired t-test p = {stats.ttest_1samp(d, 0).pvalue:.4f}")
    print("  -> at the sequence level neither equivalence nor a difference is established with six sequences.")


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
    replicated_equivalence()


if __name__ == "__main__":
    main()
