"""
Correlation between space weather and earthquakes.

Uses the detection-bias-clean band from each source catalog:
  - Earthquakes:   M >= 7 yearly count (globally complete since ~1900, certainly 1965+)
  - Space weather: peak Kp >= 7 days/year (G3+ storm days; complete since 1932)

Compares on the 1965-2025 overlap window.

Tests:
  1. Yearly counts (raw and detrended), Pearson and Spearman
  2. Lag correlations -3y to +3y on detrended yearly series
  3. Daily-level: M>=7 occurrence inside +/-N day windows around G3+ storms
                  vs the random-day expectation (one-sided binomial)
"""
import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats


def load_data(sw_db, eq_db):
    sw_con = sqlite3.connect(sw_db)
    gfz = pd.read_sql(
        """
        SELECT date_iso, year, ap_daily, sunspot_number, f107_obs,
               kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8
        FROM gfz_daily
        """,
        sw_con,
        parse_dates=["date_iso"],
    )
    gfz["peak_kp"] = gfz[[f"kp{i}" for i in range(1, 9)]].max(axis=1)
    gfz["g3plus_day"] = (gfz["peak_kp"] >= 7).astype(int)

    eq_con = sqlite3.connect(eq_db)
    quakes = pd.read_sql("SELECT time_ms, mag FROM quakes", eq_con)
    quakes["date"] = pd.to_datetime(quakes["time_ms"], unit="ms", utc=True).dt.tz_localize(None)
    quakes["year"] = quakes["date"].dt.year
    return gfz, quakes


def yearly_correlations(gfz, quakes, year_lo=1965, year_hi=2025):
    years = list(range(year_lo, year_hi + 1))
    eq_yr = (
        quakes[quakes["year"].between(year_lo, year_hi)]
        .groupby("year")
        .agg(
            m6=("mag", lambda s: (s >= 6).sum()),
            m7=("mag", lambda s: (s >= 7).sum()),
            m8=("mag", lambda s: (s >= 8).sum()),
        )
        .reindex(years, fill_value=0)
    )
    sw_yr = (
        gfz[gfz["year"].between(year_lo, year_hi)]
        .groupby("year")
        .agg(
            ap_mean=("ap_daily", "mean"),
            ssn_mean=("sunspot_number", lambda s: s[s >= 0].mean()),
            f107_mean=("f107_obs", lambda s: s[s >= 0].mean()),
            g3plus_days=("g3plus_day", "sum"),
        )
        .reindex(years)
    )
    return eq_yr.join(sw_yr)


def pr(a, b, label):
    mask = a.notna() & b.notna()
    r, p = stats.pearsonr(a[mask], b[mask])
    rs, ps = stats.spearmanr(a[mask], b[mask])
    print(f"  {label:48s} r={r:+.3f} [p={p:.3f}]  rho={rs:+.3f} [p={ps:.3f}]")


def detrend(s):
    x = np.array(s.index, dtype=float)
    y = s.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return s
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return s - (slope * x + intercept)


def daily_window_test(gfz, quakes, year_lo=1965, year_hi=2025):
    gfz_d = gfz[gfz["date_iso"].between(f"{year_lo}-01-01", f"{year_hi}-12-31")].copy()
    gfz_d["date"] = gfz_d["date_iso"].dt.normalize()
    storm_dates = set(gfz_d.loc[gfz_d["g3plus_day"] == 1, "date"])
    all_dates = set(gfz_d["date"])

    m7_dates = (
        quakes.loc[(quakes["mag"] >= 7) & (quakes["year"].between(year_lo, year_hi)), "date"]
        .dt.normalize()
        .tolist()
    )

    n_total = len(all_dates)
    print(f"Window: {year_lo}-01-01..{year_hi}-12-31  ({n_total} days)")
    print(f"G3+ storm days: {len(storm_dates)} ({len(storm_dates)/n_total*100:.2f}%)")
    print(f"M>=7 events:    {len(m7_dates)}")
    print()

    def window(width, side):
        win = set()
        for d in storm_dates:
            if side == "centered":
                ks = range(-width, width + 1)
            else:  # "after"
                ks = range(0, width + 1)
            for k in ks:
                win.add(d + pd.Timedelta(days=k))
        return win & all_dates

    print("Window centered on storm | size (% of all days) | M>=7 in win | expected | ratio | binom p")
    print("-" * 95)
    for w in (0, 1, 3, 7, 14, 30):
        win = window(w, "centered")
        n_win = len(win)
        n_in = sum(1 for d in m7_dates if d in win)
        expected = len(m7_dates) * n_win / n_total
        p = stats.binomtest(n_in, n=len(m7_dates), p=n_win / n_total, alternative="greater").pvalue
        ratio = n_in / expected if expected else float("nan")
        print(f"  +/- {w:2d} d                | {n_win:5d} ({n_win/n_total*100:5.2f}%)   | {n_in:5d}      | {expected:6.1f}  | {ratio:.3f}x | p={p:.3f}")

    print("\nDays AFTER storm only (storm..+w):")
    print("-" * 95)
    for w in (0, 1, 3, 7, 14, 30):
        win = window(w, "after")
        n_win = len(win)
        n_in = sum(1 for d in m7_dates if d in win)
        expected = len(m7_dates) * n_win / n_total
        p = stats.binomtest(n_in, n=len(m7_dates), p=n_win / n_total, alternative="greater").pvalue
        ratio = n_in / expected if expected else float("nan")
        print(f"  storm..+{w:2d}d            | {n_win:5d} ({n_win/n_total*100:5.2f}%)   | {n_in:5d}      | {expected:6.1f}  | {ratio:.3f}x | p={p:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sw-db", default="../spaceweather/spaceweather.sqlite",
                    help="Path to spaceweather.sqlite (built by Biblejustin/spaceweather)")
    ap.add_argument("--eq-db", default="../earthquakes/quakes.sqlite",
                    help="Path to quakes.sqlite (built by Biblejustin/earthquakes)")
    ap.add_argument("--year-lo", type=int, default=1965)
    ap.add_argument("--year-hi", type=int, default=2025)
    args = ap.parse_args()

    try:
        gfz, quakes = load_data(args.sw_db, args.eq_db)
    except sqlite3.OperationalError as e:
        sys.exit(
            f"Could not open databases ({e}). "
            f"Build them first: see README for clone+fetch instructions."
        )

    print("=" * 80)
    print(f"YEARLY-COUNT ANALYSIS ({args.year_lo}-{args.year_hi})")
    print("=" * 80)
    df = yearly_correlations(gfz, quakes, args.year_lo, args.year_hi)
    print(df.head(3).to_string())
    print("...")
    print(df.tail(3).to_string())

    print("\nPearson r and Spearman rho (yearly), p in []")
    print("-" * 80)
    pr(df["m7"], df["g3plus_days"], "M>=7 quakes vs G3+ storm days (Kp>=7)")
    pr(df["m7"], df["ap_mean"],     "M>=7 quakes vs mean daily Ap")
    pr(df["m7"], df["ssn_mean"],    "M>=7 quakes vs mean sunspot number")
    pr(df["m7"], df["f107_mean"],   "M>=7 quakes vs mean F10.7")
    pr(df["m6"], df["g3plus_days"], "M>=6 quakes vs G3+ storm days  (NB: M6 not detection-clean)")
    pr(df["m8"], df["g3plus_days"], "M>=8 quakes vs G3+ storm days  (NB: small-n)")

    print("\nDetrended (linear residuals on year):")
    print("-" * 80)
    pr(detrend(df["m7"]), detrend(df["g3plus_days"]),  "M>=7 vs G3+ days   (detrended)")
    pr(detrend(df["m7"]), detrend(df["ssn_mean"]),     "M>=7 vs sunspot    (detrended)")
    pr(detrend(df["m7"]), detrend(df["ap_mean"]),      "M>=7 vs Ap         (detrended)")

    print("\nLag correlations on detrended series (positive lag = SW leads EQ by k years):")
    print("-" * 80)
    dt_m7 = detrend(df["m7"])
    dt_g3 = detrend(df["g3plus_days"])
    for lag in range(-3, 4):
        a = dt_m7
        b = dt_g3.shift(lag)
        mask = a.notna() & b.notna()
        r, p = stats.pearsonr(a[mask], b[mask])
        arrow = ">" if lag > 0 else ("<" if lag < 0 else "=")
        print(f"  lag {lag:+d}y (SW {arrow} EQ): r={r:+.3f} [p={p:.3f}, n={mask.sum()}]")

    print("\n" + "=" * 80)
    print("DAILY-LEVEL TEST: M>=7 rate inside windows around G3+ storms")
    print("=" * 80)
    daily_window_test(gfz, quakes, args.year_lo, args.year_hi)
    print("\nDone.")


if __name__ == "__main__":
    main()
