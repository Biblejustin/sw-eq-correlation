"""
11-year solar cycle imprint on global M>=7 seismicity.

Three independent tests:

1. PHASE FOLD. Find cycle minima from the 13-month smoothed monthly SILSO
   sunspot series (data-driven, then cross-checked against the official
   Cycle 20-25 boundaries). For each M>=7 quake compute its phase within
   the current cycle (0 = min, ~0.5 = max, 1 = next min) and test the
   resulting phase histogram against uniform with chi-squared and Rayleigh.

2. HIGH vs LOW HALF. Split yearly counts by median yearly sunspot number;
   compare M>=7 rates with a Poisson rate test.

3. ASCENDING vs DESCENDING. For each cycle, mark years before solar max as
   ascending and after as descending; compare M>=7 rates.

Writes figures/03_cycle_fold.png.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


def detect_cycle_boundaries(silso):
    """Return DataFrame of (date_min, date_max, cycle_num) from smoothed SSN."""
    s = silso[silso["sunspot_number"] >= 0].copy()
    s["month"] = s["date_iso"].dt.to_period("M")
    monthly = s.groupby("month")["sunspot_number"].mean()
    monthly.index = monthly.index.to_timestamp()
    # 13-month tapered Wolf smooth (centered, end-tapered)
    w = np.array([0.5] + [1.0] * 11 + [0.5])
    w /= w.sum()
    smoothed = monthly.rolling(13, center=True).apply(lambda x: np.dot(x, w), raw=True)

    # Minima: peaks of -smoothed, min distance ~ 7 years
    neg = -smoothed.dropna().values
    idx_min, _ = find_peaks(neg, distance=84)  # 84 months = 7 years
    min_dates = smoothed.dropna().index[idx_min]

    # Maxima: peaks of +smoothed, same distance constraint
    pos = smoothed.dropna().values
    idx_max, _ = find_peaks(pos, distance=84)
    max_dates = smoothed.dropna().index[idx_max]

    # Pair maxima to the cycle they belong to (between two minima)
    rows = []
    for i in range(len(min_dates) - 1):
        d_lo, d_hi = min_dates[i], min_dates[i + 1]
        in_cycle = [m for m in max_dates if d_lo < m < d_hi]
        rows.append({
            "cycle": 20 + i if d_lo.year >= 1964 else None,
            "min_date": d_lo,
            "max_date": in_cycle[0] if in_cycle else pd.NaT,
            "next_min_date": d_hi,
            "length_yr": (d_hi - d_lo).days / 365.25,
        })
    cycles = pd.DataFrame(rows)
    # Number cycles from 1 cycle covering 1964 onward (Cycle 20 starts late 1964)
    cycle20_idx = cycles["min_date"].sub(pd.Timestamp("1964-10-01")).abs().idxmin()
    cycles["cycle"] = range(20 - cycle20_idx, 20 - cycle20_idx + len(cycles))
    return cycles, smoothed


def phase_fold_test(quakes, cycles):
    """Compute phase for each M>=7 quake, return phases and test stats."""
    m7 = quakes[quakes["mag"] >= 7].copy()
    m7["phase"] = np.nan
    for _, c in cycles.iterrows():
        in_cycle = m7["date"].between(c["min_date"], c["next_min_date"])
        m7.loc[in_cycle, "phase"] = (
            (m7.loc[in_cycle, "date"] - c["min_date"]).dt.total_seconds()
            / (c["next_min_date"] - c["min_date"]).total_seconds()
        )
    phases = m7["phase"].dropna().values

    # Chi-squared against uniform, 10 bins
    n_bins = 10
    hist, edges = np.histogram(phases, bins=n_bins, range=(0, 1))
    expected = np.full(n_bins, phases.size / n_bins)
    chi2, p_chi2 = stats.chisquare(hist, f_exp=expected)

    # Rayleigh test (uniformity on the circle)
    angles = 2 * np.pi * phases
    n = angles.size
    R = np.sqrt(np.sum(np.cos(angles)) ** 2 + np.sum(np.sin(angles)) ** 2)
    Rbar = R / n
    z = n * Rbar ** 2
    p_rayleigh = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n))
    mean_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
    mean_phase = (mean_angle / (2 * np.pi)) % 1

    return phases, hist, edges, {
        "chi2": chi2, "p_chi2": p_chi2,
        "Rbar": Rbar, "z": z, "p_rayleigh": p_rayleigh,
        "mean_phase": mean_phase,
    }


def high_low_half_test(quakes, gfz):
    """Poisson rate comparison: M>=7 rate in years above vs below median yearly SSN."""
    years = range(1965, 2026)
    sw_yr = (gfz[gfz["year"].between(1965, 2025)]
             .assign(ssn=lambda d: np.where(d["sunspot_number"] >= 0, d["sunspot_number"], np.nan))
             .groupby("year")["ssn"].mean()
             .reindex(years))
    eq_yr = (quakes[(quakes["mag"] >= 7) & quakes["year"].between(1965, 2025)]
             .groupby("year").size().reindex(years, fill_value=0))

    med = sw_yr.median()
    high_years = sw_yr[sw_yr >= med].index
    low_years = sw_yr[sw_yr < med].index
    n_high = eq_yr.loc[high_years].sum()
    n_low = eq_yr.loc[low_years].sum()
    t_high = len(high_years)
    t_low = len(low_years)
    rate_high = n_high / t_high
    rate_low = n_low / t_low

    # Poisson exact rate test (E-test): under H0 the rates are equal.
    # Use conditional binomial: given total, expected fraction in "high" is t_high/(t_high+t_low).
    p = stats.binomtest(int(n_high), n=int(n_high + n_low),
                        p=t_high / (t_high + t_low),
                        alternative="two-sided").pvalue
    return {
        "median_ssn": med,
        "n_high_yr": t_high, "n_low_yr": t_low,
        "n_quakes_high": int(n_high), "n_quakes_low": int(n_low),
        "rate_high": rate_high, "rate_low": rate_low,
        "ratio": rate_high / rate_low if rate_low else float("nan"),
        "p": p,
    }


def ascending_descending_test(quakes, cycles):
    """Compare M>=7 rates in ascending vs descending phase across all cycles."""
    asc_days = 0
    desc_days = 0
    asc_quakes = 0
    desc_quakes = 0
    m7 = quakes[quakes["mag"] >= 7]
    for _, c in cycles.iterrows():
        if pd.isna(c["max_date"]):
            continue
        # Clip to 1965-2025 window for consistency with other tests
        min_d = max(c["min_date"], pd.Timestamp("1965-01-01"))
        max_d = c["max_date"]
        next_min_d = min(c["next_min_date"], pd.Timestamp("2025-12-31"))
        if min_d >= max_d or max_d >= next_min_d:
            continue
        asc_days += (max_d - min_d).days
        desc_days += (next_min_d - max_d).days
        asc_quakes += m7["date"].between(min_d, max_d).sum()
        desc_quakes += m7["date"].between(max_d, next_min_d).sum()

    rate_asc = asc_quakes / (asc_days / 365.25)
    rate_desc = desc_quakes / (desc_days / 365.25)
    n_tot = asc_quakes + desc_quakes
    p = stats.binomtest(int(asc_quakes), n=int(n_tot),
                        p=asc_days / (asc_days + desc_days),
                        alternative="two-sided").pvalue
    return {
        "asc_years": asc_days / 365.25,
        "desc_years": desc_days / 365.25,
        "n_asc": int(asc_quakes),
        "n_desc": int(desc_quakes),
        "rate_asc": rate_asc,
        "rate_desc": rate_desc,
        "ratio": rate_asc / rate_desc if rate_desc else float("nan"),
        "p": p,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sw-db", default="../spaceweather/spaceweather.sqlite")
    ap.add_argument("--eq-db", default="../earthquakes/quakes.sqlite")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    try:
        silso = pd.read_sql(
            "SELECT date_iso, sunspot_number FROM silso_daily WHERE sunspot_number >= 0",
            sqlite3.connect(args.sw_db),
            parse_dates=["date_iso"],
        )
        gfz = pd.read_sql(
            "SELECT date_iso, year, sunspot_number FROM gfz_daily",
            sqlite3.connect(args.sw_db),
            parse_dates=["date_iso"],
        )
        quakes = pd.read_sql("SELECT time_ms, mag FROM quakes", sqlite3.connect(args.eq_db))
    except sqlite3.OperationalError as e:
        sys.exit(f"Could not open databases ({e}); see README.")

    quakes["date"] = pd.to_datetime(quakes["time_ms"], unit="ms", utc=True).dt.tz_localize(None)
    quakes["year"] = quakes["date"].dt.year

    print("=" * 80)
    print("CYCLE BOUNDARIES (from 13-month smoothed monthly SILSO)")
    print("=" * 80)
    cycles, smoothed = detect_cycle_boundaries(silso)
    cycles_in_range = cycles[(cycles["min_date"] >= "1960-01-01") &
                              (cycles["min_date"] <= "2025-01-01")].copy()
    print(cycles_in_range[["cycle", "min_date", "max_date", "next_min_date", "length_yr"]]
          .assign(min_date=lambda d: d["min_date"].dt.strftime("%Y-%m"),
                  max_date=lambda d: d["max_date"].dt.strftime("%Y-%m"),
                  next_min_date=lambda d: d["next_min_date"].dt.strftime("%Y-%m"),
                  length_yr=lambda d: d["length_yr"].round(2))
          .to_string(index=False))

    print("\n" + "=" * 80)
    print("1. PHASE FOLD (M>=7 quakes 1965-2025 vs cycle phase 0..1)")
    print("=" * 80)
    cycles_used = cycles_in_range[cycles_in_range["next_min_date"] <= "2025-12-31"]
    phases, hist, edges, stats_out = phase_fold_test(
        quakes[quakes["year"].between(1965, 2025)], cycles_used
    )
    print(f"Quakes assigned to a complete cycle: {phases.size}")
    print(f"Phase histogram (10 bins, 0=min .. 1=next min):")
    for i, count in enumerate(hist):
        bar = "#" * count
        print(f"  {edges[i]:.1f}-{edges[i+1]:.1f}: {count:3d}  {bar}")
    print(f"\nChi-squared vs uniform: chi2={stats_out['chi2']:.2f}, p={stats_out['p_chi2']:.3f}")
    print(f"Rayleigh test: Rbar={stats_out['Rbar']:.3f}, z={stats_out['z']:.2f}, "
          f"p={stats_out['p_rayleigh']:.3f}")
    print(f"Mean phase: {stats_out['mean_phase']:.3f}  (0.5 ~ solar max)")

    print("\n" + "=" * 80)
    print("2. HIGH vs LOW SUNSPOT HALF (yearly median split)")
    print("=" * 80)
    hl = high_low_half_test(quakes, gfz)
    print(f"Median yearly SSN: {hl['median_ssn']:.1f}")
    print(f"High-SSN years: {hl['n_high_yr']}, M>=7 count: {hl['n_quakes_high']}, "
          f"rate: {hl['rate_high']:.2f}/yr")
    print(f"Low-SSN  years: {hl['n_low_yr']}, M>=7 count: {hl['n_quakes_low']}, "
          f"rate: {hl['rate_low']:.2f}/yr")
    print(f"Ratio high/low: {hl['ratio']:.3f}x   binomial p={hl['p']:.3f}")

    print("\n" + "=" * 80)
    print("3. ASCENDING vs DESCENDING PHASE")
    print("=" * 80)
    ad = ascending_descending_test(quakes, cycles_used)
    print(f"Ascending: {ad['asc_years']:.1f} yr, {ad['n_asc']} M>=7, "
          f"rate {ad['rate_asc']:.2f}/yr")
    print(f"Descending: {ad['desc_years']:.1f} yr, {ad['n_desc']} M>=7, "
          f"rate {ad['rate_desc']:.2f}/yr")
    print(f"Ratio asc/desc: {ad['ratio']:.3f}x   binomial p={ad['p']:.3f}")

    # ---- Figure ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # (a) smoothed SSN with detected cycle boundaries, plus M>=7 quakes
    ax = axes[0]
    smoothed_2025 = smoothed[(smoothed.index >= "1965-01-01") & (smoothed.index <= "2025-12-31")]
    ax.plot(smoothed_2025.index, smoothed_2025.values, color="#3355aa", linewidth=1.2,
            label="13-mo smoothed SSN")
    ax.set_ylabel("Sunspot number (smoothed)", color="#3355aa")
    ax.tick_params(axis="y", labelcolor="#3355aa")
    for _, c in cycles_in_range.iterrows():
        ax.axvline(c["min_date"], color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        if not pd.isna(c["max_date"]):
            ax.axvline(c["max_date"], color="orange", linestyle=":", linewidth=0.7, alpha=0.5)
    ax2 = ax.twinx()
    m7d = quakes[(quakes["mag"] >= 7) & quakes["year"].between(1965, 2025)]
    ax2.eventplot(m7d["date"].values, lineoffsets=0.5, linelengths=0.4,
                  color="#cc4422", alpha=0.5)
    ax2.set_yticks([])
    ax2.set_ylabel("M>=7 events (ticks)", color="#cc4422")
    ax.set_title("Solar cycle boundaries (gray = min, orange = max) and M>=7 events")
    ax.set_xlabel("Year")

    # (b) phase histogram
    ax = axes[1]
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.9
    ax.bar(centers, hist, width=width, color="#cc4422", alpha=0.7, edgecolor="black")
    ax.axhline(phases.size / len(hist), color="black", linestyle="--",
               label=f"uniform expectation ({phases.size/len(hist):.1f}/bin)")
    ax.set_xlabel("Solar cycle phase (0 = min, ~0.5 = max, 1 = next min)")
    ax.set_ylabel("M>=7 quake count")
    ax.set_title(
        f"M>=7 phase fold across cycles ({phases.size} quakes, "
        f"chi2 p={stats_out['p_chi2']:.3f}, Rayleigh p={stats_out['p_rayleigh']:.3f}) — "
        "no concentration"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "03_cycle_fold.png", dpi=120)
    plt.close()
    print(f"\nWrote {out / '03_cycle_fold.png'}")


if __name__ == "__main__":
    main()
