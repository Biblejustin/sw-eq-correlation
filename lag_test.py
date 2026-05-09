"""
Sun-side lag test, accounting for Sun -> Earth propagation delay.

Treats high-sunspot and high-F10.7 days as upstream (Sun-side) "events"
and asks whether M>=7 occurrence is elevated 0..14 days LATER, with lag
windows chosen to bracket physical propagation regimes:

  +0..+0   light / X-ray flash (~8 minutes)
  +1..+1   1-day-fast CME or integrated SEP
  +2..+5   typical CME transit window (1-4 days, slow ones up to 5)
  +1..+5   the full CME-arrival window
  +1..+14  extended for delayed effects post-impact

For comparison the same lag scan is applied to G3+ storm days, which are
already AT Earth (lag 0 = impact day, no transit to account for).

The high-SSN / high-F10.7 thresholds are set to the top 1.82% of days so
the base rate matches G3+ storm days exactly, making the three event sets
directly comparable.
"""
import argparse
import sqlite3
import sys

import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sw-db", default="../spaceweather/spaceweather.sqlite")
    ap.add_argument("--eq-db", default="../earthquakes/quakes.sqlite")
    ap.add_argument("--year-lo", type=int, default=1965)
    ap.add_argument("--year-hi", type=int, default=2025)
    args = ap.parse_args()

    try:
        gfz = pd.read_sql(
            f"""
            SELECT date_iso, year, sunspot_number, f107_obs,
                   kp1,kp2,kp3,kp4,kp5,kp6,kp7,kp8
            FROM gfz_daily
            WHERE date_iso BETWEEN '{args.year_lo}-01-01' AND '{args.year_hi}-12-31'
            """,
            sqlite3.connect(args.sw_db),
            parse_dates=["date_iso"],
        )
    except sqlite3.OperationalError as e:
        sys.exit(f"Could not open {args.sw_db} ({e}); see README.")

    gfz["date"] = gfz["date_iso"].dt.normalize()
    gfz["peak_kp"] = gfz[[f"kp{i}" for i in range(1, 9)]].max(axis=1)

    quakes = pd.read_sql(
        "SELECT time_ms, mag FROM quakes WHERE mag >= 7",
        sqlite3.connect(args.eq_db),
    )
    quakes["date"] = (
        pd.to_datetime(quakes["time_ms"], unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    quakes = quakes[quakes["date"].between(f"{args.year_lo}-01-01", f"{args.year_hi}-12-31")]
    m7_dates = quakes["date"].tolist()

    all_dates = set(gfz["date"])
    n_total = len(all_dates)
    n_m7 = len(m7_dates)

    LAGS = [(0, 0), (1, 1), (2, 5), (1, 5), (1, 14)]

    def lag_test(event_dates, label):
        print(f"\n{label}  (n_events={len(event_dates)})")
        print("  lag (d)  | window | M>=7 in win | expected | ratio | binom p")
        print("  " + "-" * 70)
        for lag_lo, lag_hi in LAGS:
            win = set()
            for d in event_dates:
                for k in range(lag_lo, lag_hi + 1):
                    win.add(d + pd.Timedelta(days=k))
            win &= all_dates
            n_win = len(win)
            n_in = sum(1 for d in m7_dates if d in win)
            expected = n_m7 * n_win / n_total
            p = stats.binomtest(n_in, n=n_m7, p=n_win / n_total, alternative="greater").pvalue
            ratio = n_in / expected if expected else float("nan")
            print(f"   {lag_lo:+d}..{lag_hi:+d}   | {n_win:5d} ({n_win/n_total*100:5.2f}%) | {n_in:5d}       | {expected:6.1f}  | {ratio:.3f}x | p={p:.3f}")

    ssn_thr = gfz["sunspot_number"][gfz["sunspot_number"] >= 0].quantile(1 - 0.0182)
    f107_thr = gfz["f107_obs"][gfz["f107_obs"] >= 0].quantile(1 - 0.0182)
    print(f"Top 1.82% thresholds: SSN >= {ssn_thr:.0f}, F10.7 >= {f107_thr:.0f}")

    high_ssn = gfz.loc[gfz["sunspot_number"] >= ssn_thr, "date"].tolist()
    high_f107 = gfz.loc[gfz["f107_obs"] >= f107_thr, "date"].tolist()
    g3 = gfz.loc[gfz["peak_kp"] >= 7, "date"].tolist()

    lag_test(high_ssn,  "HIGH SUNSPOT DAYS (Sun-side; lag = transit + post-impact)")
    lag_test(high_f107, "HIGH F10.7 DAYS   (Sun-side; lag = transit + post-impact)")
    lag_test(g3,        "G3+ STORM DAYS    (already at Earth; lag 0 = impact)")


if __name__ == "__main__":
    main()
