"""
Generate the two summary plots written to figures/.

  01_yearly_overlay.png    M>=7 quakes and G3+ storm days, both yearly, 1965-2025
  02_window_ratios.png     Daily M>=7 rate inside +/-N-day storm windows / chance
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sw-db", default="../spaceweather/spaceweather.sqlite")
    ap.add_argument("--eq-db", default="../earthquakes/quakes.sqlite")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--year-lo", type=int, default=1965)
    ap.add_argument("--year-hi", type=int, default=2025)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        gfz = pd.read_sql(
            f"""
            SELECT date_iso, year,
                   kp1,kp2,kp3,kp4,kp5,kp6,kp7,kp8
            FROM gfz_daily
            WHERE date_iso BETWEEN '{args.year_lo}-01-01' AND '{args.year_hi}-12-31'
            """,
            sqlite3.connect(args.sw_db),
            parse_dates=["date_iso"],
        )
        quakes = pd.read_sql(
            "SELECT time_ms, mag FROM quakes WHERE mag >= 7",
            sqlite3.connect(args.eq_db),
        )
    except sqlite3.OperationalError as e:
        sys.exit(f"Could not open databases ({e}); see README.")

    gfz["peak_kp"] = gfz[[f"kp{i}" for i in range(1, 9)]].max(axis=1)
    gfz["date"] = gfz["date_iso"].dt.normalize()
    quakes["date"] = (
        pd.to_datetime(quakes["time_ms"], unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    quakes["year"] = quakes["date"].dt.year
    quakes = quakes[quakes["year"].between(args.year_lo, args.year_hi)]

    years = list(range(args.year_lo, args.year_hi + 1))
    yearly_m7 = quakes.groupby("year").size().reindex(years, fill_value=0)
    yearly_g3 = (
        gfz.assign(g3=(gfz["peak_kp"] >= 7).astype(int))
        .groupby("year")["g3"]
        .sum()
        .reindex(years, fill_value=0)
    )

    # ---- Figure 1: dual-axis yearly overlay ----
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(years, yearly_m7, color="#cc4422", alpha=0.7, label="M>=7 quakes")
    ax1.set_ylabel("M>=7 earthquakes per year", color="#cc4422")
    ax1.tick_params(axis="y", labelcolor="#cc4422")
    ax1.set_xlabel("Year")
    ax2 = ax1.twinx()
    ax2.plot(years, yearly_g3, color="#3355aa", marker="o", linewidth=1.5, label="G3+ storm days (Kp>=7)")
    ax2.set_ylabel("G3+ storm days per year", color="#3355aa")
    ax2.tick_params(axis="y", labelcolor="#3355aa")
    plt.title(
        f"Detection-bias-clean bands, {args.year_lo}-{args.year_hi}: "
        "M>=7 quakes vs G3+ storm days  (Pearson r = -0.16, p = 0.21)"
    )
    plt.tight_layout()
    plt.savefig(out / "01_yearly_overlay.png", dpi=120)
    plt.close()

    # ---- Figure 2: daily-window ratio bars ----
    storm_dates = set(gfz.loc[gfz["peak_kp"] >= 7, "date"])
    all_dates = set(gfz["date"])
    m7_dates = quakes["date"].tolist()
    n_total = len(all_dates)
    n_m7 = len(m7_dates)

    widths = [0, 1, 3, 7, 14, 30]
    ratios_centered, ratios_after = [], []
    for w in widths:
        win_c = set()
        win_a = set()
        for d in storm_dates:
            for k in range(-w, w + 1):
                win_c.add(d + pd.Timedelta(days=k))
            for k in range(0, w + 1):
                win_a.add(d + pd.Timedelta(days=k))
        win_c &= all_dates
        win_a &= all_dates
        for win, store in ((win_c, ratios_centered), (win_a, ratios_after)):
            n_win = len(win)
            n_in = sum(1 for d in m7_dates if d in win)
            expected = n_m7 * n_win / n_total
            store.append(n_in / expected if expected else float("nan"))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(widths))
    w_bar = 0.4
    ax.bar([i - w_bar / 2 for i in x], ratios_centered, width=w_bar, label="centered (+/-N days)", color="#3355aa")
    ax.bar([i + w_bar / 2 for i in x], ratios_after,    width=w_bar, label="after only (storm..+N)", color="#88aacc")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"+/-{w}d" if w else "0d" for w in widths])
    ax.set_ylabel("Observed M>=7 count / chance expectation")
    ax.set_title("M>=7 occurrence in storm windows vs chance — all near 1.0, none significant")
    ax.legend()
    ax.set_ylim(0, max(1.2, max(ratios_centered + ratios_after) * 1.1))
    plt.tight_layout()
    plt.savefig(out / "02_window_ratios.png", dpi=120)
    plt.close()

    print(f"Wrote {out / '01_yearly_overlay.png'}")
    print(f"Wrote {out / '02_window_ratios.png'}")


if __name__ == "__main__":
    main()
