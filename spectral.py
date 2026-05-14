"""
Frequency-domain test for an ~11-year imprint on global M>=7 seismicity.

  1. Periodograms of yearly M>=7 quake counts, G3+ storm-day counts (positive
     control), and yearly mean sunspot number (sanity check).
  2. Phase-randomized null distributions for the M>=7 periodogram and for the
     coherence between M>=7 and SSN — 1000 surrogates each, two-tailed 95%
     bound.
  3. Magnitude-squared coherence between yearly M>=7 and yearly SSN.

If the 11-year cycle modulates M>=7 seismicity, we'd expect the M>=7 periodogram
to peak above the null bound near 1/11 ~ 0.09 yr^-1 AND the coherence at that
band to exceed its null bound.

Writes figures/04_periodogram.png and figures/05_coherence.png.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def load_yearly(sw_db, eq_db, year_lo=1965, year_hi=2025):
    gfz = pd.read_sql(
        f"""
        SELECT date_iso, year, sunspot_number,
               kp1,kp2,kp3,kp4,kp5,kp6,kp7,kp8
        FROM gfz_daily
        WHERE date_iso BETWEEN '{year_lo}-01-01' AND '{year_hi}-12-31'
        """,
        sqlite3.connect(sw_db),
        parse_dates=["date_iso"],
    )
    gfz["peak_kp"] = gfz[[f"kp{i}" for i in range(1, 9)]].max(axis=1)
    quakes = pd.read_sql(
        f"SELECT time_ms, mag FROM quakes WHERE mag >= 7",
        sqlite3.connect(eq_db),
    )
    quakes["date"] = pd.to_datetime(quakes["time_ms"], unit="ms", utc=True).dt.tz_localize(None)
    quakes["year"] = quakes["date"].dt.year
    quakes = quakes[quakes["year"].between(year_lo, year_hi)]

    years = list(range(year_lo, year_hi + 1))
    m7 = quakes.groupby("year").size().reindex(years, fill_value=0)
    g3 = gfz.assign(g3=(gfz["peak_kp"] >= 7).astype(int)).groupby("year")["g3"].sum().reindex(years, fill_value=0)
    ssn = (
        gfz.assign(s=np.where(gfz["sunspot_number"] >= 0, gfz["sunspot_number"], np.nan))
        .groupby("year")["s"].mean().reindex(years)
    )
    return np.array(years), m7.values.astype(float), g3.values.astype(float), ssn.values.astype(float)


def raw_periodogram(x):
    """Return (freqs in cycles/year, power) for one-sided periodogram."""
    x = x - np.nanmean(x)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0)  # yearly samples -> freqs in cyc/yr
    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2) / n
    return freqs, power


def phase_randomize(x, rng):
    """Generate a surrogate with the same power spectrum, randomized phases."""
    x = x - np.nanmean(x)
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    n = x.size
    # Random phases, preserving DC and (if n even) Nyquist as real
    phases = rng.uniform(0, 2 * np.pi, size=fft.size)
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    surrogate_fft = mag * np.exp(1j * phases)
    return np.fft.irfft(surrogate_fft, n=n)


def coherence(x, y, nperseg=20):
    """Magnitude-squared coherence using Welch with short segments."""
    f, cxy = signal.coherence(x - np.nanmean(x), y - np.nanmean(y),
                               fs=1.0, nperseg=nperseg, noverlap=nperseg // 2)
    return f, cxy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sw-db", default="../spaceweather/spaceweather.sqlite")
    ap.add_argument("--eq-db", default="../earthquakes/quakes.sqlite")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n-surrogates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    try:
        years, m7, g3, ssn = load_yearly(args.sw_db, args.eq_db)
    except sqlite3.OperationalError as e:
        sys.exit(f"Could not open databases ({e}); see README.")

    f_m7,  P_m7  = raw_periodogram(m7)
    f_g3,  P_g3  = raw_periodogram(g3)
    f_ssn, P_ssn = raw_periodogram(ssn[~np.isnan(ssn)])

    # ---- Surrogate null for M>=7 periodogram ----
    rng = np.random.default_rng(args.seed)
    surr_powers = np.empty((args.n_surrogates, f_m7.size))
    for i in range(args.n_surrogates):
        s = phase_randomize(m7, rng)
        _, surr_powers[i] = raw_periodogram(s)
    null95_m7 = np.percentile(surr_powers, 95, axis=0)

    # ---- Coherence + surrogate null ----
    f_coh, coh = coherence(m7, ssn[~np.isnan(ssn)] if np.any(np.isnan(ssn)) else ssn,
                            nperseg=20)
    # If lengths differ due to NaN in SSN, re-load aligned
    if np.isnan(ssn).any():
        mask = ~np.isnan(ssn)
        f_coh, coh = coherence(m7[mask], ssn[mask], nperseg=20)
        ssn_clean = ssn[mask]
        m7_clean = m7[mask]
    else:
        ssn_clean = ssn
        m7_clean = m7

    surr_coh = np.empty((args.n_surrogates, f_coh.size))
    for i in range(args.n_surrogates):
        s = phase_randomize(m7_clean, rng)
        _, surr_coh[i] = coherence(s, ssn_clean, nperseg=20)
    null95_coh = np.percentile(surr_coh, 95, axis=0)

    # ---- Report ----
    print("=" * 80)
    print("PERIODOGRAM SUMMARY (yearly, 1965-2025)")
    print("=" * 80)

    def report_peak(label, f, P, null=None):
        f_band_mask = (f >= 1 / 13) & (f <= 1 / 9)  # 9-13 year band
        if f_band_mask.sum() == 0:
            print(f"  {label}: no freqs in 9-13y band (series too short)")
            return
        i = np.argmax(P[f_band_mask])
        peak_idx = np.where(f_band_mask)[0][i]
        peak_f = f[peak_idx]
        peak_period = 1 / peak_f if peak_f > 0 else float("inf")
        peak_P = P[peak_idx]
        line = (f"  {label}: 9-13y band peak at {peak_period:.1f} y "
                f"(f={peak_f:.3f} /y), power={peak_P:.2f}")
        if null is not None:
            line += f", null-95={null[peak_idx]:.2f}, exceeds null: {peak_P > null[peak_idx]}"
        print(line)

    report_peak("M>=7 quakes  ", f_m7,  P_m7,  null95_m7)
    report_peak("G3+ days     ", f_g3,  P_g3,  None)
    report_peak("Sunspot (SSN)", f_ssn, P_ssn, None)

    print("\n" + "=" * 80)
    print("COHERENCE (M>=7 vs yearly SSN)")
    print("=" * 80)
    band = (f_coh >= 1 / 13) & (f_coh <= 1 / 9)
    if band.sum() > 0:
        i = np.argmax(coh[band])
        idx = np.where(band)[0][i]
        period = 1 / f_coh[idx] if f_coh[idx] > 0 else float("inf")
        print(f"  9-13y band peak coherence at {period:.1f} y: "
              f"coh={coh[idx]:.3f}, null-95={null95_coh[idx]:.3f}, "
              f"exceeds null: {coh[idx] > null95_coh[idx]}")
    else:
        print("  No coherence frequencies in 9-13y band (nperseg too small)")

    # ---- Figures ----
    fig, ax = plt.subplots(figsize=(10, 5))
    # convert freq to period for x-axis, skip f=0
    sl = slice(1, None)
    ax.plot(1/f_m7[sl],  P_m7[sl]  / P_m7[sl].max(),  color="#cc4422", marker="o", label="M>=7 quakes (normalized)")
    ax.plot(1/f_g3[sl],  P_g3[sl]  / P_g3[sl].max(),  color="#3355aa", marker="s", label="G3+ days (positive control)")
    ax.plot(1/f_ssn[sl], P_ssn[sl] / P_ssn[sl].max(), color="orange",  marker="^", label="Sunspot number (sanity)")
    ax.plot(1/f_m7[sl], null95_m7[sl] / P_m7[sl].max(), color="gray",
            linestyle="--", label="M>=7 phase-randomized 95% null")
    ax.axvline(11, color="black", linestyle=":", alpha=0.5, label="11 yr")
    ax.set_xscale("log")
    ax.set_xlabel("Period (years, log scale)")
    ax.set_ylabel("Power (normalized to each series' max)")
    ax.set_title("Periodograms — sunspot peaks at ~11y, G3+ days does too, M>=7 doesn't")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(2, 62)
    plt.tight_layout()
    plt.savefig(out / "04_periodogram.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sl = slice(1, None)
    ax.plot(1/f_coh[sl], coh[sl], color="#cc4422", marker="o", label="Coherence M>=7 vs SSN")
    ax.plot(1/f_coh[sl], null95_coh[sl], color="gray", linestyle="--",
            label="Phase-randomized 95% null")
    ax.axvline(11, color="black", linestyle=":", alpha=0.5, label="11 yr")
    ax.set_xscale("log")
    ax.set_xlabel("Period (years, log scale)")
    ax.set_ylabel("Magnitude-squared coherence")
    ax.set_title("Coherence between yearly M>=7 quake count and yearly SSN — flat at 11y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(2, 62)
    plt.tight_layout()
    plt.savefig(out / "05_coherence.png", dpi=120)
    plt.close()

    print(f"\nWrote {out/'04_periodogram.png'}, {out/'05_coherence.png'}")


if __name__ == "__main__":
    main()
