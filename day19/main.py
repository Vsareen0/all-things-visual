"""
Day 19 Exercise: HDR Transfer Functions — PQ, HLG, and Gamma 2.4
=================================================================
Compare how PQ (ST 2084), HLG (ARIB STD-B67), and SDR gamma
allocate code values across the luminance range.

Run:  python day19_hdr_pq_hlg.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ── PQ (ST 2084) Constants ──────────────────────────────────────
m1 = 0.1593017578125      # 2610 / 16384
m2 = 78.84375             # 2523 / 4096 * 128
c1 = 0.8359375            # 3424 / 4096
c2 = 18.8515625           # 2413 / 4096 * 32
c3 = 18.6875              # 2392 / 4096 * 32
L_MAX_PQ = 10000.0        # PQ peak luminance in nits


def pq_eotf(N):
    """PQ code value (0-1) -> linear luminance (0-10000 nits)."""
    Np = np.power(np.clip(N, 0, 1), 1.0 / m2)
    numerator = np.maximum(Np - c1, 0)
    denominator = c2 - c3 * Np
    L = L_MAX_PQ * np.power(numerator / denominator, 1.0 / m1)
    return L


def pq_oetf_inverse(L):
    """Linear luminance (0-10000 nits) -> PQ code value (0-1).
    This is the inverse EOTF, i.e., the encoding curve."""
    Y = np.clip(L / L_MAX_PQ, 0, 1)
    Ym1 = np.power(Y, m1)
    numerator = c1 + c2 * Ym1
    denominator = 1 + c3 * Ym1
    N = np.power(numerator / denominator, m2)
    return N


# ── HLG (ARIB STD-B67) ─────────────────────────────────────────
HLG_A = 0.17883277
HLG_B = 1 - 4 * HLG_A  # 0.28466892
HLG_C = 0.5 - HLG_A * np.log(4 * HLG_A)  # 0.55991073


def hlg_oetf(E):
    """Scene-referred linear (0-1) -> HLG signal value (0-1).
    Note: HLG is relative, so E is normalized scene light, not nits."""
    out = np.where(
        E <= 1.0 / 12.0,
        np.sqrt(3.0 * E),
        HLG_A * np.log(12.0 * E - HLG_B) + HLG_C
    )
    return np.clip(out, 0, 1)


# ── SDR Gamma ───────────────────────────────────────────────────
def gamma_oetf(L, gamma=2.4):
    """Linear (0-1) -> gamma-encoded value (0-1)."""
    return np.power(np.clip(L, 0, 1), 1.0 / gamma)


# ── Main ────────────────────────────────────────────────────────
def main():
    # --- Part 1: Plot the three encoding curves ---
    linear = np.linspace(0, 1, 4096)

    pq_encoded = pq_oetf_inverse(linear * L_MAX_PQ)  # map 0-1 linear to 0-10000 nits
    hlg_encoded = hlg_oetf(linear)
    gamma_encoded = gamma_oetf(linear)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: full range (linear x-axis)
    ax1 = axes[0]
    ax1.plot(linear, pq_encoded, label="PQ (ST 2084)", color="#e74c3c", linewidth=2)
    ax1.plot(linear, hlg_encoded, label="HLG (ARIB STD-B67)", color="#2ecc71", linewidth=2)
    ax1.plot(linear, gamma_encoded, label="Gamma 2.4 (SDR)", color="#3498db", linewidth=2)
    ax1.plot(linear, linear, "--", label="Linear (no curve)", color="#888", linewidth=1)
    ax1.set_xlabel("Normalized Linear Luminance")
    ax1.set_ylabel("Encoded Code Value (normalized 0-1)")
    ax1.set_title("Transfer Functions — Full Range")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Right panel: log x-axis to see shadow detail allocation
    ax2 = axes[1]
    linear_log = np.logspace(-4, 0, 4096)  # 0.0001 to 1.0
    pq_log = pq_oetf_inverse(linear_log * L_MAX_PQ)
    hlg_log = hlg_oetf(linear_log)
    gamma_log = gamma_oetf(linear_log)

    ax2.semilogx(linear_log, pq_log, label="PQ", color="#e74c3c", linewidth=2)
    ax2.semilogx(linear_log, hlg_log, label="HLG", color="#2ecc71", linewidth=2)
    ax2.semilogx(linear_log, gamma_log, label="Gamma 2.4", color="#3498db", linewidth=2)
    ax2.set_xlabel("Normalized Linear Luminance (log scale)")
    ax2.set_ylabel("Encoded Code Value (normalized 0-1)")
    ax2.set_title("Shadow Region Detail (Log Scale)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("day19_transfer_curves.png", dpi=150)
    plt.show()
    print("Saved: day19_transfer_curves.png")

    # --- Part 2: Quantization step-size analysis (10-bit) ---
    print("\n" + "=" * 65)
    print("10-BIT QUANTIZATION: STEP SIZE IN NITS AT VARIOUS LUMINANCES")
    print("=" * 65)

    bits = 10
    levels = 2 ** bits  # 1024
    code_values = np.arange(levels) / (levels - 1)  # 0 to 1

    # Decode each code value to nits via PQ
    pq_nits = pq_eotf(code_values)
    pq_steps = np.diff(pq_nits)

    # For gamma, map to 0-100 nit SDR range
    sdr_peak = 100.0
    gamma_nits = np.power(code_values, 2.4) * sdr_peak
    gamma_steps = np.diff(gamma_nits)

    probe_nits = [0.1, 1, 10, 100, 1000, 5000]
    print(f"\n{'Luminance':>12}  {'PQ step (nits)':>16}  {'Gamma step (nits)':>18}  {'PQ code':>10}")
    print("-" * 65)
    for target in probe_nits:
        # Find nearest PQ code value
        pq_idx = np.argmin(np.abs(pq_nits - target))
        pq_step = pq_steps[min(pq_idx, len(pq_steps) - 1)]

        # Gamma only goes to 100 nits
        if target <= sdr_peak:
            g_idx = np.argmin(np.abs(gamma_nits - target))
            g_step = gamma_steps[min(g_idx, len(gamma_steps) - 1)]
            g_str = f"{g_step:.4f}"
        else:
            g_str = "N/A (>100)"

        print(f"{target:>10.1f} cd/m2  {pq_step:>14.4f}    {g_str:>16}  {pq_idx:>10}")

    print("\nKey insight: PQ uses tiny steps in shadows (where your eyes are")
    print("most sensitive) and larger steps in highlights (where you can't")
    print("tell the difference). This is *perceptual* quantization at work.")

    # --- Part 3: Code-value density histogram ---
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    # How many code values fall in each luminance decade?
    decades = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10000)]
    pq_counts = []
    labels = []
    for lo, hi in decades:
        count = np.sum((pq_nits >= lo) & (pq_nits < hi))
        pq_counts.append(count)
        labels.append(f"{lo}-{hi}")

    bars = ax3.bar(labels, pq_counts, color="#e74c3c", alpha=0.8, edgecolor="white")
    ax3.set_xlabel("Luminance Range (nits)")
    ax3.set_ylabel("Number of 10-bit Code Values")
    ax3.set_title("PQ: Where Do the 1024 Code Values Go?")
    ax3.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, pq_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("day19_pq_code_distribution.png", dpi=150)
    plt.show()
    print("\nSaved: day19_pq_code_distribution.png")


if __name__ == "__main__":
    main()