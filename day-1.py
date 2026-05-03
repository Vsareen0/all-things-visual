"""
Day 1 — The Sampling Theorem
All Things Visual Roadmap · Phase 1 · Signal & Frequency Theory

Exercise: See aliasing happen with your own eyes.
Run this script, study the plots, then try the bonus at the bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# --- Step 1: Generate the "ground truth" continuous signal ---
# A 5 Hz sine wave, sampled at 200 Hz so it looks smooth
freq = 5                       # signal frequency in Hz
duration = 1.0                 # 1 second
fs_high = 200                  # high sample rate (our "continuous" reference)
t_high = np.arange(0, duration, 1 / fs_high)
signal = np.sin(2 * np.pi * freq * t_high)

# --- Step 2: Downsample to several rates and reconstruct ---
sample_rates = [12, 10, 8, 6]  # Hz — try each one
nyquist_rate = 2 * freq        # 10 Hz for a 5 Hz signal

fig, axes = plt.subplots(len(sample_rates), 1, figsize=(10, 8), sharex=True)
fig.suptitle(f"Sampling a {freq} Hz sine — Nyquist rate = {nyquist_rate} Hz",
             fontsize=14, fontweight="bold")

for ax, fs in zip(axes, sample_rates):
    # Sample the signal at the lower rate
    n_samples = int(fs * duration)
    t_sampled = np.arange(n_samples) / fs
    sampled = np.sin(2 * np.pi * freq * t_sampled)

    # Reconstruct back to 200 Hz using scipy
    reconstructed = resample(sampled, len(t_high))

    # Plot everything
    ax.plot(t_high, signal, color="gray", alpha=0.4, linewidth=1, label="Original (200 Hz)")
    ax.stem(t_sampled, sampled, linefmt="C0-", markerfmt="C0o", basefmt=" ",
            label=f"Samples @ {fs} Hz")
    ax.plot(t_high, reconstructed, color="red", linewidth=1.2, linestyle="--",
            label="Reconstructed")

    status = "ALIASED" if fs < nyquist_rate else ("EDGE" if fs == nyquist_rate else "OK")
    ax.set_ylabel(f"{fs} Hz\n[{status}]", fontsize=11, rotation=0, labelpad=50, va="center")
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig("sampling_theorem_results.png", dpi=150)
plt.show()

# --- Step 3: Answer these questions ---
# Look at the plots:
#   • At which sample rate does the reconstructed wave first look wrong?
#     - 9 HZ
#   • Compare that to the Nyquist rate (2 × 5 = 10 Hz).
#    - 9Hz is less than Nyquist rate, so it aliases. 
#   • What happens at exactly 10 Hz? Is it perfect or slightly off?
#    - It matches exactly to Nquist rate and conditions says it should be more than Nyquist rate.

# --- Step 4 (Bonus): Two-frequency signal ---
# Uncomment below, run again, and figure out the new Nyquist rate.

# freq2 = 13  # Hz
# signal_combo = np.sin(2 * np.pi * freq * t_high) + np.sin(2 * np.pi * freq2 * t_high)
#
# # What's the highest frequency now? → 13 Hz
# # So Nyquist rate = 2 × 13 = 26 Hz
# # Try sampling at 30, 26, 20, and 14 Hz — which ones alias?
#
# bonus_rates = [30, 26, 20, 14]
# fig2, axes2 = plt.subplots(len(bonus_rates), 1, figsize=(10, 8), sharex=True)
# fig2.suptitle(f"Sampling a {freq}+{freq2} Hz signal — Nyquist = {2*freq2} Hz",
#               fontsize=14, fontweight="bold")
#
# for ax, fs in zip(axes2, bonus_rates):
#     n_samples = int(fs * duration)
#     t_sampled = np.arange(n_samples) / fs
#     sampled = np.sin(2*np.pi*freq*t_sampled) + np.sin(2*np.pi*freq2*t_sampled)
#     reconstructed = resample(sampled, len(t_high))
#
#     ax.plot(t_high, signal_combo, color="gray", alpha=0.4, linewidth=1)
#     ax.stem(t_sampled, sampled, linefmt="C0-", markerfmt="C0o", basefmt=" ")
#     ax.plot(t_high, reconstructed, color="red", linewidth=1.2, linestyle="--")
#
#     status = "ALIASED" if fs < 2*freq2 else "OK"
#     ax.set_ylabel(f"{fs} Hz\n[{status}]", fontsize=11, rotation=0, labelpad=50, va="center")
#     ax.set_ylim(-2.5, 2.5)
#     ax.grid(True, alpha=0.2)
#
# axes2[-1].set_xlabel("Time (seconds)")
# plt.tight_layout()
# plt.savefig("sampling_bonus_results.png", dpi=150)
# plt.show()