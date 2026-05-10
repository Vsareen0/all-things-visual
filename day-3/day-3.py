"""
Day 03 — Human Vision: Cone Spectral Sensitivity & Metamerism
=============================================================
All Things Visual Roadmap | Phase 1: Color Science

This exercise helps you:
1. Visualize the LMS cone fundamentals (how your three cone types respond to light)
2. Compute cone responses for arbitrary spectral power distributions
3. Find a metamer pair — two different spectra that look identical to your eyes

Requirements: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Part 1: Approximate CIE 2006 Cone Fundamentals (LMS) ---
# We'll use Gaussian approximations to the cone sensitivity curves.
# Real data can be downloaded from cvrl.org, but these capture the key shape.

wavelengths = np.arange(380, 781, 1)  # 380–780 nm in 1 nm steps


def gaussian(lam, peak, sigma, amplitude=1.0):
    """Simple Gaussian spectral sensitivity."""
    return amplitude * np.exp(-0.5 * ((lam - peak) / sigma) ** 2)


# Approximate cone fundamentals (normalized peak = 1)
# S-cones: peak ~420 nm, narrow
# M-cones: peak ~534 nm, broader
# L-cones: peak ~564 nm, broader, heavily overlapping with M
S_cone = gaussian(wavelengths, peak=420, sigma=26, amplitude=1.0)
M_cone = gaussian(wavelengths, peak=534, sigma=44, amplitude=1.0)
L_cone = gaussian(wavelengths, peak=564, sigma=46, amplitude=1.0)

# --- Plot the cone fundamentals ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wavelengths, L_cone, color="red", linewidth=2, label="L-cone (~564 nm)")
ax.plot(wavelengths, M_cone, color="green", linewidth=2, label="M-cone (~534 nm)")
ax.plot(wavelengths, S_cone, color="blue", linewidth=2, label="S-cone (~420 nm)")
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Relative Sensitivity", fontsize=12)
ax.set_title("Approximate Human Cone Spectral Sensitivities (LMS)", fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(380, 780)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day03_cone_fundamentals.png", dpi=150)
plt.show()
print("Saved: day03_cone_fundamentals.png")


# --- Part 2: Cone response to monochromatic light ---
# For a single wavelength λ, the cone responses are just the sensitivity values at λ.

def cone_response(spectrum):
    """Compute (L, M, S) cone responses for a given spectral power distribution."""
    L = np.sum(spectrum * L_cone)
    M = np.sum(spectrum * M_cone)
    S = np.sum(spectrum * S_cone)
    return L, M, S


# Sweep monochromatic light across the visible spectrum
L_responses = []
M_responses = []
S_responses = []

for i, lam in enumerate(wavelengths):
    # Monochromatic = delta function at this wavelength
    mono = np.zeros_like(wavelengths, dtype=float)
    mono[i] = 1.0
    L, M, S = cone_response(mono)
    L_responses.append(L)
    M_responses.append(M)
    S_responses.append(S)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wavelengths, L_responses, "r-", label="L response")
ax.plot(wavelengths, M_responses, "g-", label="M response")
ax.plot(wavelengths, S_responses, "b-", label="S response")
ax.set_xlabel("Monochromatic Wavelength (nm)")
ax.set_ylabel("Cone Response")
ax.set_title("Cone Responses to Monochromatic Light Across the Spectrum")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day03_monochromatic_responses.png", dpi=150)
plt.show()
print("Saved: day03_monochromatic_responses.png")


# --- Part 3: Finding a Metamer Pair ---
# Two different spectra that produce the SAME (L, M, S) triplet.

# Spectrum A: a monochromatic yellow at 580 nm
spectrum_a = np.zeros_like(wavelengths, dtype=float)
idx_580 = np.argmin(np.abs(wavelengths - 580))
spectrum_a[idx_580] = 1.0

L_a, M_a, S_a = cone_response(spectrum_a)
print(f"\nSpectrum A (monochromatic 580 nm):")
print(f"  L={L_a:.4f}, M={M_a:.4f}, S={S_a:.4f}")

# Spectrum B: a mix of red (620 nm) and green (540 nm)
# We need to find weights w_r and w_g such that:
#   w_r * L(620) + w_g * L(540) = L(580)
#   w_r * M(620) + w_g * M(540) = M(580)
# Two equations, two unknowns — solve with linear algebra!

idx_620 = np.argmin(np.abs(wavelengths - 620))
idx_540 = np.argmin(np.abs(wavelengths - 540))

# Build the 2x2 system using L and M cones
A_matrix = np.array([
    [L_cone[idx_620], L_cone[idx_540]],
    [M_cone[idx_620], M_cone[idx_540]]
])
b_vector = np.array([L_a, M_a])

weights = np.linalg.solve(A_matrix, b_vector)
w_red, w_green = weights

spectrum_b = np.zeros_like(wavelengths, dtype=float)
spectrum_b[idx_620] = w_red
spectrum_b[idx_540] = w_green

L_b, M_b, S_b = cone_response(spectrum_b)
print(f"\nSpectrum B (mix of {w_red:.3f} x 620nm + {w_green:.3f} x 540nm):")
print(f"  L={L_b:.4f}, M={M_b:.4f}, S={S_b:.4f}")

print(f"\nL match: {np.isclose(L_a, L_b)}")
print(f"M match: {np.isclose(M_a, M_b)}")
print(f"S match: {np.isclose(S_a, S_b)} (S-cones are nearly zero at these wavelengths)")

# Plot the two spectra side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.stem(wavelengths, spectrum_a, linefmt="orange", markerfmt="o", basefmt=" ")
ax1.set_title("Spectrum A: Monochromatic 580 nm")
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Power")
ax1.set_xlim(380, 780)

ax2.stem(wavelengths, spectrum_b, linefmt="purple", markerfmt="o", basefmt=" ")
ax2.set_title(f"Spectrum B: Mix of 620 nm + 540 nm")
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Power")
ax2.set_xlim(380, 780)

fig.suptitle("Metamer Pair: Different spectra, same perceived color!", fontsize=13)
plt.tight_layout()
plt.savefig("day03_metamer_pair.png", dpi=150)
plt.show()
print("Saved: day03_metamer_pair.png")


# --- Exercises for You ---
print("\n" + "=" * 60)
print("EXERCISES TO TRY:")
print("=" * 60)
print("""
1. OVERLAP QUESTION: At what wavelength do L and M cones have
   equal sensitivity? Print it out. Why does this matter for
   red-green color blindness?

2. S-CONE DROPOFF: At what wavelength does S-cone sensitivity
   drop below 1% of its peak? This explains why 4:2:0 chroma
   subsampling works — short-wavelength detail is invisible.

3. BUILD YOUR OWN METAMER: Can you find a 3-wavelength mix
   (e.g., 450nm + 540nm + 620nm) that matches a flat "white"
   spectrum? This is exactly what an RGB monitor does.

4. CHALLENGE: Download the real CIE 2006 2-degree cone
   fundamentals from https://cvrl.org and replace the Gaussian
   approximations. How different are the results?
""")