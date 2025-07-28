import numpy as np

# -------------------------------
# Physical Constants
# -------------------------------
pi = np.pi
hbar_j = 1.054571817e-34        # Reduced Planck's constant (J·s)
kb_j = 1.380649e-23             # Boltzmann constant (J/K)
u_to_kg = 1.66053906660e-27     # Atomic mass unit to kg
cm_to_J = 1.98644586e-23        # Conversion: 1 cm⁻¹ to J
NA = 6.02214076e23              # Avogadro's number
HetoJmol = 2625.5002e3          # 1 Hartree to J/mol
Bhtom = 5.29177e-11             # Bohr to meter
gradstoSI = HetoJmol / Bhtom   # Gradient conversion to J/mol/m

# -------------------------------
# Input Parameters (example values)
# -------------------------------
SOC_cm = 91.95                # Spin-orbit coupling (cm⁻¹)
grad1 = 0.094447             # dF: gradient difference at crossing (Hartree/Bohr)
grad2 = 0.074942                # F: average gradient (Hartree/Bohr)
red_mass_u = 18.576100          # Reduced mass (u)
T = 673.15                     # Temperature in K

# -------------------------------
# Convert all quantities to SI units
# -------------------------------
SOC_J = SOC_cm * cm_to_J
grad1_Jm = grad1 * gradstoSI / NA     # J/m
grad2_Jm = grad2 * gradstoSI / NA     # J/m
mu_kg = red_mass_u * u_to_kg

# -------------------------------
# Compute Taylor Expansion Term (Lambda)
# Formula: Λ = (2π * V^2 / (ħ * |ΔF|)) * sqrt(μ / (kB * T))
# Formula: b1 = (4 * (SOC)**(3/2)) / ħ
# Formula: b2 = sqrt(redmass / (grad2 * grad1))
# Formula: u = pi^(3/2) * b1 * b2
# Formula: e0 = grad1 / (2 * SOC  * grad2)
# Formula: d = 2 * (e0 / (R * T))^0.5
# Formula: P_LZ = (u/d)   * (1/h)
# Formula: kLZ = (QMECP / QR) * P_LZ * e^(dE /(RT))
# -------------------------------
numerator = 2 * pi * SOC_J**2
denominator = hbar_j * grad1_Jm
sqrt_term = np.sqrt(mu_kg / (kb_j * T))

Lambda = (numerator / denominator) * sqrt_term

# -------------------------------
# Output
# -------------------------------
print(f"Taylor expansion term (Lambda) = {Lambda:.4e}")
if Lambda < 0.1:
    print("→ Weak coupling regime: Taylor expansion is valid.")
elif Lambda < 1:
    print("→ Moderate coupling: Taylor expansion may be marginal.")
else:
    print("→ Strong coupling: Taylor expansion is NOT valid; use full P_LZ = 1 - exp(-Lambda).")
