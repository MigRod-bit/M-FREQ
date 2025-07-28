import numpy as np

# -------------------------------
# Physical Constants
# -------------------------------
pi = np.pi
h = 6.62607015e-34             # Planck's constant (J·s)
hbar_j = 1.054571817e-34       # Reduced Planck constant (J·s)
kb_j = 1.380649e-23            # Boltzmann constant (J/K)
R = 8.314462618                # Gas constant (J/mol·K)
u_to_kg = 1.66053906660e-27    # Atomic mass unit to kg
cm_to_J = 1.98644586e-23       # Conversion: 1 cm⁻¹ to J
NA = 6.02214076e23             # Avogadro's number
HetoJmol = 2625.5002e3         # 1 Hartree to J/mol
Bhtom = 5.29177e-11            # Bohr to meter
gradstoSI = HetoJmol / Bhtom   # Gradient conversion to J/mol/m
evtoJ = 1.602176634e-19        # eV to J

# -------------------------------
# Input Parameters (example values)
# -------------------------------
SOC_cm = 91.95                 # Spin-orbit coupling (cm⁻¹)
grad1 = 0.094447               # ΔF: gradient difference (Hartree/Bohr)
grad2 = 0.074942               # F: average gradient (Hartree/Bohr)
red_mass_u = 18.576100         # Reduced mass (u)
T = 673.15                     # Temperature (K)
dE_eV = 0.12                   # Electronic energy gap ΔE (eV)
QMECP = 1.0                    # Partition function at crossing
QR = 1.0                       # Partition function at reactant

# -------------------------------
# Convert to SI Units
# -------------------------------
SOC_J = SOC_cm * cm_to_J
grad1_Jm = grad1 * gradstoSI / NA
grad2_Jm = grad2 * gradstoSI / NA
mu_kg = red_mass_u * u_to_kg
dE_Jmol = dE_eV * evtoJ * NA

# -------------------------------
# Taylor Expansion Term (Lambda)
# -------------------------------
numerator = 2 * pi * SOC_J**2
denominator = hbar_j * grad1_Jm
sqrt_term = np.sqrt(mu_kg / (kb_j * T))
Lambda = (numerator / denominator) * sqrt_term

# -------------------------------
# Extended Landau-Zener Terms
# -------------------------------
# b1 = (4 * V**(3/2)) / ħ
b1 = (4 * SOC_J**1.5) / hbar_j

# b2 = sqrt(mu / (F * ΔF))
b2 = np.sqrt(mu_kg / (grad2_Jm * grad1_Jm))

# u = π^(3/2) * b1 * b2
u = (pi**1.5) * b1 * b2

# e0 = ΔF / (2 * V * F)
e0 = grad1_Jm / (2 * SOC_J * grad2_Jm)

# d = 2 * sqrt(e0 / (R * T))
d = 2 * np.sqrt(e0 / (R * T))

# P_LZ = (u / d) * (1 / h)
P_LZ = (u / d) * (1 / h)

# kLZ = (QMECP / QR) * P_LZ * exp(-ΔE / RT)
k_LZ = (QMECP / QR) * Lambda * np.exp(-dE_Jmol / (R * T)) * (kb_j* T / h)

# kLZ = (QMECP / QR) * P_LZ * exp(-ΔE / RT)
k_TS = (QMECP / QR)  * np.exp(-dE_Jmol / (R * T)) * (kb_j* T / h)

# -------------------------------
# Output Results
# -------------------------------
print(f"Taylor expansion term (Lambda) = {Lambda:.4e}")
print(f"b1 = {b1:.4e}  [1/s^1.5]")
print(f"b2 = {b2:.4e}  [kg^0.5·m/J]")
print(f"u = {u:.4e}    [1/s]")
print(f"e0 = {e0:.4e}  [J/mol]")
print(f"d = {d:.4e}    [unitless]")
print(f"P_LZ = {P_LZ:.4e}  [1/s]")
print(f"k_LZ = {k_LZ:.4e}  [1/s]")
print(f"k_TS = {k_TS:.4e}  [1/s]")

if Lambda < 0.1:
    print("→ Weak coupling regime: Taylor expansion is valid.")
elif Lambda < 1:
    print("→ Moderate coupling: Taylor expansion may be marginal.")
else:
    print("→ Strong coupling: Use full expression for P_LZ.")
