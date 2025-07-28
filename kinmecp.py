#!/usr/bin/env python
import numpy as np
import math
import sys
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a folder path.")
parser.add_argument("folder_path", type=str, help="Path to the folder containing the GLOW outputs")
parser.add_argument("--do_kin", nargs=1, type=str, help="optional: will do the kinetic calculation if the thermo.out of the reactant is provided.")
parser.add_argument("--temp", nargs=1, type=str, help="optional: For the kinetic analysis, temperature is needed")
parser.add_argument("--SOC", nargs=1, type=float, help="optional: SOC constant of the MECP")

# Parse arguments
args = parser.parse_args()
folder_path = args.folder_path
do_kin = args.do_kin
temp = args.temp
SOC = args.SOC

# Check if the folder exists
if not os.path.isdir(folder_path):
    print(f"Error: The specified folder '{folder_path}' does not exist.")
    exit(1)


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
kcaltoev = 0.0433641           # kcal to eV

class MECP:
    def __init__(self):
        # THE FILES WHICH LOAD MECP PROPS:
        self.MECPinfo = ""
        self.oszicar = ""
        self.mecpout = ""
        self.outcar = ""
        # SPIN-ORBIT COUPLING:
        self.hso = 0
        # READ BY MECP.info.molden:
        self.F = 0
        self.dF = 0
        self.redmass = 0
        self.vibs = []

    def load_MECPinfo(self, MECPprop):
        ''' LOADS THE MECP.info FILE AND READS DATA '''
        with open(MECPprop) as f:
            self.l = f.readlines()
        self.lf = len(self.l)
        for i in range(len(self.l)):
            self.l[i] = self.l[i].split()
            if 'Reduced' in self.l[i]:
                self.redmass = float(self.l[i+1].split()[0])
                print(self.redmass)
            #if 'molecules' in self.l[i]:
            #   self.natoms = int(self.l[i][4])
            #   print(self.natoms)  
            if 'geometric' in self.l[i]:
                self.F = float(self.l[i+1].split()[0])
            if 'norm' in self.l[i]:
                self.dF = float(self.l[i+1].split()[0])
        
    def load_oszicar(self, oszicar):
        ''' LOADS THE OSZICAR FILE AND RETURNS THE ENERGY '''
        with open(oszicar) as f:
            self.l = f.readlines()
        self.lf = len(self.l)
        energy = float(self.l[self.lf-1].split()[4]) # in eV
        print(energy)
        return energy

    def load_outcar(self, outcar):
        ''' LOADS THE OUTCAR.freq FILE AND RETURNS THE ENERGY '''
        with open(outcar) as f:
            self.l = f.readlines()
        self.lf = len(self.l)
        for i in range(len(self.l)):
            if 'entropy=' in self.l[i]:
                energy = float(self.l[i].split()[3])
                print(energy)
        return energy

    def load_mecpout(self, mecpout):
        ''' LOADS THE MECPOUT FILE GENERATED WITH ML-MECP '''
        with open(mecpout) as f:
            self.l = f.readlines()
        self.lf = len(self.l)
        last_step_line = None
        for line in self.l:
            if 'Step' in line:
                last_step_line = line
        if last_step_line:
            energy = float(last_step_line.split()[4])  # 5th column (index 4)
            print(energy)
            return energy
        else:
            print("No 'Step' line found.")
            return

a = MECP()
print(folder_path)
a.MECPinfo = folder_path + 'MECP.info.molden'
a.oszicar = folder_path + 'OSZICAR.opt'
a.outcar = folder_path + 'OUTCAR.freq'
a.mecpout = folder_path + 'mecp.out'
a.load_MECPinfo(a.MECPinfo)

try:
    en = a.load_outcar(a.outcar)
except:
    print('No OSZICAR file found, using energy from MECPprop')
    en = a.load_mecpout(a.mecpout)



# CALCULATE RATE CONSTANTS:

def read_thermo(thermo_file):
    with open(thermo_file) as f:
        l = f.readlines()
    lf = len(l)
    for i in range(len(l)):
        if "Qtot" in l[i]:
            for j in range(i, len(l)):
                if temp[0] in l[j]:
                    Q = float(l[j].split()[8])
                if '-----------------' in l[j]:
                    break
        if "Electronic energy:" in l[i]:
            Elect = float(l[i].split()[2])    
    return Q, Elect

R = 8.31446261815324
if do_kin:
    QMECP, ElenMECP = read_thermo(str(folder_path) + 'thermo.out')
    QR, ElenR = read_thermo(str(do_kin[0]))
    
    T = float(temp[0])
    SOC_cm = SOC[0]
    grad1 = a.dF
    grad2 = a.F
    red_mass_u = a.redmass
    dE_eV =  (ElenMECP - ElenR) * kcaltoev
    # -------------------------------
    # Convert to SI Units
    # -------------------------------
    SOC_J = SOC_cm * cm_to_J
    grad1_Jm = grad1 * gradstoSI / NA
    grad2_Jm = grad2 * gradstoSI / NA
    mu_kg = red_mass_u * u_to_kg
    dE_Jmol = dE_eV * evtoJ * NA

    # Partition functions
    print(f"QMECP          : {QMECP:.6e}")
    print(f"QREACT         : {QR:.6e}")

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


    Arr = (QMECP / QR) * P_LZ
    print(f"Preexponential factor (A)     : {Arr:.6e}")

    print(f"Partition Ratio: {QMECP / QR:.6e}")

