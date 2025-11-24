#!/usr/bin/env python
import numpy as np
import math
import sys
import argparse
import os
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a folder path.")
parser.add_argument("folder_path", type=str, help="Path to the folder containing the GLOW outputs")
parser.add_argument("--do_kin", nargs=1, type=str, help="optional: will do the kinetic calculation if the thermo.out of the reactant is provided.")
parser.add_argument("--temp", nargs=1, type=str, help="optional: For the kinetic analysis, temperature is needed")
parser.add_argument("--SOC", nargs=1, type=float, help="optional: SOC constant of the MECP")
parser.add_argument("--plot", action='store_true', help="optional: will plot k vs T.")

# Parse arguments
args = parser.parse_args()
folder_path = args.folder_path
do_kin = args.do_kin
temp = args.temp
SOC = args.SOC
kplot = args.plot

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
kcaltoJ = 4184                 # kcal to J
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
    Eavg = 0
    with open(thermo_file) as f:
        l = f.readlines()
    lf = len(l)
    if kplot == True:
        Q_temps = []
        Tps = []
    for i in range(len(l)):
        if "Qtot" in l[i]:
            if kplot == False:
                for j in range(i, len(l)):
                    if temp[0] in l[j]:
                        Q = float(l[j].split()[8])
                    if '-----------------' in l[j]:
                        break
            elif kplot == True:
                for j in range(i, len(l)):
                    try: (l[j].split()[0])
                    except: break
                    if (l[j].split()[0]) == 'T':
                        continue
                    Tps.append(float(l[j].split()[0]))
                    Q_temps.append(float(l[j].split()[8]))
                    if '-----------------' in l[j]:
                        break

        if "Electronic energy:" in l[i]:
            Elect = float(l[i].split()[2])    
        if "Eavg" in l[i]:
            for j in range(i, len(l)):
                if temp[0] in l[j]:
                    Eavg = float(l[j].split()[5])

    if Eavg== 0:
        Eavg = 0.0
    if kplot == False:
        return Q, Elect, Eavg
    elif kplot == True:
        return Q_temps, Elect, Eavg, Tps

# Functions to make the integral
def LZ_prob(E, H12, dF, mu):
    """
    Calculate the Landau-Zener probability.
    E: Energy 
    H12: Coupling term
    dF: Gradient difference
    mu: Reduced mass
    """
    if E <= 0:
        return 0.0
    gamma = (2 * np.pi * H12**2) / (hbar_j * dF * np.sqrt(2 * E / mu))   
    return np.exp(-gamma)

def max_boltz_distr(E, T):
    """
    Calculate the maximum Boltzmann distribution.
    E: Energy 
    T: Temperature
    """
    if E <= 0:
        return 0.0
    return np.exp(-E / (kb_j * T)) * (1 / (kb_j * T))

def integrand_single(E, H12, dF, mu, T):
    """
    Integrand for the Landau-Zener integral.
    E: Energy
    H12: Coupling term
    dF: Gradient difference
    mu: Reduced mass
    T: Temperature
    """
    return (1-LZ_prob(E, H12, dF, mu)) * max_boltz_distr(E, T)

def integrand_double(E, H12, dF, mu, T):
    """
    Integrand for the Landau-Zener integral.
    E: Energy
    H12: Coupling term
    dF: Gradient difference
    mu: Reduced mass
    T: Temperature
    """
    return (1-LZ_prob(E, H12, dF, mu)**2) * max_boltz_distr(E, T)

R = 8.31446261815324
if do_kin:
    if kplot == False:
        QMECP, ElenMECP, EavgMECP = read_thermo(str(folder_path) + 'thermo.out')
        QR, ElenR, dummi = read_thermo(str(do_kin[0]))
    elif kplot == True:
        QMECP_list, ElenMECP, EavgMECP, Tps = read_thermo(str(folder_path) + 'thermo.out')
        print(f"QMECP_list: {QMECP_list}")
        QR_list, ElenR, dummi, dummi2 = read_thermo(str(do_kin[0]))
        print(f"QR_list: {QR_list}")
    
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
    if kplot == True:
        QMECP = QMECP_list[-1]
        QR = QR_list[-1]
        print(f"Temperature    : {Tps[-1]:.2f} K")

    print(f"QMECP          : {QMECP:.6e}")
    print(f"QREACT         : {QR:.6e}")
    

    lower_limit = 0.0
    upper_limit = 40 * kb_j * T  
    result, error = quad(integrand_single, lower_limit, upper_limit, args=(SOC_J, grad1_Jm, mu_kg, T))
    
    print(f"Thermally averaged probability of single cross: {result}")
    print(f"Estimated error: {error}")

    result2, error2 = quad(integrand_double, lower_limit, upper_limit, args=(SOC_J, grad1_Jm, mu_kg, T))

    print(f"Thermally averaged probability of double cross: {result2}")
    print(f"Estimated error: {error}")

    k_TS = (QMECP / QR)  * np.exp(-dE_Jmol / (R * T)) * (kb_j* T / h)
    k_LZ = k_TS * result
    k_LZ2 = k_TS * result2
    # -------------------------------
    # Output Results
    # -------------------------------

    print(f"k_TS = {k_TS:.4e}  [1/s]")
    print(f"k_LZ (single) = {k_LZ:.4e}  [1/s]")
    print(f"k_LZ (double) = {k_LZ2:.4e}  [1/s]")
    # Calculating the pre-exponential factor
    Arr = (QMECP / QR)* result * (kb_j* T / h)
    Arr2 = (QMECP / QR)* result2 * (kb_j* T / h)
    print(f"Preexponential factor single (A)     : {Arr:.6e}")
    print(f"Preexponential factor double (A)     : {Arr2:.6e}")
    print(f"Partition Ratio: {QMECP / QR:.6e}")

    # Different temperature plot
    if kplot == True:
        k_TS_list = []
        k_LZ2_list = []
        for t in Tps:
            T = t
            lower_limit = 0.0
            upper_limit = 40 * kb_j * T  
            result2, error2 = quad(integrand_double, lower_limit, upper_limit, args=(SOC_J, grad1_Jm, mu_kg, T))
            k_TS = (QMECP / QR)  * np.exp(-dE_Jmol / (R * T)) * (kb_j* T / h)
            k_LZ2 = k_TS * result2
            k_TS_list.append(k_TS)
            k_LZ2_list.append(k_LZ2)
        kTS_array = np.array(k_TS_list)
        kLZ_2_array = np.array(k_LZ2_list)
        logkTS = np.log10(kTS_array)
        logkLZ2 = np.log10(kLZ_2_array)
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame({
            'Temperature': Tps,
            'logkTS': logkTS,
            'logkLZ2': logkLZ2
        })
        df.to_csv('rates.csv', index=False)
        plt.plot(Tps, logkTS, label='k_TS')
        plt.plot(Tps, logkLZ2, label='k_LZ (double)')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Rate constant (1/s)')
        plt.title('Rate Constants vs Temperature')
        plt.legend()
        plt.grid()
        plt.show()

