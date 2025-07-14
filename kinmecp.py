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


# CONSTANTS
c = 299792458
pi = 3.1415927
kb = 3.167E-6 # He/K
kbj = 1.380649E-23 # J/K
e = 2.718281828
h = 4.1357E-15 # eV.s
hj = 6.62607015E-34 # J.s

#CONVERSIONS
evtoHe = 0.0367493
cmtoJ = 1.98645E-23
cmtoJ_mol = 0.01196265919209E-3
HetoJ = 4.35974E10-18
HetoJ_mol = 2625.5002E3
uamtokg = 1.660540199E-27
Bhtom = 5.29177e-11
gradstoSI = HetoJ_mol / Bhtom
NA = 6.023E23
evtoJ = 96.4853075E3
hev = 4.1357E-15
kbev = 8.6173E-16
hbarj = 1.054571817E-34 #J*s
hbarev = 6.5821E-16
cmtoeV = 0.00012398425731484
kcaltoJ = 4184
kcaltoev = 0.04336411530877

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
    print('QMECP', QMECP)
    QR, ElenR = read_thermo(str(do_kin[0]))
    print('QREACT', QR)
    e0 = (a.dF  ) / ( 2 * SOC[0] * cmtoJ *  a.F )
    print('e0', e0)
    print('redmass', a.redmass)
    print('F', a.F)
    print('dF', a.dF)
    print('SOC', SOC[0])
    #s = (((4 * ((SOC[0]*NA* cmtoJ) ** (1.5)))/((hj)/(2*pi))) * ((a.redmass/(1000*NA))/(a.F * gradstoSI * a.dF * gradstoSI))**0.5)
    b1 = (4 * (SOC[0] * cmtoJ)**(3/2))/((hbarj))
    print('b1', b1)
    b2 = ((a.redmass*uamtokg)/(a.F* gradstoSI * a.dF * gradstoSI))**(1/2)
    print('b2', b2)
    s = b1 * b2
    print('s', s)
    #k_LZ = (kb * float(temp[0]))**(0.5)  *  (QMECP/(QR*h*evtoHe))  *  ( (pi**(1.5)*s) / (2 * (e0**(0.5)) ) )  *  e**(-((0.1282) * evtoHe)/(kb * float(temp[0])))
    #P_LZ = ( ((pi**(3/2))*s) / (2 * ((e0/(R * float(temp[0])))**(0.5)) ) ) * e**(-((0.2193) * evtoJ)/(R * float(temp[0])))
    u = pi**(3/2) * s
    d = 2 * (e0/(R*float(temp[0])))**(1/2)
    P_LZ =  (u/d) * (1/hj)
    diffElen = (ElenMECP - ElenR) * kcaltoev
    print('diff_elect_energy', diffElen)
    print('Prob', P_LZ)
    print('exp * Q', (e**(-(((diffElen)) * evtoJ)/(R * float(temp[0]))))* (QMECP/QR))
    k_LZ =   (QMECP/QR) * P_LZ * e**(-((diffElen) * evtoJ)/(R * float(temp[0])))
    print('rate', k_LZ)
    k_TST = (QMECP/QR)* e**(-((diffElen) * evtoJ)/(R * float(temp[0]))) * ((float(temp[0])*kbj)/hj)
    print('rate_TST', k_TST)
    #print('exp', e**(-((0.1282) * evtoHe)/(kb * float(temp[0]))))
    #print('s/e0', ( (pi**(1.5)*s) / (2 * (e0**(0.5)) ) ))
    print('factorQ', (QMECP/(QR)))

