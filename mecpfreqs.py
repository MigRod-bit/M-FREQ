import numpy as np
from scipy.linalg import eigh

# Conversion Constants
hartree_to_joule = 4.359744e-18 # Hartree to Joule
bohr_to_meter = 5.291772e-11 # Bohr radius to meters
amu_to_kg = 1.66053906660e-27 # Atomic mass unit to kg
speed_of_light = 2.99792458e10  # cm/s
Angstrom_to_Bohr = 1.8897259886  # Conversion factor from Angstrom to Bohr
eV_to_J = 1.602176634e-19  # Conversion factor from eV to J
eVtoHe = 0.0367493  # Conversion factor from eV to Hartree
c = 299792458  # Speed of light in m/s
pi = np.pi # Pi



# ================== FUNCTIONS ==================

import numpy as np

def normalize(v):
    """Function to normalize a vector, handles near-zero norm."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else np.zeros_like(v)



def construct_projection_matrix_orthogonalized(coords, masses, grad_diff, grad_avg):
    """
    Constructs the projection matrix by sequentially orthogonalizing 
    grad_diff (g) and grad_avg (h) against the T+R space and each other.
    This prevents over-projection.
    """
    natoms = len(masses)
    dim = 3 * natoms
    basis_vectors = []
    
    # --- 1. Define T and R Vectors ---         ONLY REMOVED IN GAS PHASE, KEEP TRANSL AND ROT FOR SOLIDS
    # Translation vectors (T)
   #  for i in range(3):
   #      vec = np.zeros(dim)
   #      for j in range(natoms):
   #          vec[3*j + i] = np.sqrt(masses[j])
   #      basis_vectors.append(normalize(vec))

   #  # Rotation vectors (R)
   #  center = np.average(coords, axis=0, weights=masses)
   #  coords_centered = coords - center
   #  for i in range(3):
   #      axis = np.zeros(3); axis[i] = 1.0
   #      vec = np.cross(coords_centered, axis)
   #      flat = (vec * np.sqrt(masses)[:, np.newaxis]).flatten()
   #      if np.linalg.norm(flat) > 1e-8:
   #          # We enforce orthogonality here by running Gram-Schmidt against T
   #          B_TR = np.array(basis_vectors).T
   #          # P_TR_out is the projector orthogonal to T (for the current rotation vector)
   #          P_TR_out = np.eye(dim) - B_TR @ B_TR.T
   #          rot_perp = P_TR_out @ flat
   #          basis_vectors.append(normalize(rot_perp))
            
    # --- 2. Orthogonalize Gradient Difference (g) against T+R ---
    # Current basis: B_TR = {T, R}. We define the projector P_TR_out
    B_TR = np.array(basis_vectors).T
    P_TR_out = np.eye(dim) - B_TR @ B_TR.T
    
    g_raw = grad_diff.flatten()
    # g_perp is the component of g ORTHOGONAL to T+R
    g_perp = P_TR_out @ g_raw
    
    # Add g_perp to the basis
    if np.linalg.norm(g_perp) > 1e-10:
        basis_vectors.append(normalize(g_perp))
    # Note: If g_perp is zero, g is already fully contained in the T/R subspace.

    # --- 3. Orthogonalize Gradient Average (h) against T+R and g_perp ---
    # Current basis: B_TRG = {T, R, g_perp}. We define the projector P_TRG_out
    B_TRG = np.array(basis_vectors).T
    P_TRG_out = np.eye(dim) - B_TRG @ B_TRG.T
    
    h_raw = grad_avg.flatten()
    # h_perp is the component of h ORTHOGONAL to T+R AND g_perp
    h_perp = P_TRG_out @ h_raw
    
    # Add h_perp to the basis
    if np.linalg.norm(h_perp) > 1e-10:
        basis_vectors.append(normalize(h_perp))
    # Note: If h_perp is zero, h is fully contained in the T/R/g subspace.

    # --- 4. Final Projection Operator (P) ---
    # B_final is the complete, numerically orthonormal basis of all vectors to be removed
    B_final = np.array(basis_vectors).T 
    
    # P removes all directions in B_final
    P = np.eye(dim) - B_final @ B_final.T 

    return P

def hessian_to_frequencies(e_v):
    """Convert Hessian eigenvalues to frequencies in cm^-1"""
    factor = np.sqrt(hartree_to_joule / (amu_to_kg * bohr_to_meter**2)) / (2 * np.pi * speed_of_light) # Conversion factor from Hartree/Bohr^2 to cm^-1
    frequencies = []
    for val in e_v:
        if val < 0:
            freq = -np.sqrt(-val) * factor 
        else:
            freq = np.sqrt(val) * factor
        frequencies.append(freq)

    return np.array(frequencies)

def read_vasprun(filename):
    """Read VASP vasprun.xml file and extract Hessian, gradient, and electronic energy."""
    global n_F_atoms
    tree = ET.parse(filename)
    root = tree.getroot()
    # First of all, get the id of non-frozen atoms
    for varray in root.iter('varray'):
        if varray.attrib.get('name') == 'selective' and varray.attrib.get('type') == 'logical':
            n_F_atoms = []
            counter = 0
            for v in varray.findall('v'):
                row = list(map(str, v.text.strip().split()))
                if row[0] == 'T':
                    n_F_atoms.append(counter) # Append the index of the non-frozen atoms
                counter += 1

    # Get the Hessian and gradient data
    hessian_data = None
    gradient_data = None
    # Get the electronic energy
    energy = None
    # Find the varray with name="hessian" in vasprun.xml
    # and the one with name="forces" for the gradient
    for varray in root.iter('varray'):
        if varray.attrib.get('name') == 'hessian':
            hessian_data = []
            for v in varray.findall('v'):
                # Convert string of numbers to a list of floats
                row = list(map(float, v.text.strip().split()))
                hessian_data.append(row)
        elif varray.attrib.get('name') == 'forces':
            # Always update gradient_data to keep last occurrence
            gradient_data = []
            for v in varray.findall('v'):
                row = list(map(float, v.text.strip().split()))
                gradient_data.append(row)
            gradient_data = np.array(gradient_data)
    for e in root.iter('i'):
        if e.attrib.get('name') == "e_fr_energy":
            # Get the electronic energy
            try:
                energy = float(e.text.strip()) # Get the last occurrence of the electronic energy
            except ValueError:
                continue
    n_F_grad = gradient_data[n_F_atoms] # Only keep the gradient of non-frozen atoms

    return np.array(hessian_data), n_F_grad, energy

def read_poscar(filename):
    """Read POSCAR file and extract coordinates and elements of non-frozen atoms."""
    with open(filename) as f:
        lines = f.readlines()
        
    # Skip the first two lines (comment and scaling factor)
    coords = []
    for line in lines[2:]:   # First get total atoms
        if "Direct" in line:
            print("Expected Cartesian coordinates, but found Direct.") # Need to implement coordinates conversion
        elif "Cartesian" in line:
            print("Found Cartesian coordinates.")
            tot_atoms = 0
            tot_atoms_list = lines[6].replace("     ", ",").split()
            tot_atoms_list = list(map(int, tot_atoms_list[0].split(',')))
            tot_atoms = sum([int(i) for i in tot_atoms_list])
            Element_list = ",".join(lines[5].split()).split(',')
            el_list_long = []
            for ind, i in enumerate(tot_atoms_list):
                for _ in range(i):
                    el_list_long.append(Element_list[ind])
            n_F_elements = [el_list_long[i] for i in n_F_atoms]

            # Now get the coodinates of non-Frozen atoms:
            for lin in lines[8:8+tot_atoms]:
                coords.append(list(map(float, lin.split())))
            coords = np.array(coords)
            n_F_coords = coords[n_F_atoms]

    return  n_F_coords, n_F_elements

def gen_fake_OUTCAR(freqs, RC_id, energy):
    """Generate a fake OUTCAR file with frequencies and energy."""
    with open('OUTCAR.fake', 'w') as f:
        f.write(f'energy  without entropy=     {energy:.6f}  energy(sigma->0) =   {energy:.6f}\n')
        for i, freq in enumerate(freqs):
            if freq < 0:
                freq_label = f"{freq:.3f}i"
            else:
                freq_label = f"{freq:.3f}"

            THz = (freq * c * 1e-12 * 100)
            two_pi_THz = THz * 2 * pi
            mode_number = i + 1
            meV = 0.00012398426 * freq * 1000

            if mode_number == RC_id and doRC:
                tag = " <-- Reaction Coordinate"
            else:
                tag = ""

            f.write(f"{mode_number:3d} f  = {THz:10.6f} THz  {two_pi_THz:10.6f}  2PiTHz  {freq_label:>10} cm-1   {meV:.6f} meV{tag}\n")

    



def write_molden(filename, atom_labels, atom_coords, frequencies, normal_modes, norm_diff_grad, geom_mean_grads, red_mass):
    """Write vibrational data and MECP properties in Molden format."""
    n_atoms = len(atom_labels)
    with open(filename, "w") as f:
        f.write("[Molden Format]\n")
        f.write(" Frequencies: (cm-1)\n")
        for freq in frequencies:
            f.write(f"    {freq:12.7f}\n")
        f.write(f"The Reduced Mass orthogonal to the seam of crossing is (a.m.u.): \n {red_mass: .4f}  \n")
        f.write(f"The geometric mean of the norms of the two gradients is: \n {geom_mean_grads: .8f} \n")
        f.write(f"The norm of the difference gradient is: \n {norm_diff_grad: .8f}  \n")
        f.write("\n [Atoms] Angs\n")
        for i, (label, coord) in enumerate(zip(atom_labels, atom_coords), 1):
            f.write(f"{label:2s} {i:3d}  {a_number[label]}  {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}\n")
        f.write(" [FREQ]\n")
        for freq in np.flip(frequencies):
            f.write(f"{freq:20.2f}\n")
        f.write(" [FR-COORD]\n")
        for i, (label, coord) in enumerate(zip(atom_labels, atom_coords), 1):
            f.write(f"{label:2s}   {2*coord[0]: .10f} {2*coord[1]: .10f} {2*coord[2]: .10f}\n")
        f.write(" [FR-NORM-COORD]\n")
        for mode_idx, mode in enumerate(normal_modes, 1):
            f.write(f"Vibration {mode_idx:>30d}\n")
            for i in range(n_atoms):
                x, y, z = mode[3*i:3*i+3]
                f.write(f"{x:20.8f}{y:20.8f}{z:20.8f}\n")


# ================== INPUT SECTION ==================
# Set up argument parser
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description="Process a folder path.")
parser.add_argument("vasp_run_list", nargs=2, help="vasprun from singlet and triplet states")
parser.add_argument("poscar", nargs=1, help="POSCAR file with the coordinates of the MECP")
parser.add_argument("--RC", action="store_true", help="generate OUTCAR.rc with the reaction coordinate mode")

args = parser.parse_args()
vasprun_S = args.vasp_run_list[0]
vasprun_T = args.vasp_run_list[1]
poscar = args.poscar[0]
doRC = args.RC

# Read Hessian and gradient from VASP vasprun.xml files
H1, G1, energy1 = read_vasprun(vasprun_S)
H2, G2, energy2 = read_vasprun(vasprun_T)
# Average the electronic energy
energy = (energy1 + energy2) / 2
# Unit conversions
H1*= -eVtoHe  / (Angstrom_to_Bohr**2) # VASP Hessian is opposite signed
H2*= -eVtoHe / (Angstrom_to_Bohr**2)  # Convert to Hartree/Bohr^2
G1 = G1 * eVtoHe / Angstrom_to_Bohr  # Convert to Hartree/Bohr
G2 = G2 * eVtoHe / Angstrom_to_Bohr  
# Read POSCAR file to get coordinates and elements of non-frozen atoms
MECP_coords, n_F_elem = read_poscar(poscar)
# Atomic masses
a_mass = {'Al': 26.981538, 'Si': 28.085, 'O': 15.999, 'Cu': 63.55, 'N': 14.007, 
          'H': 1.00784, 'C': 12.011, 'F': 18.998403, 'P': 30.973762, 'S': 32.06,
          'Fe': 55.845, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90447, 'B': 10.81,
          'Li': 6.94, 'Na': 22.989769, 'K': 39.0983, 'Ca': 40.078, 'Mg': 24.305,
           'Zn': 65.38, 'Mn': 54.938044, 'Cr': 51.9961, 'Ni': 58.6934, 'Co': 58.933194}
# Atomic numbers
a_number = {'Al': 13, 'Si': 14, 'O': 8, 'Cu': 29, 'N': 7,
             'H': 1, 'C': 6, 'F': 9, 'P': 15, 'S': 16, 
             'Fe': 26, 'Cl': 17, 'Br': 35, 'I': 53, 'B': 5,
             'Li': 3, 'Na': 11, 'K': 19, 'Ca': 20, 'Mg': 12,
            'Zn': 30, 'Mn': 25, 'Cr': 24, 'Ni': 28, 'Co': 27}

MECP_coords *= 1.8897259886  # Convert to Bohr

natoms = len(n_F_atoms)  # Number of non-frozen atoms
dim = 3 * natoms # Dimension of the system (3N)

# Ensure Hessians and gradients are symmetric
H1 = (H1 + H1.T) / 2
H2 = (H2 + H2.T) / 2

# ================== PROCESSING ==================

# Average quantities
H_avg = 0.5 * (H1 + H2) # Average Hessian
grad_avg = 0.5 * (G1 + G2) # Average gradient
grad_diff = G1 - G2 # Difference gradient points to the direction of greater energy difference between both gradients which is normally the reaction coordinate direction.
# Projecting this gradient will remove both directions so it is not important which one is subtracted from the other (G2-G1 or G1-G2).

# Norm of the difference gradient:
grad_diff_norm = np.linalg.norm(grad_diff)

# Geometric mean of the norm of both gradients
grad_avg_norm = np.linalg.norm(grad_avg)

# Calculate the masses of non-frozen atoms
masses = np.array([a_mass[elem] for elem in n_F_elem])

# Mass-weight Hessian and gradients
masses3N = np.repeat(masses, 3)

# Mass-weighted Hessian and gradients:
#H_mw = mass_weight_hessian(H_avg, masses) # Use if hessian is not already mass-weighted
H_mw = H_avg # Use if hessian is already mass-weighted

grad_avg_mw = grad_avg.flatten() / np.sqrt(masses3N) # Mass-weighted gradients
grad_diff_mw = grad_diff.flatten() / np.sqrt(masses3N)



# Construct projection matrix
P = construct_projection_matrix_orthogonalized(MECP_coords, masses, grad_diff_mw, grad_avg_mw)

# Project Hessian to remove translation, rotation, and grad_diff directions
H_proj = P.T @ H_mw @ P

# Diagonalize to find normal modes
# The eigenvalues are the squared frequencies, and the eigenvectors are the normal modes
e_vals, e_vecs = eigh(H_proj)

# Frequencies
frequencies = np.flip(hessian_to_frequencies(e_vals))

# Print vibrational modes
print("Vibrational Frequencies at MECP (cm⁻¹):")
for i, freq in enumerate(frequencies):
        print(f"Mode {i+1:2d}: {freq:10.2f}")

max_idx = -1  # Default value if doRC is False
if doRC:
    # Find the mode that aligns best with the reaction coordinate
    grad_avg_mw = normalize(grad_avg_mw.flatten())
    grad_diff_mw = np.flip(normalize(grad_diff_mw)) # Make sure the gradient is normalized
    projections = [np.abs(np.dot(grad_diff_mw, vec)) for vec in np.flip(e_vecs.T)] # Project the gradient difference onto each normal mode
    max_idx = np.argmax(projections) # Find the index of the maximum projection. This will correspond to the reaction coordinate mode
    print(f"Mode {max_idx+1} aligns best with the reaction coordinate with dot product {projections[max_idx]:.3f}")

# Calculation of the reduced mass orthogonal to the seam of crossing
vecs_mw = np.flip(e_vecs.T)  # Each row is a mass-weighted normal mode
reduced_masses = []

for vec_mw in np.flip(e_vecs.T):  # Uses all the modes orthogonal to the crossing seam
    mu_inv = np.sum(vec_mw**2 / masses3N) 
    mu = 1.0 / mu_inv
    reduced_masses.append(mu)

# Calculate the mean reduced mass orthogonal to the crossing seam
mean_mu = np.mean(reduced_masses)

gen_fake_OUTCAR(frequencies, max_idx+1, energy)

MECP_coords_ang = MECP_coords / 1.8897259886



write_molden(
    "MECP.info.molden",
    n_F_elem,
    MECP_coords_ang,
    frequencies,
    e_vecs.T,
    grad_diff_norm,
    grad_avg_norm, 
    mean_mu
)

