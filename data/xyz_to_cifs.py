from ase.io import read, write
import numpy as np
import os

# Converts an appended .xyz to a folder of CIFs

# Relevant filenames
refcode_path = 'opt-refcodes.csv' # path to refcodes
xyz_path = 'opt-geometries.xyz' # path to XYZ of all structures

# ----------------------
refs = np.genfromtxt(refcode_path,delimiter=',',dtype=str)
mofs = read(xyz_path,index=':')

for i, mof in enumerate(mofs):
	write(os.path.join(refs[i]+'.cif'),mof)
