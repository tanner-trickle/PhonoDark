"""
    Material properties. For each atom in the primitive cell specify N, S, L, and L^i S^j
    values.

    Note: the order of the parameters must match the phonopy output. For a given material run phonopy
    once and examine the output to get this list

    TODO: automate this process, specify values for unique atoms in cell and fill these vectors with the
    appropriate values.
"""
import numpy as np
import src.constants as const

material = 'LiF'

# number of atoms in the primitive cell
num_atoms = 2
mat_properties_dict = {
        # dimension of supercell used in DFT calculation
        "supercell_dim": [2., 2., 2.], 
	"mass":{
		"e": const.M_ELEC,
		"p": const.M_NUCL,
		"n": const.M_NUCL
	},
	"N_list": {
		"e": np.array([
                    3.0-1.0, 
                    9.0+1.0
                    ]),
		"p": np.array([
                    3.0, 
                    9.0
                    ]),
		"n": np.array([
                    6.94 - 3.0, 
                    18.998 - 9.0])
	},
	"L_S_list": {
		"e": np.zeros(num_atoms),
		"p": np.zeros(num_atoms),
		"n": np.zeros(num_atoms)
	},
	"S_list": {
		"e": np.zeros((num_atoms, 3)),
		"p": np.zeros((num_atoms, 3)),
		"n": np.zeros((num_atoms, 3))
	},
	"L_list": {
		"e": np.zeros((num_atoms, 3)),
		"p": np.zeros((num_atoms, 3)),
		"n": np.zeros((num_atoms, 3))
	},
	"L_tens_S_list": {
		"e": np.zeros((num_atoms, 3, 3)),
		"p": np.zeros((num_atoms, 3, 3)),
		"n": np.zeros((num_atoms, 3, 3))
	},
}
