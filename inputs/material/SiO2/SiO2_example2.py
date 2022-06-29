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

material = 'SiO2'
supercell_dim = [3, 3, 3]

# # number of atoms in the primitive cell
# num_atoms = 9
# mat_properties_dict = {
#         # dimension of supercell used in DFT calculation
#         "supercell_dim": [3., 3., 3.],
# 	"mass":{
# 		"e": const.M_ELEC,
# 		"p": const.M_NUCL,
# 		"n": const.M_NUCL
# 	},
# 	"N_list": {
# 		"e": np.array([
#                     14.0-4.0, 14.0-4.0, 14.0-4.0,
#                     8.0+2.0, 8.0+2.0, 8.0+2.0,
#                     8.0+2.0, 8.0+2.0, 8.0+2.0
#                     ]),
# 		"p": np.array([
#                     14.0, 14.0, 14.0,
#                     8.0, 8.0, 8.0,
#                     8.0, 8.0, 8.0
#                     ]),
# 		"n": np.array([
# 					28.0855 - 14.0, 28.0855 - 14.0, 28.0855 - 14.0,
# 					15.999 - 8.0, 15.999 - 8.0, 15.999 - 8.0,
#                     15.999 - 8.0, 15.999 - 8.0, 15.999 - 8.0])
# 	},
# 	"S_list": {
# 		"e": np.zeros((num_atoms, 3)),
# 		"p": np.zeros((num_atoms, 3)),
# 		"n": np.zeros((num_atoms, 3))
# 	},
# 	"L_list": {
# 		"e": np.zeros((num_atoms, 3)),
# 		"p": np.zeros((num_atoms, 3)),
# 		"n": np.zeros((num_atoms, 3))
# 	},
# 	"LxS_list": {
# 		"e": np.zeros((num_atoms, 3, 3)),
# 		"p": np.zeros((num_atoms, 3, 3)),
# 		"n": np.zeros((num_atoms, 3, 3))
# 	},
# }
