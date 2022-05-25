"""
    
    Functions which run phonopy and convert the output quantities to Natural (eV) units.

"""

import os
import phonopy
import numpy as np

import src.constants as const

def run_phonopy(phonon_file, k_mesh):
    """
        Given a phonon file and k mesh, Returns eigenvectors and frequencies in eV
    """

    # run phonopy in mesh mode 
    phonon_file.run_qpoints(k_mesh, with_eigenvectors=True)

    n_k = len(k_mesh)

    mesh_dict = phonon_file.get_qpoints_dict()

    eigenvectors_pre = mesh_dict['eigenvectors']

    # convert frequencies to correct units
    omega = 2*const.PI*(const.THz_To_eV)*mesh_dict['frequencies']

    num_atoms = phonon_file.primitive.get_number_of_atoms()
    num_modes = 3*num_atoms 

    # q, nu, i, alpha
    eigenvectors = np.zeros((n_k, num_modes, num_atoms, 3), dtype=complex)

    # sort the eigenvectors
    for q in range(n_k):
        for nu in range(num_modes):
            eigenvectors[q][nu][:][:] = np.array_split(
                    eigenvectors_pre[q].T[nu], num_atoms)

    return [eigenvectors, omega]

#def load_phonopy_file(material, io_parameters, supercell, poscar_path, force_sets_path, born_path,
#                        proc_id = 0, root_process = 0):
#
#
#    if os.path.exists(io_parameters['material_data_folder']+material+'/BORN'):
#
#        born_exists = True
#
#    else:
#
#        if proc_id == root_process:
#
#            print('\tThere is no BORN file for '+material)
#            print()
#
#        born_exists = False
#
#    if born_exists:
#
#        phonon_file = phonopy.load(
#                            supercell_matrix    = supercell,
#                            primitive_matrix    = 'auto',
#                            unitcell_filename   = poscar_path,
#                            force_sets_filename = force_sets_path,
#                            is_nac              = True,
#                            born_filename       = born_path
#                           )
#
#    else:
#
#        if proc_id == root_process:
#
#            print('\tNo BORN file found for : '+material)
#
#        phonon_file = phonopy.load(
#                            supercell_matrix    = supercell_data[material],
#                            primitive_matrix    = 'auto',
#                            unitcell_filename   = poscar_path,
#                            force_sets_filename = force_sets_path
#                           )
#
#    return [phonon_file, born_exists]


def get_phonon_file_data(phonon_file, born_exists):
    """
        Returns:

            n_atoms - number of atoms in primitive cell

            n_modes - number of modes = 3*n_atoms

            Transformation matrices

            pos_red_to_XYZ - reduced coordinate positions to XYZ

            pos_XYZ_to_red - XYZ coordinates to red

            recip_red_to_XYZ - reduced coordinates to XYZ

            recip_XYZ_to_red - XYZ coordinates to reduced

            eq_positions - equilibrium positions of atoms

            atom_masses - masses of atoms in eV

            A_list - Mass numbers (A)

            Z_list - atomic numbers (Z)

            born - Z_j

            dielectric - high frequency dielectric

    """

    num_atoms = phonon_file.primitive.get_number_of_atoms()
    num_modes = 3*num_atoms 

    A_list = phonon_file.primitive.get_masses()
    Z_list = phonon_file.primitive.get_atomic_numbers()

    eq_positions_XYZ = const.Ang_To_inveV*phonon_file.primitive.get_positions()

    atom_masses = const.AMU_To_eV*phonon_file.primitive.get_masses()

    primitive_mat = phonon_file.primitive.get_cell()

    pos_red_to_XYZ = const.Ang_To_inveV*np.transpose(primitive_mat)
    pos_XYZ_to_red = np.linalg.inv(pos_red_to_XYZ)

    a_vec = np.matmul(pos_red_to_XYZ, [1, 0, 0])
    b_vec = np.matmul(pos_red_to_XYZ, [0, 1, 0])
    c_vec = np.matmul(pos_red_to_XYZ, [0, 0, 1])

    recip_lat_a = 2*const.PI*(np.cross(b_vec, c_vec))/(np.matmul(a_vec, np.cross(b_vec, c_vec)))
    recip_lat_b = 2*const.PI*(np.cross(c_vec, a_vec))/(np.matmul(b_vec, np.cross(c_vec, a_vec)))
    recip_lat_c = 2*const.PI*(np.cross(a_vec, b_vec))/(np.matmul(c_vec, np.cross(a_vec, b_vec)))

    recip_red_to_XYZ = np.transpose([recip_lat_a, recip_lat_b, recip_lat_c])
    recip_XYZ_to_red = np.linalg.inv(recip_red_to_XYZ)

    if born_exists:

        born       = phonon_file.nac_params['born']
        dielectric = phonon_file.nac_params['dielectric']

    else:

        born       = np.zeros((num_atoms, 3, 3))
        dielectric = np.identity(3)

    return {
            'num_atoms': num_atoms,
            'num_modes': num_modes,
            'pos_red_to_XYZ': pos_red_to_XYZ,
            'pos_XYZ_to_red': pos_XYZ_to_red,
            'recip_red_to_XYZ': recip_red_to_XYZ,
            'recip_XYZ_to_red': recip_XYZ_to_red,
            'eq_positions_XYZ': eq_positions_XYZ,
            'atom_masses': atom_masses,
            'A_list': A_list,
            'Z_list': Z_list,
            'born': born,
            'dielectric': dielectric}
