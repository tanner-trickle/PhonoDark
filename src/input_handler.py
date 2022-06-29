"""
    Handle all of the input to PhonoDark (PD).
"""

import os
import optparse
import numpy as np

import phonopy
import pymatgen.core as pmgCore

from src.phonopy_funcs import get_phonon_file_data

def load_material_inputs(material_input_filename, proc_id, root_process):
    """
        Loads all the material input file variables.
    """

    cwd = os.getcwd()

    mat_input_mod_name = os.path.splitext(os.path.basename(material_input_filename))[0] 

    # load the variables from the input file
    mat_mod = import_file(mat_input_mod_name, os.path.join(cwd, material_input_filename))

    # load the variables which can be found from the phonopy configuration 
    phonopy_config, born_exists = load_phonopy_configuration(mat_mod.supercell_dim, material_input_filename, 
            proc_id, root_process)

    phonopy_config_info = get_phonon_file_data(phonopy_config, born_exists) 

    n_atoms = phonopy_config_info['num_atoms']
    n_p_list = phonopy_config_info['Z_list']
    n_n_list = phonopy_config_info['A_list'] - n_p_list

    n_e_list = []

    # automatically fill in best guess for number of electrons at each site
    composition = pmgCore.Composition(mat_mod.material)
    oxi_state_guesses = composition.oxi_state_guesses()

    for s, symbol in enumerate(phonopy_config_info['symbols']):

        oxi_number = 0

        if len(oxi_state_guesses) >= 1:
            if symbol in oxi_state_guesses[0]:
                oxi_number = oxi_state_guesses[0][symbol]

        n_e_list.append( n_p_list[s] - oxi_number )

    S_list = {'e': np.zeros(n_atoms),
                'p': np.zeros(n_atoms),
                'n': np.zeros(n_atoms)}
    L_list = {'e': np.zeros((n_atoms, 3)),
                'p': np.zeros((n_atoms, 3)),
                'n': np.zeros((n_atoms, 3))}
    LxS_list = {'e': np.zeros((n_atoms, 3, 3)),
                'p': np.zeros((n_atoms, 3, 3)),
                'n': np.zeros((n_atoms, 3, 3))}

    if 'mat_mod.properties_dict' in locals():
        if 'S_list' in mat_mod.properties_dict:
            S_list = mat_mod.properties_dict['S_list']

        if 'L_list' in mat_mod.properties_dict:
            L_list = mat_mod.properties_dict['L_list']

        if 'LxS_list' in mat_mod.properties_dict:
            LxS_list = mat_mod.properties_dict['LxS_list']

    material_input_dict = {
                'name': mat_mod.material, 
                'supercell_dim': mat_mod.supercell_dim, 
                'n_atoms': n_atoms, 
                'properties': {
                                'N_list': {
                                    'e': np.array(n_e_list), 
                                    'p': n_p_list, 
                                    'n': n_n_list
                                },
                                'S_list': S_list,
                                'L_list': L_list,
                                'LxS_list': LxS_list
                }
            }

    return material_input_dict


def load_phonopy_configuration(supercell_dim, material_input_filename, proc_id, root_process):
    """
        Loads the phonopy configuration.
    """
    
    # NOTE: we assume that the material input file is in the same folder as the
    # POSCAR, FORCE_SETS, and BORN files

    material_input_folder = os.path.split(material_input_filename)[0]

    poscar_path = os.path.join(material_input_folder, 'POSCAR')
    force_sets_path = os.path.join(material_input_folder, 'FORCE_SETS')
    born_path = os.path.join(material_input_folder, 'BORN')

    # check if the BORN file exists
    if os.path.exists(born_path):
        born_exists = True
    else:
        if proc_id == root_process: 
            print('  There is no BORN file. Phonopy calculations will process with .NAC. = False\n')
        born_exists = False

    if born_exists: 
        phonopy_config = phonopy.load(
                        supercell_matrix    = supercell_dim,
                        primitive_matrix    = 'auto',
                        unitcell_filename   = poscar_path,
                        force_sets_filename = force_sets_path,
                        is_nac              = True,
                        born_filename       = born_path
                       )
    else:
        phonopy_config = phonopy.load(
                            supercell_matrix    = supercell_dim,
                            primitive_matrix    = 'auto',
                            unitcell_filename   = poscar_path,
                            force_sets_filename = force_sets_path
                           )

    return phonopy_config, born_exists

def load_inputs(input_options, proc_id, root_process):
    """
        Load the input parameters from the relevant files.
    """

    material_input_filename = input_options['m']

    material_input_dict = load_material_inputs(material_input_filename, proc_id, root_process)

    return {
            'material': material_input_dict,
            # 'physics_model': phys_mod_input_dict
            }

def import_file(full_name, path):
    """
        Import a python module from a path. 3.4+ only.
        Does not call sys.modules[full_name] = path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_cmdline_arguments():
    """
        Returns dictionary of command line arguments supplied to PhonoDark.
    """

    parser = optparse.OptionParser()
    parser.add_option('-m', action="store", default="", 
            help="Material info file. Contains the crystal lattice degrees of freedom for a given material.")
    parser.add_option('-p', action="store", default="", 
            help="Physics model input. Contains the coupling coefficients and defines which operators enter the scattering potential.")
    parser.add_option('-n', action="store", default="", 
            help="Numerics input. Sets the parameters used for the integration over momentum space and the input and output files.")

    options_in, args = parser.parse_args()

    options = vars(options_in)

    cmd_input_okay = False
    if options['m'] != '' and options['p'] != '' and options['n'] != '':
        cmd_input_okay = True

    return options, cmd_input_okay
