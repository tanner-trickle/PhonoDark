"""
    Handle all of the input to PhonoDark (PD).
"""

import os
import optparse
import numpy as np

import phonopy
import pymatgen.core as pmgCore

from src.phonopy_funcs import get_phonon_file_data, run_phonopy

def print_inputs(inputs):

    print('Inputs :\n')
    print('  Material :\n')
    print('    Name : '+inputs['material']['name']+'\n')
    print('    DFT supercell grid size : '+str(inputs['material']['supercell_dim'])+'\n')
    print('    N : '+str(inputs['material']['properties']['N_list'])+'\n')
    print('    S : '+str(inputs['material']['properties']['S_list'])+'\n')
    print('    L : '+str(inputs['material']['properties']['L_list'])+'\n')
    print('    LxS : '+str(inputs['material']['properties']['LxS_list'])+'\n')
    print('  Physics Model :\n')
    print('    DM spin : '+str(inputs['physics_model']['dm_properties']['spin'])+'\n')
    print('    DM mass (eV) : '+str(inputs['physics_model']['dm_properties']['mass_list'])+'\n')
    print('    d log V / d log q : '+str(inputs['physics_model']['physics_parameters']['power_V'])+'\n')
    print('    - d log F_med / d log q : '+str(inputs['physics_model']['physics_parameters']['Fmed_power'])+'\n')
    if 'vE' in inputs['physics_model']['physics_parameters']:
        print('    vE : '+str(inputs['physics_model']['physics_parameters']['vE'])+'\n')
    elif 'times' in inputs['physics_model']['physics_parameters']:
        print('    Time of day : '+str(inputs['physics_model']['physics_parameters']['times'])+'\n')
    else:
        print('    vE : VE * (0, 0, 1)\n') # default vE along z direction
    print('    Threshold : '+str(inputs['physics_model']['physics_parameters']['threshold'])+' eV\n')
    print('    c coefficients : '+str(list(inputs['physics_model']['c_dict'].keys()))+'\n')
    print('  Numerics :\n')
    print('    Energy bin width : '+str(inputs['numerics']['numerics']['energy_bin_width'])+' eV\n')
    print('    Number of energy bins (above threshold): '+str(inputs['numerics']['numerics']['n_E_bins']))
    print('    N_a : '+str(inputs['numerics']['numerics']['n_a'])+'\n')
    print('    N_b : '+str(inputs['numerics']['numerics']['n_b'])+'\n')
    print('    N_c : '+str(inputs['numerics']['numerics']['n_c'])+'\n')
    if inputs['numerics']['numerics']['special_mesh']:
        print('    Special mesh : True\n')
    else:
        print('    power_a : '+str(inputs['numerics']['numerics']['power_a'])+'\n')
        print('    power_b : '+str(inputs['numerics']['numerics']['power_b'])+'\n')
        print('    power_c : '+str(inputs['numerics']['numerics']['power_c'])+'\n')
    print('    q cut : '+str(inputs['numerics']['numerics']['q_cut'])+'\n')
    print('    N_DW_x : '+str(inputs['numerics']['numerics']['n_DW_x'])+'\n')
    print('    N_DW_y : '+str(inputs['numerics']['numerics']['n_DW_y'])+'\n')
    print('    N_DW_z : '+str(inputs['numerics']['numerics']['n_DW_z'])+'\n')
    print('------\n')


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

    N_list = {'e': np.array(n_e_list), 
                'p': n_p_list, 
                'n': n_n_list}
    S_list = {'e': np.zeros((n_atoms, 3)),
                'p': np.zeros((n_atoms, 3)),
                'n': np.zeros((n_atoms, 3))}
    L_list = {'e': np.zeros((n_atoms, 3)),
                'p': np.zeros((n_atoms, 3)),
                'n': np.zeros((n_atoms, 3))}
    LxS_list = {'e': np.zeros((n_atoms, 3, 3)),
                'p': np.zeros((n_atoms, 3, 3)),
                'n': np.zeros((n_atoms, 3, 3))}

    # overwrite the variables if they're defined in the input file
    if hasattr(mat_mod, 'properties_dict'):
        if 'N_list' in mat_mod.properties_dict:
            N_list = mat_mod.properties_dict['N_list']
        if 'S_list' in mat_mod.properties_dict:
            S_list = mat_mod.properties_dict['S_list']

        if 'L_list' in mat_mod.properties_dict:
            L_list = mat_mod.properties_dict['L_list']

        if 'LxS_list' in mat_mod.properties_dict:
            LxS_list = mat_mod.properties_dict['LxS_list']

    [ph_eigenvectors, ph_omega] = run_phonopy(phonopy_config,
                                            np.zeros((1, 3)))

    # 'Debye-Waller' scale
    q_DW = np.sqrt(np.amax(ph_omega)*np.amax(phonopy_config_info['atom_masses']))

    material_input_dict = {
                'name': mat_mod.material, 
                'supercell_dim': mat_mod.supercell_dim, 
                'properties': {
                    'N_list': N_list,
                    'S_list': S_list,
                    'L_list': L_list,
                    'LxS_list': LxS_list
                },
                'phonopy_config_info': phonopy_config_info,
                'phonopy_config': phonopy_config,
                'q_DW': q_DW
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

def load_physics_model_inputs(filename, proc_id, root_process):
    cwd = os.getcwd()

    phys_mod_input_mod_name = os.path.splitext(os.path.basename(filename))[0] 

    # load the variables from the input file
    phys_mod = import_file(phys_mod_input_mod_name, os.path.join(cwd, filename))

    return {'physics_parameters': phys_mod.physics_parameters,
            'dm_properties': phys_mod.dm_properties_dict,
            'c_dict': phys_mod.c_dict, 
            'c_dict_form':phys_mod.c_dict_form}

def load_numerics_inputs(filename, proc_id, root_process):

    cwd = os.getcwd()

    numerics_mod_input_mod_name = os.path.splitext(os.path.basename(filename))[0] 

    # load the variables from the input file
    numerics_mod = import_file(numerics_mod_input_mod_name, os.path.join(cwd, filename))

    return {'io_parameters': numerics_mod.io_parameters, 
            'numerics': numerics_mod.numerics_parameters}

def load_inputs(input_options, proc_id, root_process):
    """
        Load the input parameters from the relevant files.
    """

    material_input_filename = input_options['m']
    physics_model_input_filename = input_options['p']
    numerics_input_filename = input_options['n']

    numerics_mod_input_name = os.path.splitext(os.path.basename(numerics_input_filename))[0] 
    material_mod_input_name = os.path.splitext(os.path.basename(material_input_filename))[0] 
    physics_model_mod_input_name = os.path.splitext(os.path.basename(physics_model_input_filename))[0] 

    material_input_dict = load_material_inputs(material_input_filename, proc_id, root_process)
    physics_model_input_dict = load_physics_model_inputs(physics_model_input_filename, proc_id, root_process)
    numerics_input_dict = load_numerics_inputs(numerics_input_filename, proc_id, root_process)

    return {
            'material': material_input_dict,
            'physics_model': physics_model_input_dict, 
            'numerics': numerics_input_dict, 
            'mod_names': {
                'm': material_mod_input_name, 
                'p': physics_model_mod_input_name, 
                'n': numerics_mod_input_name
                }
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
