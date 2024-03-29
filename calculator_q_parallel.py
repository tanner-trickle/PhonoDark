"""
    Compute the DM - phonon scattering rate for a general effective operator.

    Parallelizes over points in the momentum space integration

    Necessary input files:
        - DFT input files:
            - POSCAR - equilibrium positions of atoms
            - FORCE_SETS - second order force constants
            (optional) BORN - Born effective charges

        - inputs/
            - material/
                - (material name)/ 
                    (material_info).py - set the material parameters such as nucleon numbers and spins
            - numerics/
                (numerics_info).py - list of numerics parameters
            - physics_model/
                (physics_model_info).py - parameters defining the scattering potential
                    

    Created by : Tanner Trickle, Zhengkang Zhang
"""

from mpi4py import MPI
import numpy as np
import phonopy
import os
import sys
import optparse
import math

import src.constants as const
import src.parallel_util as put 
import src.mesh as mesh
import src.phonopy_funcs as phonopy_funcs
import src.physics as physics
import src.hdf5_output as hdf5_output

#####

version = "1.1.0"

# initializing MPI
comm = MPI.COMM_WORLD

# total number of processors
n_proc = comm.Get_size()

# processor if
proc_id = comm.Get_rank()

# ID of root process
root_process = 0

if proc_id == root_process:

    print('\n--- Dark Matter - Phonon Scattering Rate Calculator ---\n')
    print('  version: '+version+'\n')
    print('  Running on '+str(n_proc)+' processors\n')
    print('------\n')

# Parse the input parameters ###
cwd = os.getcwd()

parser = optparse.OptionParser()
parser.add_option('-m', action="store", default="", 
        help="Material info file. Contains the crystal lattice degrees of freedom for a given material.")
parser.add_option('-p', action="store", default="", 
        help="Physics model input. Contains the coupling coefficients and defines which operators enter the scattering potential.")
parser.add_option('-n', action="store", default="", 
        help="Numerics input. Sets the parameters used for the integration over momentum space and the input and output files.")

options_in, args = parser.parse_args()

options = vars(options_in)

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

if options['m'] != '' and options['p'] != '' and options['n'] != '':

    # import modules
    material_input = options['m']
    physics_model_input = options['p']
    numerics_input = options['n']

    mat_input_mod_name = os.path.splitext(os.path.basename(material_input))[0] 
    phys_input_mod_name = os.path.splitext(os.path.basename(physics_model_input))[0] 
    num_input_mod_name = os.path.splitext(os.path.basename(numerics_input))[0] 

    mat_mod = import_file(mat_input_mod_name, os.path.join(cwd, material_input))
    phys_mod = import_file(phys_input_mod_name, os.path.join(cwd, physics_model_input))
    num_mod = import_file(num_input_mod_name, os.path.join(cwd, numerics_input))

    if proc_id == root_process:

        print('Inputs :\n')
        print('  Material :\n')
        print('    Name : '+mat_mod.material+'\n')
        print('    DFT supercell grid size : '+str(mat_mod.mat_properties_dict['supercell_dim'])+'\n')
        print('    N : '+str(mat_mod.mat_properties_dict['N_list'])+'\n')
        print('    L.S : '+str(mat_mod.mat_properties_dict['L_S_list'])+'\n')
        print('    S : '+str(mat_mod.mat_properties_dict['S_list'])+'\n')
        print('    L : '+str(mat_mod.mat_properties_dict['L_list'])+'\n')
        print('    L.S_2 : '+str(mat_mod.mat_properties_dict['L_tens_S_list'])+'\n')
        print('  Physics Model :\n')
        print('    DM spin : '+str(phys_mod.dm_properties_dict['spin'])+'\n')
        print('    DM mass : '+str(phys_mod.dm_properties_dict['mass_list'])+'\n')
        print('    d log V / d log q : '+str(phys_mod.physics_parameters['power_V'])+'\n')
        print('    - d log F_med / d log q : '+str(phys_mod.physics_parameters['Fmed_power'])+'\n')
        print('    Time of day : '+str(phys_mod.physics_parameters['times'])+'\n')
        print('    Threshold : '+str(phys_mod.physics_parameters['threshold'])+' eV\n')
        print('    c coefficients : '+str(phys_mod.c_dict)+'\n')
        print('  Numerics :\n')
        print('    Energy bin width : '+str(num_mod.numerics_parameters['energy_bin_width'])+' eV\n')
        print('    N_a : '+str(num_mod.numerics_parameters['n_a'])+'\n')
        print('    N_b : '+str(num_mod.numerics_parameters['n_b'])+'\n')
        print('    N_c : '+str(num_mod.numerics_parameters['n_c'])+'\n')
        if num_mod.numerics_parameters['special_mesh']:
            print('    Special mesh : True\n')
        else:
            print('    power_a : '+str(num_mod.numerics_parameters['power_a'])+'\n')
            print('    power_b : '+str(num_mod.numerics_parameters['power_b'])+'\n')
            print('    power_c : '+str(num_mod.numerics_parameters['power_c'])+'\n')
        print('    q cut : '+str(num_mod.numerics_parameters['q_cut'])+'\n')
        print('    N_DW_x : '+str(num_mod.numerics_parameters['n_DW_x'])+'\n')
        print('    N_DW_y : '+str(num_mod.numerics_parameters['n_DW_y'])+'\n')
        print('    N_DW_z : '+str(num_mod.numerics_parameters['n_DW_z'])+'\n')
        print('------\n')

    job_list      = None
    job_list_recv = None

    if proc_id == root_process:

        print('Configuring calculation ...\n')


    num_times     = len(phys_mod.physics_parameters['times'])
    if num_times > 1:

        print('--- ERROR ---\n')
        print('    This calculation must be run for a single time point.')
        print('\n----------')

        sys.exit()

    time = phys_mod.physics_parameters['times'][0]
    material = mat_mod.material

    if proc_id == root_process:

        print('Done configuring calculation\n\n------\n')
        print('Loading DFT files ...\n')

    # load phonon file
    poscar_path = os.path.join(
            os.path.split(material_input)[0], 'POSCAR'
            )
    force_sets_path = os.path.join(
            os.path.split(material_input)[0], 'FORCE_SETS'
            )
    born_path = os.path.join(
            os.path.split(material_input)[0], 'BORN'
            )

    # check if the born file exists
    if os.path.exists(born_path):
        born_exists = True
    else:
        if proc_id == root_process: 
            print('  There is no BORN file for '+material+'. PHONOPY calculations will process with .NAC. = FALSE\n')
        born_exists = False

    if born_exists: 
        phonon_file = phonopy.load(
                        supercell_matrix    = mat_mod.mat_properties_dict['supercell_dim'],
                        primitive_matrix    = 'auto',
                        unitcell_filename   = poscar_path,
                        force_sets_filename = force_sets_path,
                        is_nac              = True,
                        born_filename       = born_path
                       )
    else:
        phonon_file = phonopy.load(
                            supercell_matrix    = mat_mod.mat_properties_dict['supercell_dim'],
                            primitive_matrix    = 'auto',
                            unitcell_filename   = poscar_path,
                            force_sets_filename = force_sets_path
                           )
    if proc_id == root_process:
        print('\nDone loading DFT files\n\n------\n')
        print('Starting rate computation ...\n')

    # using this to set the maximum bin number, otherwise the size of the diff rate
    # vector will change depending on the q-mesh
    [ph_eigenvectors_delta_E, ph_omega_delta_E] = phonopy_funcs.run_phonopy(phonon_file, 
                    [[0., 0., 0.]])

    max_delta_E = 4*np.amax(ph_omega_delta_E)

    max_bin_num = math.floor((
        max_delta_E - phys_mod.physics_parameters["threshold"]
                    )/num_mod.numerics_parameters["energy_bin_width"]) + 1

    phonopy_params = phonopy_funcs.get_phonon_file_data(phonon_file, born_exists)

    vE_vec = physics.create_vE_vec(time)
    delta = 2*phys_mod.physics_parameters['power_V'] - 2*phys_mod.physics_parameters['Fmed_power']

    W_tensor_send = None

    if proc_id == root_process:

        print('Computing W tensor ... \n')

        W_tensor_main = physics.calculate_W_tensor(phonon_file,
                                                phonopy_params['num_atoms'],
                                                phonopy_params['atom_masses'],
                                                num_mod.numerics_parameters['n_DW_x'], 
                                                num_mod.numerics_parameters['n_DW_y'], 
                                                num_mod.numerics_parameters['n_DW_z'], 
                                                phonopy_params['recip_red_to_XYZ'])


        W_tensor_send = []

        for n in range(n_proc):

            W_tensor_send.append(W_tensor_main)

    W_tensor = comm.scatter(W_tensor_send, root=root_process)

    # total list of all the q vectors
    q_list_total = []

    num_jobs = 0

    num_masses = len(phys_mod.dm_properties_dict['mass_list'])

    for m in range(num_masses):

        [q_XYZ_list, jacob_list] = mesh.create_q_mesh(
                                        phys_mod.dm_properties_dict['mass_list'][m], 
                                        phys_mod.physics_parameters['threshold'], 
                                        vE_vec, 
                                        num_mod.numerics_parameters,
                                        phonon_file,
                                        phonopy_params['atom_masses'],
                                        delta)

        k_red_list = mesh.generate_k_red_mesh_from_q_XYZ_mesh(q_XYZ_list, 
                phonopy_params['recip_red_to_XYZ'])
        G_XYZ_list = mesh.get_G_XYZ_list_from_q_XYZ_list(q_XYZ_list,
                phonopy_params['recip_red_to_XYZ'])
        
        q_list_total.append({
                'q_XYZ': q_XYZ_list,
                'k_red': k_red_list,
                'G_XYZ': G_XYZ_list,
                'jacob': jacob_list
            })

        num_jobs += len(k_red_list)

    q_list_total_recv = q_list_total

    if proc_id == root_process:
        print('Done computing q meshes.')
        print()
        print('Sending job lists...')
        print()
        
        print('\tTotal number of jobs : '+str(num_jobs))
        print()

        total_job_list = []

        for m in range(num_masses):
            for q in range(len(q_list_total_recv[m]['k_red'])):

                total_job_list.append([m, q])

        job_list = put.generate_job_list(n_proc, np.array(total_job_list))

    # scatter the job list
    job_list_recv = comm.scatter(job_list, root=root_process)

    diff_rate_list = np.zeros((num_masses, max_bin_num))
    binned_rate_list = np.zeros((num_masses, phonopy_params['num_modes']))
    total_rate_list = np.zeros(num_masses)

    diff_rate = np.zeros((num_masses, max_bin_num))
    binned_rate = np.zeros((num_masses, phonopy_params['num_modes']))
    total_rate = np.zeros(num_masses)

    if proc_id == root_process:
        print('\nDone loading DFT files\n\n------\n')
        print('Starting rate computation ...\n')

    first_job = True
    
    for job in range(len(job_list_recv)):

        if (job_list_recv[job, 0] != -1 and job_list_recv[job, 1] != -1 ):

            job_id = job_list_recv[job]

            mass_index = int(job_id[0])
            mass = phys_mod.dm_properties_dict['mass_list'][mass_index]
            q_index = int(job_id[1])

            if first_job and proc_id == root_process:

                print('    Number of atoms : '+str(phonopy_params['num_atoms'])+'\n')
                print('    Number of modes : '+str(phonopy_params['num_modes'])+'\n')
                print('    Atom masses : '+str(phonopy_params['atom_masses'])+'\n')
            
            if born_exists and proc_id == root_process and first_job:
                
                print('    dielectric : \n')
                print(phonopy_params['dielectric'])
                print()
            
            if not phys_mod.include_screen:
                if proc_id == root_process:
                    print('  Include screen is FALSE. Setting the dielectric to the identity.\n')
                    print()

                phonopy_params['dielectric'] = np.identity(3)

            if first_job and proc_id == root_process:
                print('  Done loading data to PHONOPY\n')

            # run phonopy
            [ph_eigenvectors, ph_omega] = phonopy_funcs.run_phonopy(phonon_file, 
                    [q_list_total_recv[mass_index]['k_red'][q_index]])

            if 'special_model' in phys_mod.physics_parameters.keys():

                if first_job and proc_id == root_process:
                    print('Computing the scattering rate for a specific model...')
                    print()
                    print('    Model : '+phys_mod.physics_parameters['special_model'])
                    print()

                if phys_mod.physics_parameters['special_model'] == 'dark_photon':

                    [diff_rate, binned_rate, total_rate] = physics.calc_diff_rates_SI_q(
                                                    mass, 
                                                    q_list_total_recv[mass_index]['q_XYZ'], 
                                                    q_list_total_recv[mass_index]['G_XYZ'],
                                                    q_list_total_recv[mass_index]['jacob'],
                                                    phys_mod.physics_parameters,
                                                    vE_vec, 
                                                    num_mod.numerics_parameters, 
                                                    phonopy_params,
                                                    ph_omega, 
                                                    ph_eigenvectors,
                                                    W_tensor, 
                                                    mat_mod.mat_properties_dict, 
                                                    phys_mod.dm_properties_dict, 
                                                    phonon_file, max_bin_num, q_index, phys_mod.c_dict)

                else:

                    if first_job and proc_id == root_process:
                        print('    This specific model does not have a unique implementation.')
                        print('    Try using the more general calculation by specifying the ')
                        print('    coefficients of the operators you want to include.')
                        print()

            else:

            # compute the differential, binned and total rates
                [diff_rate, binned_rate, total_rate] = physics.calc_diff_rates_general_q(mass, 
                                                    q_list_total_recv[mass_index]['q_XYZ'], 
                                                    q_list_total_recv[mass_index]['G_XYZ'], 
                                                    q_list_total_recv[mass_index]['jacob'], 
                                                    phys_mod.physics_parameters,
                                                    vE_vec, 
                                                    num_mod.numerics_parameters, 
                                                    phonopy_params,
                                                    ph_omega, 
                                                    ph_eigenvectors,
                                                    W_tensor, 
                                                    phys_mod.c_dict, 
                                                    mat_mod.mat_properties_dict, 
                                                    phys_mod.dm_properties_dict, 
                                                    phys_mod.c_dict_form, phonon_file, 
                                                    max_bin_num, q_index)

            diff_rate_list[mass_index] += np.real(diff_rate)
            binned_rate_list[mass_index] += np.real(binned_rate)
            total_rate_list[mass_index] += np.real(total_rate)

            first_job = False

    if proc_id == root_process:
        print('Done computing rate. Returning all data to root node to write.\n\n------\n')

    # return data back to root
    all_diff_rate_list   = comm.gather(diff_rate_list, root=root_process)
    all_binned_rate_list = comm.gather(binned_rate_list, root=root_process)
    all_total_rate_list  = comm.gather(total_rate_list, root=root_process)

    # write to output file
    if proc_id == root_process:

        out_filename = os.path.join(
                num_mod.io_parameters['output_folder'], 
                material+'_'+phys_input_mod_name+'_'+num_input_mod_name+num_mod.io_parameters['output_filename_extra']+'.hdf5')
        
        hdf5_output.hdf5_write_output_q(out_filename,
                                       num_mod.numerics_parameters,
                                       phys_mod.physics_parameters,
                                       phys_mod.dm_properties_dict,
                                       phys_mod.c_dict,
                                       all_total_rate_list,
                                       n_proc, 
                                       material,
                                       all_diff_rate_list,
                                       all_binned_rate_list)

        print('Done writing rate.\n\n------\n')

else:

    if proc_id == root_process:
        print('ERROR:')
        print("\t- material info file\n"+
        "\t- physics model file\n"+
        "\t- numerics input file\n\n"+
        "must be present. These are added with the -m, -p, -n input flags. Add -h (--help) flags for more help.\n\n"+
        "See inputs/material/GaAs/GaAs_example.py, inputs/physics_model/dark_photon_example.py, inputs/numerics/standard.py for examples.")

    sys.exit()
