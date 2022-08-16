"""
    Compute the DM - phonon scattering rate for a general effective operator.

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
import sys
import numpy as np

import src.parallel_util as put 
import src.input_handler as input_handler
import src.output_handler as output_handler
import src.physics as physics
import src.mesh as mesh
import src.phonopy_funcs as phonopy_funcs

###############################################################################

version = "1.2.0"
comm = MPI.COMM_WORLD # initializing MPI
n_proc = comm.Get_size() # total number of processors
proc_id = comm.Get_rank() # processor if
root_process = 0 # ID of root process

if proc_id == root_process:

    print('\n--- PhonoDark ---\n')
    print('  version: '+version+'\n')
    print('  Running on '+str(n_proc)+' processors\n')
    print('------\n')

# parse command line inputs
input_options, cmd_input_okay = input_handler.get_cmdline_arguments()

if not cmd_input_okay:

    if proc_id == root_process:
        print('ERROR:')
        print("\t- material info file\n"+
        "\t- physics model file\n"+
        "\t- numerics input file\n\n"+
        "must be present. These are added with the -m, -p, -n input flags. Add -h (--help) flags for more help.\n\n"+
        "See inputs/ folder for specific example inputs")

    MPI.Finalize()
    sys.exit()

###############################################################################
# load inputs
###############################################################################

inputs = input_handler.load_inputs(input_options, proc_id, root_process)

# print the inputs
if proc_id == root_process:
    input_handler.print_inputs(inputs)

###############################################################################
# initialize calculation
###############################################################################

# number of jobs to do
n_masses    = len(inputs['physics_model']['dm_properties']['mass_list'])
if 'vE' in inputs['physics_model']['physics_parameters']:
    n_vE = len(inputs['physics_model']['physics_parameters']['vE'])
elif 'times' in inputs['physics_model']['physics_parameters']:
    n_vE = len(inputs['physics_model']['physics_parameters']['times'])
else:
    n_vE = 1

job_list      = None
job_list_recv = None
if proc_id == root_process:

    print('Configuring calculation ...\n')

    # parallelize over masses and times
    num_jobs = n_masses*n_vE
    print('  Total number of jobs : '+str(num_jobs))
    print()

    total_job_list = []

    for m in range(n_masses):
        for t in range(n_vE):

            total_job_list.append([m, t])

    job_list = put.generate_job_list(n_proc, np.array(total_job_list))

# scatter the job list
job_list_recv = comm.scatter(job_list, root=root_process)

###############################################################################
# perform calculation
###############################################################################

if proc_id == root_process:
    # print('\nDone loading DFT files\n\n------\n')
    print('Starting binned rate calculation ...\n')

binned_rate = np.zeros((n_masses, n_vE, inputs['numerics']['numerics']['n_E_bins']))
total_binned_rate = np.zeros((n_masses, n_vE, inputs['numerics']['numerics']['n_E_bins']))

# compute for all the jobs the processor is supposed to
for job in range(len(job_list_recv)):

    if (job_list_recv[job, 0] != -1 and job_list_recv[job, 1] != -1 ):

        job_id = job_list_recv[job]

        mass = inputs['physics_model']['dm_properties']['mass_list'][int(job_id[0])]
        
        if 'vE' in inputs['physics_model']['physics_parameters']:
            vE_vec = inputs['physics_model']['physics_parameters']['vE'][int(job_id[1])]
        elif 'times' in inputs['physics_model']['physics_parameters']:
            time = inputs['physics_model']['physics_parameters']['times'][int(job_id[1])]
            vE_vec = physics.create_vE_vec(time)
        else:
            vE_vec = physics.create_vE_vec(0.0)

        # create q integration mesh
        delta = 2*(
                inputs['physics_model']['physics_parameters']['power_V']
                - inputs['physics_model']['physics_parameters']['Fmed_power'] )

        [q_XYZ_list, jacob_list] = mesh.create_q_mesh(mass,                                     
                inputs['physics_model']['physics_parameters']['threshold'], 
                vE_vec, 
                inputs['numerics']['numerics'],
                inputs['material']['q_DW'], 
                delta)

        k_red_list = mesh.generate_k_red_mesh_from_q_XYZ_mesh(
                q_XYZ_list, 
                inputs['material']['phonopy_config_info']['recip_red_to_XYZ'])
        G_XYZ_list = mesh.get_G_XYZ_list_from_q_XYZ_list(
                q_XYZ_list, 
                inputs['material']['phonopy_config_info']['recip_red_to_XYZ'])

        # run phonopy
        [ph_eigenvectors, ph_omega] = phonopy_funcs.run_phonopy(inputs['material']['phonopy_config'], k_red_list)

        # compute Debye-Waller tensor
        W_tensor = physics.calculate_W_tensor(
                inputs['material']['phonopy_config'],
                inputs['material']['phonopy_config_info']['num_atoms'], 
                inputs['material']['phonopy_config_info']['atom_masses'], 
                inputs['numerics']['numerics']['n_DW_x'], 
                inputs['numerics']['numerics']['n_DW_y'], 
                inputs['numerics']['numerics']['n_DW_z'], 
                inputs['material']['phonopy_config_info']['recip_red_to_XYZ'])

        binned_rate[int(job_id[0]), int(job_id[1]), :] += \
                                        physics.calc_binned_rate(
                                                mass, vE_vec, inputs, 
                                                q_XYZ_list, G_XYZ_list, 
                                                jacob_list, W_tensor, 
                                                ph_omega, ph_eigenvectors)

###############################################################################
# save: binned_rate, total_rate
###############################################################################

# reduce output
comm.Reduce(binned_rate, total_binned_rate, op=MPI.SUM, root=root_process)

if proc_id == root_process:

    print('Saving output...')

    out_filename = output_handler.create_output_filename(inputs)

    output_handler.save_output(out_filename, inputs, total_binned_rate, version)
