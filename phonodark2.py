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

import src.input_handler as input_handler

###########

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

# load inputs
inputs = input_handler.load_inputs(input_options, proc_id, root_process)

print(inputs)

# initialize calculation

# perform calculation

# save outputs


