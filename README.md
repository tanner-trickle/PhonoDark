# Dark Matter - Phonon Scattering Rate Calculator

**version: 1.0.0**

Computes dark matter (DM)-phonon scattering rate for a general scattering potential and target. 

The user can specify:

- target material (with appropriate DFT input files)
- crystal lattice degrees of freedom (nucleon number, electron spin, etc.)
- list of DM masses
- list of times (to compute the daily modulation)
- any scattering potential: any scattering potential can be written in terms of a basis of non-relativistic operators, and the user can specify the relative combinations of the operators
- experimental energy threshold
- DM mediator form factor

The program will then output:

- total scattering rate
- differential scattering rate : total rate binned in (user specified) energy bins, allowing new thresholds to be computed for in post-processing, or including energy dependent backgrounds 
- scattering rate by band : the scattering rate to the individual phonon bands in the material

the program is also parallelized with OPENMPI, allowing for speed ups on a local machine or on a cluster.

# Installation

## Recommended

1) Install `(mini)conda`, an environment management system for python, from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html])
2)  

    > conda create --name <env> --file requirements.txt
    > conda activate <env>

## Not recommended

This program does not have very many dependencies so you might be able to get away with running

    > python calculator.py

and `pip install` the relevant modules as the errors pop up.

## Check

To check installation, run

    > python calculator.py

in the directory containing `calculator.py`. If all of the necessary packages are installed the program will terminate on its own and print an error message asking for input files.

# Input

The program requires a few different input files to run, all of which should be stored inside `inputs/`. There are two catergories of inputs, the ones generated from a density functional theory (DFT) calculation, and those specified by the user, e.g. the DM model.

## DFT

The DM-phonon scattering rate depends on the phonon energies and eigenvectors. We compute this with [phonopy](https://phonopy.github.io/phonopy/), "an open source package for phonon calculations at harmonic and quasi-harmonic levels". For our purposes this program takes in structural (equilibrium positions) and dynamics (second derivatives with respect to atomic displacements of the potential energies) information and diagonalize the Hamiltonian in terms of canonical phonon energies and eigenvectors.

Therefore in order to compute the scattering rate from a given target, you must have the standard input files necessary for `phonopy` to run:

- **POSCAR** : equilibrium positions
- **FORCE_SETS** : second derivatives of the potential energy with respect to ionic displacements
- (Optional) **BORN**: effective ionic charges

These files should be located in `inputs/materials/(material name)/`, for example the LiF POSCAR, FORCE_SETS, and BORN files are in `inputs/materials/LiF/`.

More examples can be found at an online database [here](http://phonondb.mtl.kyoto-u.ac.jp/) hosted by the creators of phonopy.

More information about `phonopy` can be found [here](https://phonopy.github.io/phonopy/).

## User

The DM-phonon scattering rate also depends on many different particle physics parameters which can be adjusted by changing these input files. The input files are just python files, so any modifications to these can call other python modules/libraries, or create arrays of values algorithmically. These files should be placed in the folders listed, and the names of the specific files will be added as input options when running the program.

    -m material info file
    -p particle physics model info file
    -n numerical integration parameters file
    
Other constants can be tweaked by adjusting their value in `src/constants.py`.

### Material info : `inputs/material/(material name)/`

Example : `inputs/material/LiF/LiF_example.py`

Specify the lattice degrees of freedom in the mat_properties_dict dictionary:  

- mat_properties_dict["N_list"][particle_id] : total particle number, N
- mat_properties_dict["S_list"][particle_id] : total spin, S
- mat_properties_dict["L_list"][particle_id] : total angular momentum, L
- mat_properties_dict["L_S_list"][particle_id] : total spin-orbit coupling scalar component, L.S
- mat_properties_dict["L_tens_S_list"][particle_id] : total spin-orbit coupling, traceless symmetric tensor component, (LxS)___.

for each ion and each SM particle (electron (particle_id = "e"), proton (particle_id = "p"), neutron (particle_id = "n")) in the primitive cell, defined more rigorously in the corresponding paper. Also specify

- material : material name
- num_atoms : number of atoms in the primitive cell
    
### Particle physics model : `inputs/physics_model`

Example : `inputs/physics_model/dark_photon_example.py`

Specify the particle physics model and dark matter properties.

- input_masses : list of the DM masses to compute for 
    - Units : eV
- physics_parameters['threshold'] : energy threshold (eV), ignore all processes with energy less than this
- physics_parameters['times'] : list of the time of days to compute for, useful for looking for daily modulation signals
- physics_parameters['Fmed_power'] : - d log F_med / d log q, mediator form factors, for a light mediator this is -2, for a heavy mediator it's 0
- physics_parameters['power_V'] : d log V / d log q : the power of momentum in the scattering potential, used to optimally integrate over momentum space
- dm_properties_dict['spin'] : DM spin

- c_dict : List of bare, constant c coefficients. The general scattering potential will be a linear combination of O_1 -> O_11, sum_(i, psi) c_(i, psi) O_(i,psi). Setting values in c_dict specifies which terms should be included in the scattering potential.
    
- c_dict_form : Change this function to specify momentum/DM mass dependence in the c coefficients above.
- include_screen : specify whether screening effects are included

### Numerical integration parameters : `inputs/numerics/` 

Example : `inputs/numerics/standard.py`

Specify the parameters for the numeric integration over momentum space, along with some io_parameters

- io_parameters['output_folder'] : specify the output folder
- io_parameters['output_filename_extra'] : by default the output file name will be the material name and the material, physics model and numeric file names appended. To add more specification simply fill this value with a string.

- numerics_parameters['n_a']: number of grid points in radial direction
- numerics_parameters['n_b']: number of grid points in theta direction
- numerics_parameters['n_c']: number of grid points in phi direction
- numerics_parameters['power_a']: sampling parameter in the radial direction
- numerics_parameters['power_b']: sampling parameter in the theta direction
- numerics_parameters['power_c']: sampling parameter in the phi direction
- numerics_parameters['n_DW_x']: number of grid points in x direction to compute the DW factor
- numerics_parameters['n_DW_y']: number of grid points in x direction to compute the DW factor
- numerics_parameters['n_DW_z']: number of grid points in x direction to compute the DW factor

- numerics_parameters['q_cut']: option to automatically cut off the q integral when q is much greater than the Debye Waller factor

- numerics_parameters['special_mesh']: option to use a predefined special mesh which is optimal when the scattering potential is a power in q

# How to use

## Quick start

    > python calculator.py -m (material info file) -p (physics model info file) -n (numerics info file)

## Parallel

MPI parallelization is used to compute the scattering rates in parallel over the mass points in input_masses, and times in physics_parameters['times']. The optimal number of processors to compute with is (number of masses) x (number of times).

    > mpirun -np (number of processors) python calculator.py -m (material info file) -p (physics model info file) -n (numerics info file)
    
add the options `--use-hwthread-cpus` to enable hyperthreading.

For running on a cluster do not forget to load the relevant `openmpi` and `hdf5` modules.

# Output

By default, output data is stored in the `data/` folder. This works nicely when running the program locally, however for running on a cluster this will most likely have to be changed. To change the output folder simply change the value of `io_parameters['output_folder']` in your numerics input file (an example can be found in `inputs/numerics/standard.py`)

The output data is stored in HDF5 (hierarchical data format) files. The data is saved in a hierarchical structure with self-describing entry names. To show how easy it is to import this data we provide examples of reading the rate in `python` and `Mathematica`, from the example file `data/LiF_dark_photon_example_standard.hdf5`

## Python

    import h5py
    
    hdf5_data = h5py.File(<path to dm-phonon-scatter>'data/LiF_dark_photon_example_standard.hdf5', 'r')
    
    rate = hdf5_data['data']['rate']['0']['0']
    
    hdf5_data.close()
    
where the third and fourth indicies, both 0 above, are the time of day and mass index respectively.

## Mathematica

    data = Import[<path to dm-phonon-scatter>'data/LiF_dark_photon_example_standard.hdf5', "Data"]
    
    rate = data['/data/rate/0/0']

# References

Some papers which used a previous version of this program:

- T. Trickle, Z. Zhang, K. M. Zurek, K. Inzani, and S. Griffin, JHEP 03, 036 (2020), arXiv:1910.08092 [hep-ph]

- S. M. Griffin, K. Inzani, T. Trickle, Z. Zhang, and K. M. Zurek, Phys. Rev. D 101, 055004 (2020), arXiv:1910.10716 [hep-ph]

# Citation

If you use this program for your work please cite this paper:

[Effective Field Theory of Dark Matter Direct Detection With Collective Excitations](https://arxiv.org/abs/2009.13534)

with the citation:

    @article{Trickle:2020oki,
    author = "Trickle, Tanner and Zhang, Zhengkang and Zurek, Kathryn M.",
    title = "{Effective Field Theory of Dark Matter Direct Detection With Collective Excitations}",
    eprint = "2009.13534",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "CALT-TH-2020-037",
    month = "9",
    year = "2020"
    }
    
Thanks!
