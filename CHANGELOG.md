# Changelog

## [1.2.0] - 6/30/2022

- Completed the set of EFT operators.
- Generalized the Earth velocity specification. These vectors can be specified by the time of the day, or directly (i.e. to compute annual modulation signals).
- Inputs are now handled via a new "input_handler" module which allows defaults for the variables to be set.
    - By default, the number of protons, neutrons, and electrons ("N_list") is set automatically from the input material configuration.
        - Note, these variables may still be specified if need be.
- Refactored the main program to follow a more logical flow (load inputs, configure calculation, perform calculation, save results).
- Greatly simplified parallelization scheme (only a single MPI reduce call is used now).
- Greatly simplified the code which writes the output data.
- **Output units match for all calculations**. The binned number of events and total number of events are computed assuming that \( g_\chi = g_\psi = 1 \). The output rate is the total number of events assuming a kg-yr exposure (**Note**: this is identical to the event rate per kg-yr).
- Mediator masses can now be supplied to the propagator. ' m_med = 'light' ' specifies the special case of a light mediator, and ' m_med = 'heavy' ' specifies the case of a heavy mediator (assumed to be a TeV). Otherwise the mediator mass will be whatever 'm_med' is set to. 
- **Bug Fix**: Corrected factor of 2 in anapole operator.

## [1.1.0] - 1/15/2021

- Adding program to parallelize over points in the momentum space integration, which makes the calculation tractable for targets with many (N > 50) atoms in the primitive cell.

## [1.0.1] - 11/3/2020

- Fixed bug in equilibrium positions, was not converting the units from phonopy output.
