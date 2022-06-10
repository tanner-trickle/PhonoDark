"""
    Using the BORN effective charges.

    Natural (eV) units are used throughout unless otherwise specified
"""

import src.constants as const
import numpy as np

#################

# Create the list of masses to compute for
# input_masses = [10**6, 10**7, 10**8, 10**9, 10**10]

input_masses = []

log_start_mass = 3
log_end_mass = 7

n_masses = 80

for i in range(n_masses):
    log_m = log_start_mass + (log_end_mass - log_start_mass)*(i/(n_masses - 1))
    input_masses.append(10**log_m)

input_masses = list(dict.fromkeys(input_masses))
input_masses.sort()

##################

"""
    TODO: automatically set power_V variable from list of c coefficients specified
"""
physics_parameters = {
        # energy threshold
        'threshold' : 10**(-3),
        # time of days (hr)
        'times'     : [0.],
        # - d log FDM / d log q. q dependence of mediator propagator
        'Fmed_power': 2.,
        # power of q in the potential, used to find optimal integration mesh
        'power_V'   : 2.,
        # flag to compute for a specific model
        # SI computes using the algorithm presented in 1910.08092
        #'special_model': 'SI',
        # include Born - bare charge correction piece
        'born_cor'  : True
}

dm_properties_dict = {
    'spin': 0.5,
    'mass_list': input_masses,
}

"""
    Dictionary containing tree level c coefficients, which determines which operators 
    contribute to a scattering process. Numbering follows the convention in 
    the paper. 

    To include different oeprators simply change the value of c_dict.

    Note: This should only contain constant values. If you want to include 
    q/m_chi dependence add it to c_dict_form below
"""

c_dict = {
	"1": {
            "e": 0.25,
            "p": -0.25,
            "n": 0,
            "screened": True
	},
#	"4": {
#            "e": 1,
#            "p": -2.8,
#            "n": 0
#	},
	"5a": {
            "e": 1,
            "p": -1,
            "n": 0,
            "screened": True
	},
#	"5b": {
#            "e": 1,
#            "p": -1,
#            "n": 0
#	},
#	"6": {
#            "e": -1,
#            "p": 2.8,
#            "n": 0
#	}
}


def c_dict_form(op_id, particle_id, q_vec, mass, spin):
    """
        q/m_chi dependence of the c coefficients. 

        Input:
            op_id : integer, operator id number
            particle_id : string, {"e", "p", "n"} for electron, proton, neutron resp.

            q_vec : (real, real, real), momentum vector in XYZ coordinates
            mass : dark matter mass
            spin : dark matter spin

        Output:
            real, the q/m_chi dependence of the c coefficients that isn't stored above in 
            c_dict


        Note: To add different operators simply add more functions inside of here, and replace
            one_func in the output dict
    """
    def one_func(q_vec, mass, spin):
        return 1.0
    
    def q_sq_on_mchi_sq(q_vec, mass, spin):
        return np.dot(q_vec,q_vec)/mass**2
        
    def q_sq_on_mchi_me(q_vec, mass, spin):
        return np.dot(q_vec,q_vec)/(mass*const.M_ELEC)
        
    def q_sq_on_mchi_mp(q_vec, mass, spin):
        return np.dot(q_vec,q_vec)/(mass*const.M_NUCL)
        
    def me_on_mchi(q_vec, mass, spin):
        return const.M_ELEC/mass

    def mp_on_mchi(q_vec, mass, spin):
        return const.M_NUCL/mass

    return {
            "1": {
                "e": q_sq_on_mchi_sq,
                "p": q_sq_on_mchi_sq,
                "n": one_func
            },
#            "4": {
#                "e": q_sq_on_mchi_me,
#                "p": q_sq_on_mchi_mp,
#                "n": one_func
#            },
            "5a": {
                "e": me_on_mchi,
                "p": mp_on_mchi,
                "n": one_func
            },
#            "5b": {
#                "e": me_on_mchi,
#                "p": mp_on_mchi,
#                "n": one_func
#            },
#            "6": {
#                "e": me_on_mchi,
#                "p": mp_on_mchi,
#                "n": one_func
#            }
        }[op_id][particle_id](q_vec, mass, spin)
