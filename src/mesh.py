"""
    
    Functions for computing the momentum integral and mapping a general q vector to the 1BZ.
    
    To sample q space the steps are:

        1) sample uniformly in a, b, c
        2) control the sampling density with y(a), beta(b), chi(c)
        3) convert y, beta, chi to q
        4) make sure the q point is kinematically allowed

    Note: red is shorthand for reduced coordinates, XYZ means the vector is in physical XYZ coordinates

"""

import numpy as np
import math

import src.constants as const
import src.phonopy_funcs as phonopy_funcs

def y_func(a, power_a, special_mesh_option, threshold, q_max, delta):

    if (special_mesh_option):
        q_min = threshold/(const.VE + const.VESC)
        if (q_max > q_min):
            q_ratio = q_max/q_min
            if delta == -4:
                y = (q_ratio)**(a - 1)
            else:
                y = ((1.0 - (q_ratio)**(-4.0 - delta))*a + (q_ratio)**(-4.0 - delta))**(1.0/(4.0 + delta))
        else:
            y = 0
    else:
        y = a**power_a
    return y

def dy_da_func(a, power_a, special_mesh_option, threshold, q_max, delta):

    if (special_mesh_option):
        q_min = threshold/(const.VE + const.VESC)
        if (q_max > q_min):
            q_ratio = q_max/q_min
            if delta == -4:
                dyda = np.log(q_ratio)*(q_ratio)**(a - 1)
            else:
                dyda = (4.0 + delta)**(-1)*\
                    ((1.0 - q_ratio**(-4.0 - delta))*a\
                    + q_ratio**(-4.0 - delta))**((1.0/(4.0 + delta)) - 1.0)*\
                    (1.0 - q_ratio**(-4.0 - delta))
        else:
            dyda = 0
    else:
        dyda = power_a*a**(power_a-1)
    return dyda

def beta_func(b, power_b):
    return b**power_b

def dbeta_db_func(b, power_b):
    return power_b*b**(power_b - 1)

def chi_func(c, power_c):
    return c**power_c

def dchi_dc_func(c, power_c):
    return power_c*c**(power_c - 1)

def qabc_to_XYZ(q_max, a, b, c, power_a, power_b, power_c, threshold, special_mesh_option, delta):
    """
        Converts (a, b, c) to q in radial coordinates then q in XYZ coordinates
    """

    # a, b, c -> radial coordinates
    q_mag = q_max*y_func(a, power_a, special_mesh_option, threshold, q_max, delta)
    q_theta = np.arccos(2*beta_func(b, power_b) - 1)
    q_phi = 2*const.PI*chi_func(c, power_c)

    # radial coordinates -> XYZ
    q_x = q_mag*np.sin(q_theta)*np.cos(q_phi)
    q_y = q_mag*np.sin(q_theta)*np.sin(q_phi)
    q_z = q_mag*np.cos(q_theta)

    return np.array([q_x, q_y, q_z])

def jacobian_func(q_max, a, b, c, power_a, power_b, power_c, threshold, special_mesh_option, delta):
    """
        The integration jacobian, d^3q -> (1/N_a)(1/N_b)(1/N_c) sum_(a, b, c) jacob
    """

    abs_dyda = np.abs(dy_da_func(a, power_a, special_mesh_option, threshold, q_max, delta))
    abs_dbetadb = np.abs(dbeta_db_func(b, power_b))
    abs_dchidc = np.abs(dchi_dc_func(c, power_c))

    y_val = y_func(a, power_a, special_mesh_option, threshold, q_max, delta)

    jacob = (4*const.PI)*(q_max**3)*(y_val**2)*abs_dyda*abs_dbetadb*abs_dchidc

    return jacob


def compute_q_cut(phonon_file, atom_masses):
    """
        Returns q = 10*sqrt(max(m)*max(omega))
    """

    [ph_eigenvectors, ph_omega] = phonopy_funcs.run_phonopy(
                                                            phonon_file,
                                                            np.array([[0, 0, 0]])
                                                            )

    q_cut = 10.0*np.sqrt(np.amax(atom_masses)*np.amax(ph_omega))

    return q_cut

def create_q_mesh(mass, threshold, vE_vec, numerics_parameters, phonon_file, atom_masses,
                    delta):
    """
    Retruns list of q_XYZ points and jacobian for points  which pass kinematic cut

    Output: [q_XYZ_list, jacob_list]
    """

    power_a             = numerics_parameters['power_a']
    power_b             = numerics_parameters['power_b']
    power_c             = numerics_parameters['power_c']
    n_a                 = numerics_parameters['n_a']
    n_b                 = numerics_parameters['n_b']
    n_c                 = numerics_parameters['n_c']
    q_cut_option        = numerics_parameters['q_cut']
    special_mesh_option = numerics_parameters['special_mesh']

    if q_cut_option:
        q_cut = compute_q_cut(phonon_file, atom_masses)
        q_max = min(2*mass*(const.VESC + const.VE), q_cut)
    else:
        q_max = 2*mass*(const.VESC + const.VE)

    # Create uniform mesh in a, b, c
    a_list = np.zeros(n_a)
    b_list = np.zeros(n_b)
    c_list = np.zeros(n_c)

    for i in range(n_a):
        a_list[i] = ((i+1) - 0.5)/n_a

    for i in range(n_b):
        b_list[i] = ((i+1) - 0.5)/n_b

    for i in range(n_c):
        c_list[i] = ((i+1) - 0.5)/n_c

    q_XYZ_list = []
    jacob_list = []

    for i in range(n_a):
        a_val = a_list[i]
        for j in range(n_b):
            b_val = b_list[j]
            for k in range(n_c):
                c_val = c_list[k]

                # convert uniform a, b, c to q in XYZ coordinates
                q_vec = qabc_to_XYZ(q_max, a_val, b_val, c_val, 
                                        power_a, power_b, power_c, 
                                        threshold, special_mesh_option, delta)

                q_mag = np.sqrt(np.dot(q_vec, q_vec))

                jacob = jacobian_func(q_max, a_val, b_val, c_val, power_a, power_b, 
                                        power_c, threshold, special_mesh_option, delta) 

                # make sure that q passes kinematic cut, q vectors which don't will always return 
                # 0 in the final integral
                if ( np.abs((q_mag)**(-1)*(np.dot(q_vec, vE_vec) + (q_mag**2)/(2*mass))) < const.VESC 
                     and q_mag > (threshold/(const.VE + const.VESC)) ):

                    q_XYZ_list.append(q_vec)
                    jacob_list.append(jacob)

    return [np.array(q_XYZ_list), np.array(jacob_list)]

def get_kG_from_q_red(q_red_vec, q_red_to_XYZ):
    """
        q_red_vec: q vector in reduced coordinates
        q_red_to_XYZ: matrix converting q in reduced coordinates to XYZ

        output: [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates 

    """
    set_of_closest_G_red = []

    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])])

    q_XYZ_vec = np.matmul(q_red_to_XYZ, q_red_vec)

    first = True

    for vec in set_of_closest_G_red:

        diff_vec = q_XYZ_vec - np.matmul(q_red_to_XYZ, vec)

        if first:
            min_dist_sq = np.dot(diff_vec, diff_vec)
            first = False

        if np.dot(diff_vec, diff_vec) <= min_dist_sq:
            min_vec = vec

    G_red_vec = min_vec

    k_red_vec = np.array(q_red_vec) - np.array(G_red_vec)

    return [k_red_vec, G_red_vec]


def get_kG_from_q_XYZ(q_XYZ_vec, q_red_to_XYZ):
    """
        q_XYZ_vec: q vector in XYZ coordinates
        q_red_to_XYZ: matrix converting q in reduced coordinates to XYZ

        output: [k_red_vec, G_red_vec]: the k and G vectors in reduced coordinates 
    """

    q_red_vec = np.dot(np.linalg.inv(q_red_to_XYZ), q_XYZ_vec)

    set_of_closest_G_red = []

    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.floor(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.floor(q_red_vec[1]), math.ceil(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.floor(q_red_vec[2])])
    set_of_closest_G_red.append([math.ceil(q_red_vec[0]), math.ceil(q_red_vec[1]), math.ceil(q_red_vec[2])])

    first = True

    for vec in set_of_closest_G_red:

        diff_vec = q_XYZ_vec - np.matmul(q_red_to_XYZ, vec)

        if first:
            min_dist_sq = np.dot(diff_vec, diff_vec)
            min_vec=vec
            first = False

        if np.dot(diff_vec, diff_vec) <= min_dist_sq:
            min_dist_sq = np.dot(diff_vec, diff_vec)
            min_vec = vec

    G_red_vec = min_vec

    k_red_vec = np.array(q_red_vec) - np.array(G_red_vec)

    return [k_red_vec, G_red_vec]

def generate_k_red_mesh_from_q_XYZ_mesh(q_XYZ_mesh, recip_red_to_XYZ):
    """
        Takes in a q mesh and returns the k mesh in reduced coordinates to be fed in to phonopy.
    """

    k_mesh = []

    for q in range(len(q_XYZ_mesh)):

        q_vec = q_XYZ_mesh[q]

        [k_vec, G_vec] = get_kG_from_q_XYZ(q_vec, recip_red_to_XYZ)

        k_mesh.append(k_vec)

    return np.array(k_mesh) 

def get_G_XYZ_list_from_q_XYZ_list(q_XYZ_list, recip_red_to_XYZ):
    """
        Returns the list of G vectors given a list of q vectors.
    """

    n_q = len(q_XYZ_list)

    G_XYZ_list = []

    for q in range(n_q):

        q_vec = q_XYZ_list[q]

        [k_vec, G_red_vec] = get_kG_from_q_XYZ(q_vec, recip_red_to_XYZ)

        G_XYZ_list.append(np.matmul(recip_red_to_XYZ, G_red_vec))

    return np.array(G_XYZ_list)
