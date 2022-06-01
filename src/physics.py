"""
    Collection of functions which compute most relevant physics quantities
"""

from scipy import special
import numpy as np
import math

import src.phonopy_funcs as phonopy_funcs
import src.mesh as mesh
import src.constants as const
import src.potential_operators as potential_operators
import src.vel_g_function_integrals as vel_g_function_integrals

def mu(m1,m2):
    """
        Reduced mass
    """
    return (m1*m2)/(m1+m2)

def create_vE_vec(t):
    """
            Returns vE_vec for time t given in hours
    """
    phi = 2*const.PI*(t/24.0)

    vEx = const.VE*np.sin(const.THETA_E)*np.sin(phi)
    vEy = const.VE*np.cos(const.THETA_E)*np.sin(const.THETA_E)*(np.cos(phi) - 1)
    vEz = const.VE*((np.sin(const.THETA_E)**2)*np.cos(phi) + np.cos(const.THETA_E)**2)

    return np.array([vEx, vEy, vEz])

def calculate_W_tensor(
                        phonon_file,
                        num_atoms, 
                        atom_masses,
                        n_k_1,
                        n_k_2,
                        n_k_3,
                        q_red_to_XYZ
                ):
    """
        Calculate the W tensor, which is related to the Debye-Waller factor by q.W.q = DW_factor, based
        on a Monkhort-Pack mesh with total number of k points < n_k_1*n_k_2*n_k_3

        Returns: W_tens(j, a, b)
    """

    n_k_tot = 0
    k_list = []

    for i in range(n_k_1):
        for j in range(n_k_2):
            for k in range(n_k_3):

                q_red_vec = []

                q_red_vec.append((2.0*i - n_k_1 + 1.0)/n_k_1)
                q_red_vec.append((2.0*j - n_k_2 + 1.0)/n_k_2)
                q_red_vec.append((2.0*k - n_k_3 + 1.0)/n_k_3)

                [k_vec, G_vec] = mesh.get_kG_from_q_red(q_red_vec, q_red_to_XYZ)

                if G_vec[0] == 0.0 and G_vec[1] == 0.0 and G_vec[2] == 0.0:
                    n_k_tot += 1

                    k_list.append(k_vec)

    [eigenvectors, omega] = phonopy_funcs.run_phonopy(phonon_file, k_list)

    W_tensor = np.zeros((num_atoms, 3, 3), dtype=complex)

    for j in range(num_atoms):
        for k in range(n_k_tot):
            for nu in range(3*num_atoms):
                for a in range(3):
                    for b in range(3):

                        W_tensor[j][a][b] += (4.0*atom_masses[j]*n_k_tot*omega[k, nu])**(-1)*\
                        eigenvectors[k, nu, j, a]*\
                        np.conj(eigenvectors[k, nu, j, b])

    return np.array(W_tensor)

def c_dict_form_full(q_vec, dielectric, c_dict, c_dict_form, mass, spin):
    """

        Takes the bare c_dict_form function and returns the in-medium form function

        fp = fp_bare + ( 1 - q^2/(q.e.q) )*fe_bare

        fn = fn_bare

        fe = fe_bare*q^2/(q.e.q)

    """

    c_dict_in_med = {}

    screen_val = np.dot(q_vec, q_vec)/np.dot(q_vec, np.matmul(dielectric, q_vec))

    for op_id in c_dict:

        c_dict_in_med[op_id] = {}
 
        fe_bare = c_dict[op_id]['e']
        fn_bare = c_dict[op_id]['n']
        fp_bare = c_dict[op_id]['p']
        
        if c_dict[op_id].get('screened'):
        
            c_dict_in_med[op_id]['e'] = (screen_val*fe_bare)*c_dict_form(op_id, 'e', q_vec, mass, spin)
            c_dict_in_med[op_id]['n'] = (fn_bare)*c_dict_form(op_id, 'n', q_vec, mass, spin)
            c_dict_in_med[op_id]['p'] = (fp_bare + ( 1.0 - screen_val )*fe_bare)*c_dict_form(op_id, 'p', q_vec, mass, spin)
            
        else:
            c_dict_in_med[op_id]['e'] = fe_bare*c_dict_form(op_id, 'e', q_vec, mass, spin)
            c_dict_in_med[op_id]['n'] = fn_bare*c_dict_form(op_id, 'n', q_vec, mass, spin)
            c_dict_in_med[op_id]['p'] = fp_bare*c_dict_form(op_id, 'p', q_vec, mass, spin)
			
    return c_dict_in_med

def calc_diff_rates_general(mass, q_XYZ_list, G_XYZ_list, jacob_list, physics_parameters,
                    vE_vec, numerics_parameters, phonopy_params,ph_omega, ph_eigenvectors,
                    W_tensor, c_dict, mat_properties_dict, dm_properties_dict, 
                    c_dict_form, phonon_file):
    """
        Computes the differential rate
    """

    n_q = len(q_XYZ_list)

    n_a = numerics_parameters['n_a']
    n_b = numerics_parameters['n_b']
    n_c = numerics_parameters['n_c']
    energy_bin_width = numerics_parameters['energy_bin_width']
    
    Fmed_power    = physics_parameters['Fmed_power']
    threshold     = physics_parameters['threshold']
    born_cor      = physics_parameters.get('born_cor')

    m_cell = sum(phonopy_params['atom_masses'])

    spin = dm_properties_dict["spin"]
    
    [ph_eigenvectors_delta_E, ph_omega_delta_E] = phonopy_funcs.run_phonopy(phonon_file, 
                [[0., 0., 0.]])
    
    max_delta_E = 4*np.amax(ph_omega_delta_E)

    max_bin_num = math.floor((max_delta_E-threshold)/energy_bin_width) + 1

    diff_rate = np.zeros(max_bin_num, dtype=complex)
    binned_rate = np.zeros(phonopy_params['num_modes'], dtype=complex)
    
    for q in range(n_q):
    
        q_vec = q_XYZ_list[q]
        q_mag = np.linalg.norm(q_vec)
        
        # F_prop_val holds propagator dependence of rate
        F_prop_val = (1.0/q_mag)**Fmed_power
        
        c_dict_full = c_dict_form_full(q_vec, phonopy_params['dielectric'], c_dict, c_dict_form, 
                mass, spin)

        non_zero_indices = potential_operators.get_non_zero_c_indicies(c_dict_full)
        
        total_V_func_eval = potential_operators.total_V_func(non_zero_indices, 
                                q_vec, phonopy_params['num_atoms'], mat_properties_dict, 
                                mass, spin, c_dict_full)  

        if born_cor:
        
            cNe_func_eval = potential_operators.cNe_func(non_zero_indices,
                                    q_vec, phonopy_params['num_atoms'], mat_properties_dict,
                                    mass, spin, c_dict_full)
                                    
            cNe00 = cNe_func_eval["00"]
            cNe01 = cNe_func_eval["01"]
            cNe11 = cNe_func_eval["11"]
        
        for nu in range(phonopy_params['num_modes']):

            energy_diff = ph_omega[q][nu]
           
            if energy_diff >= threshold:
            # g function value
                v_star_val = vel_g_function_integrals.v_star_func(q_vec, energy_diff, mass, vE_vec)
                v_minus_val = np.abs(v_star_val)

                if v_minus_val < const.VESC:

                    g0_val = vel_g_function_integrals.g0_func_opt(q_vec, energy_diff, mass,
                                                vE_vec, v_minus_val)
                    g1_vec = vel_g_function_integrals.g1_func_opt(q_vec, energy_diff, mass, vE_vec,
                                                g0_val, v_star_val)
                    g2_tens = vel_g_function_integrals.g2_func_opt(q_vec, energy_diff, mass, vE_vec,
                                                v_minus_val, g0_val, g1_vec)
     
                    bin_num = math.floor((energy_diff-threshold)/energy_bin_width)
                    
                    dw_val = np.zeros(phonopy_params['num_atoms'], dtype=complex)
                    pos_phase = np.zeros(phonopy_params['num_atoms'], dtype=complex)
                    
                    q_dot_e = np.zeros(phonopy_params['num_atoms'], dtype=complex)
                    q_dot_ZminusQ_dot_e = np.zeros(phonopy_params['num_atoms'], dtype=complex)
                    
                    prefactor = 0.5*(const.RHO_DM/mass)\
                                *(1.0/m_cell)*(2*const.PI)**(-3)*jacob_list[q]\
                                *(1.0/(n_a*n_b*n_c))*(1.0/energy_diff)\
                                *F_prop_val**2

                    
                    for j in range(phonopy_params['num_atoms']):
                        
                        dw_val[j] = np.dot(q_vec, np.matmul(W_tensor[j], q_vec))
                        pos_phase[j] = (1j)*np.dot(G_XYZ_list[q], phonopy_params['eq_positions_XYZ'][j])

                        q_dot_e[j] = np.dot(q_vec, ph_eigenvectors[q][nu][j])
                        
                        if born_cor:
                            q_dot_ZminusQ_dot_e[j] = np.matmul(q_vec, np.matmul(phonopy_params['born'][j], ph_eigenvectors[q][nu][j]))\
                                                        - (mat_properties_dict['N_list']['p'][j]-mat_properties_dict['N_list']['e'][j])*q_dot_e[j]
                        
                    for j in range(phonopy_params['num_atoms']):
                        
                        dw_val_j = dw_val[j]
                        pos_phase_j = pos_phase[j]
                        
                        q_dot_e_star_j = np.conj(q_dot_e[j])
                        
                        if born_cor:
                            q_dot_ZminusQ_dot_e_star_j = np.conj(q_dot_ZminusQ_dot_e[j])
                        
                        V00_j = total_V_func_eval["00"][j]
                        V01_j = total_V_func_eval["01"][j]
                        V10_j = total_V_func_eval["10"][j]
                        V11_j = total_V_func_eval["11"][j]
                        
                        for jp in range(phonopy_params['num_atoms']):

                            dw_val_jp = dw_val[jp]
                            pos_phase_jp = pos_phase[jp]

                            q_dot_e_jp = q_dot_e[jp]
                            
                            if born_cor:
                                q_dot_ZminusQ_dot_e_jp = q_dot_ZminusQ_dot_e[jp]

                            V00_jp = total_V_func_eval["00"][jp]
                            V01_jp = total_V_func_eval["01"][jp]
                            V10_jp = total_V_func_eval["10"][jp]
                            V11_jp = total_V_func_eval["11"][jp]

                            g0_rate = g0_val*(
                                V00_j*np.conj(V00_jp)
                                + ((spin*(spin+1))/3.0)*(
                                        np.dot(V01_j, np.conj(V01_jp))
                                    )
                                )

                            g1_rate = np.dot(g1_vec, 
                                    (
                                        V00_j*np.conj(V10_jp) + np.conj(V00_jp)*V10_j
                                        + ((spin*(spin+1))/3.0)*(
                                             np.matmul(np.conj(V11_jp), V01_j) + 
                                             np.matmul(V11_j, np.conj(V01_jp))
                                            )
                                    )
                                )

                            g2_rate = (
                                np.dot(V10_j, np.matmul(g2_tens, np.conj(V10_jp)))
                                + ((spin*(spin+1))/3.0)*np.trace(
                                        np.matmul(
                                                V11_j.T, np.matmul(g2_tens, np.conj(V11_jp))
                                            )
                                    )
                                )
                           
                            exp_val = - dw_val_j - dw_val_jp + pos_phase_j - pos_phase_jp

                            delta_rate = (q_dot_e_star_j*q_dot_e_jp)*(g0_rate + g1_rate + g2_rate)
                            
                            if born_cor:
                            
                                g0_rate = g0_val*(
                                    cNe00*np.conj(V00_jp)
                                    + ((spin*(spin+1))/3.0)*(
                                            np.dot(cNe01, np.conj(V01_jp))
                                        )
                                    )
                                
                                g1_rate = np.dot(g1_vec,
                                        (
                                            cNe00*np.conj(V10_jp)
                                            + ((spin*(spin+1))/3.0)*(
                                                np.matmul(np.conj(V11_jp), cNe01) +
                                                np.matmul(cNe11, np.conj(V01_jp))
                                                )
                                        )
                                    )

                                g2_rate = (
                                    ((spin*(spin+1))/3.0)*np.trace(
                                            np.matmul(
                                                    cNe11.T, np.matmul(g2_tens, np.conj(V11_jp))
                                                )
                                        )
                                    )
                                
                                delta_rate -= (q_dot_ZminusQ_dot_e_star_j*q_dot_e_jp)*(g0_rate + g1_rate + g2_rate)
                                
                                
                                g0_rate = g0_val*(
                                    V00_j*np.conj(cNe00)
                                    + ((spin*(spin+1))/3.0)*(
                                            np.dot(V01_j, np.conj(cNe01))
                                        )
                                    )

                                g1_rate = np.dot(g1_vec,
                                        (
                                            np.conj(cNe00)*V10_j
                                            + ((spin*(spin+1))/3.0)*(
                                                np.matmul(np.conj(cNe11), V01_j) +
                                                np.matmul(V11_j, np.conj(cNe01))
                                                )
                                        )
                                    )

                                g2_rate = (
                                    ((spin*(spin+1))/3.0)*np.trace(
                                            np.matmul(
                                                    V11_j.T, np.matmul(g2_tens, np.conj(cNe11))
                                                )
                                        )
                                    )

                                delta_rate -= (q_dot_e_star_j*q_dot_ZminusQ_dot_e_jp)*(g0_rate + g1_rate + g2_rate)
                                
                                
                                g0_rate = g0_val*(
                                    cNe00*np.conj(cNe00)
                                    + ((spin*(spin+1))/3.0)*(
                                            np.dot(cNe01, np.conj(cNe01))
                                        )
                                    )

                                g1_rate = np.dot(g1_vec,
                                        (
                                            ((spin*(spin+1))/3.0)*(
                                                np.matmul(np.conj(cNe11), cNe01) +
                                                np.matmul(cNe11, np.conj(cNe01))
                                                )
                                        )
                                    )

                                g2_rate = (
                                    ((spin*(spin+1))/3.0)*np.trace(
                                            np.matmul(
                                                    cNe11.T, np.matmul(g2_tens, np.conj(cNe11))
                                                )
                                        )
                                    )

                                delta_rate += (q_dot_ZminusQ_dot_e_star_j*q_dot_ZminusQ_dot_e_jp)*(g0_rate + g1_rate + g2_rate)

                            delta_rate *= prefactor*(phonopy_params['atom_masses'][j]*phonopy_params['atom_masses'][jp])**(-0.5)*np.exp(exp_val)
                            
                            binned_rate[nu] += delta_rate
                            diff_rate[bin_num] += delta_rate

    total_rate = sum(diff_rate)

    return [diff_rate, binned_rate, total_rate]

def calc_diff_rates_SI(mass, q_XYZ_list, G_XYZ_list, jacob_list, physics_parameters,
                    vE_vec, numerics_parameters, phonopy_params, ph_omega, ph_eigenvectors,
                    W_tensor, mat_properties_dict, dm_properties_dict, 
                    phonon_file, c_dict):
    """
        Computes the differential rate specifically for SI models, taking in to account the BORN effective
        charges when necessary.

        Coupling parameterization is the same as in arXiv:1910.08092.

        Born effective charges from the BORN file are used.
    """

    n_q = len(q_XYZ_list)

    n_a = numerics_parameters['n_a']
    n_b = numerics_parameters['n_b']
    n_c = numerics_parameters['n_c']
    energy_bin_width = numerics_parameters['energy_bin_width']
    
    Fmed_power    = physics_parameters['Fmed_power']
    threshold     = physics_parameters['threshold']

    m_cell = sum(phonopy_params['atom_masses'])
    
    [ph_eigenvectors_delta_E, ph_omega_delta_E] = phonopy_funcs.run_phonopy(phonon_file, 
                [[0., 0., 0.]])
    
    max_delta_E = 4*np.amax(ph_omega_delta_E)

    max_bin_num = math.floor((max_delta_E-threshold)/energy_bin_width) + 1

    diff_rate = np.zeros(max_bin_num, dtype=complex)
    binned_rate = np.zeros(phonopy_params['num_modes'], dtype=complex)

    fe0 = c_dict[1]['e']
    fn0 = c_dict[1]['n']
    fp0 = c_dict[1]['p']
    
    for q in range(n_q):
    
        q_vec = q_XYZ_list[q]
        q_mag = np.linalg.norm(q_vec)

        q_hat = q_vec/q_mag
        
        # F_prop_val holds propagator dependence of rate
        F_prop_val = (1.0/q_mag)**Fmed_power

        # NOTE (TT, 10/5/21): scalar mediators should be screened, as opposed to what is in Eq. 67 of 1910.08092.
        screen_val = 1.0/np.dot(q_hat, np.matmul(phonopy_params['dielectric'], q_hat))

        # Eq 50 in 1910.08092
        fe = screen_val*fe0
        fp = fp0 + (1.0 - screen_val)*fe0 
        fn = fn0 

        for nu in range(phonopy_params['num_modes']):

            energy_diff = ph_omega[q][nu]
           
            if energy_diff >= threshold:
            # g function value
                v_star_val = vel_g_function_integrals.v_star_func(q_vec, energy_diff, mass, vE_vec)
                v_minus_val = np.abs(v_star_val)

                g0_val = vel_g_function_integrals.g0_func_opt(q_vec, energy_diff, mass, 
                                            vE_vec, v_minus_val)
     
                bin_num = math.floor((energy_diff-threshold)/energy_bin_width)
                
                if v_minus_val < const.VESC:

                    S_nu = 0.0

                    for j in range(phonopy_params['num_atoms']):

                        # Eq. 104
                        A_j = phonopy_params['A_list'][j]
                        Z_j = phonopy_params['Z_list'][j]

                        Y_j = -fe*np.matmul(phonopy_params['born'][j], q_vec) \
                                + fe*Z_j*q_vec \
                                + fn*(A_j - Z_j)*q_vec \
                                + fp*Z_j*q_vec

                        Y_dot_e_star = np.dot(Y_j, np.conj(ph_eigenvectors[q][nu][j]))
                        
                        dw_val_j = np.dot(q_vec, np.matmul(W_tensor[j], q_vec))
                        
                        pos_phase_j = (1j)*np.dot(G_XYZ_list[q], phonopy_params['eq_positions_XYZ'][j])

                        S_nu += (phonopy_params['atom_masses'][j])**(-0.5)*\
                                np.exp(-dw_val_j + pos_phase_j)*Y_dot_e_star

                    delta_rate = (
                        0.5*(const.RHO_DM/mass)\
                        *(1.0/m_cell)*(2*const.PI)**(-3)*jacob_list[q]\
                        *(1.0/(n_a*n_b*n_c))*(1.0/energy_diff)\
                        *S_nu*np.conj(S_nu)\
                        *F_prop_val**2\
                        *g0_val
                        )

                    binned_rate[nu] += delta_rate
                    diff_rate[bin_num] += delta_rate

    total_rate = sum(diff_rate)

    return [diff_rate, binned_rate, total_rate]

def calc_diff_rates_SI_q(mass, q_XYZ_list, G_XYZ_list, jacob_list, physics_parameters,
                    vE_vec, numerics_parameters, phonopy_params,ph_omega, ph_eigenvectors,
                    W_tensor, mat_properties_dict, dm_properties_dict, 
                    phonon_file, max_bin_num, q_index, c_dict):
    """
        Computes the differential rate for a specific q point for the dark photon model.
    """

    n_a = numerics_parameters['n_a']
    n_b = numerics_parameters['n_b']
    n_c = numerics_parameters['n_c']
    energy_bin_width = numerics_parameters['energy_bin_width']
    
    Fmed_power    = physics_parameters['Fmed_power']
    threshold     = physics_parameters['threshold']

    m_cell = sum(phonopy_params['atom_masses'])

    diff_rate = np.zeros(max_bin_num, dtype=complex)
    binned_rate = np.zeros(phonopy_params['num_modes'], dtype=complex)

    q_vec = q_XYZ_list[q_index]
    q_mag = np.linalg.norm(q_vec)

    q_hat = q_vec/q_mag
    
    # F_prop_val holds propagator dependence of rate
    F_prop_val = (1.0/q_mag)**Fmed_power

    fe0 = c_dict[1]['e']
    fn0 = c_dict[1]['n']
    fp0 = c_dict[1]['p']

    # NOTE (TT, 10/5/21): scalar mediators should be screened, as opposed to what is in Eq. 67 of 1910.08092.
    screen_val = 1.0/np.dot(q_hat, np.matmul(phonopy_params['dielectric'], q_hat))

    # Eq 50 in 1910.08092
    fe = screen_val*fe0
    fp = fp0 + (1.0 - screen_val)*fe0 
    fn = fn0 

    for nu in range(phonopy_params['num_modes']):

        energy_diff = ph_omega[0][nu]
       
        if energy_diff >= threshold:
        # g function value
            v_star_val = vel_g_function_integrals.v_star_func(q_vec, energy_diff, mass, vE_vec)
            v_minus_val = np.abs(v_star_val)

            g0_val = vel_g_function_integrals.g0_func_opt(q_vec, energy_diff, mass, 
                                        vE_vec, v_minus_val)
 
            bin_num = math.floor((energy_diff-threshold)/energy_bin_width)
            
            if v_minus_val < const.VESC:

                S_nu = 0.0

                for j in range(phonopy_params['num_atoms']):

                    # Eq. 104
                    A_j = phonopy_params['A_list'][j]
                    Z_j = phonopy_params['Z_list'][j]

                    Y_j = -fe*np.matmul(phonopy_params['born'][j], q_vec) \
                            + fe*Z_j*q_vec \
                            + fn*(A_j - Z_j)*q_vec \
                            + fp*Z_j*q_vec

                    Y_dot_e_star = np.dot(Y_j, np.conj(ph_eigenvectors[0][nu][j]))

                    Y_j = np.matmul(phonopy_params['born'][j], q_vec)
                    
                    dw_val_j = np.dot(q_vec, np.matmul(W_tensor[j], q_vec))
                    
                    pos_phase_j = (1j)*np.dot(G_XYZ_list[q_index], phonopy_params['eq_positions_XYZ'][j])

                    S_nu += (phonopy_params['atom_masses'][j])**(-0.5)*\
                            np.exp(-dw_val_j + pos_phase_j)*Y_dot_e_star

                delta_rate = (
                    0.5*(const.RHO_DM/mass)\
                    *(1.0/m_cell)*(2*const.PI)**(-3)*jacob_list[q_index]\
                    *(1.0/(n_a*n_b*n_c))*(1.0/energy_diff)\
                    *S_nu*np.conj(S_nu)\
                    *F_prop_val**2\
                    *g0_val
                    )

                binned_rate[nu] += delta_rate
                diff_rate[bin_num] += delta_rate

    total_rate = sum(diff_rate)

    return [diff_rate, binned_rate, total_rate]

def calc_diff_rates_general_q(mass, q_XYZ_list, G_XYZ_list, jacob_list, physics_parameters,
                    vE_vec, numerics_parameters, phonopy_params,ph_omega, ph_eigenvectors,
                    W_tensor, c_dict, mat_properties_dict, dm_properties_dict, 
                    c_dict_form, phonon_file, max_bin_num, q_index):
    """
        Computes the differential rate for a specific q point
    """

    n_a = numerics_parameters['n_a']
    n_b = numerics_parameters['n_b']
    n_c = numerics_parameters['n_c']
    energy_bin_width = numerics_parameters['energy_bin_width']
    
    Fmed_power    = physics_parameters['Fmed_power']
    threshold     = physics_parameters['threshold']

    m_cell = sum(phonopy_params['atom_masses'])

    spin = dm_properties_dict["spin"]

    q_vec = q_XYZ_list[q_index]
    q_mag = np.linalg.norm(q_vec)

    diff_rate = np.zeros(max_bin_num, dtype=complex)
    binned_rate = np.zeros(phonopy_params['num_modes'], dtype=complex)
    
    # F_prop_val holds propagator dependence of rate
    F_prop_val = (1.0/q_mag)**Fmed_power
    
    c_dict_full = c_dict_form_full(q_vec, phonopy_params['dielectric'], c_dict, c_dict_form, 
            mass, spin)

    non_zero_indices = potential_operators.get_non_zero_c_indicies(c_dict_full)
    
    total_V_func_eval = potential_operators.total_V_func(non_zero_indices, 
                            q_vec, phonopy_params['num_atoms'], mat_properties_dict, 
                            mass, spin, c_dict_full)  

    for nu in range(phonopy_params['num_modes']):

        energy_diff = ph_omega[0][nu]
       
        if energy_diff >= threshold:
        # g function value
            v_star_val = vel_g_function_integrals.v_star_func(q_vec, energy_diff, mass, vE_vec)
            v_minus_val = np.abs(v_star_val)

            g0_val = vel_g_function_integrals.g0_func_opt(q_vec, energy_diff, mass, 
                                        vE_vec, v_minus_val)
            g1_vec = vel_g_function_integrals.g1_func_opt(q_vec, energy_diff, mass, vE_vec,
                                        g0_val, v_star_val)
            g2_tens = vel_g_function_integrals.g2_func_opt(q_vec, energy_diff, mass, vE_vec,
                                        v_minus_val, g0_val, g1_vec)
 
            bin_num = math.floor((energy_diff-threshold)/energy_bin_width)
            
            if v_minus_val < const.VESC:

                for j in range(phonopy_params['num_atoms']):
                    
                    dw_val_j = np.dot(q_vec, np.matmul(W_tensor[j], q_vec))
                    
                    pos_phase_j = (1j)*np.dot(G_XYZ_list[q_index], phonopy_params['eq_positions_XYZ'][j])

                    q_dot_e_star_j = np.dot(q_vec, np.conj(ph_eigenvectors[0][nu][j]))
                    V00_j = total_V_func_eval["00"][j]
                    V01_j = total_V_func_eval["01"][j]
                    V10_j = total_V_func_eval["10"][j]
                    V11_j = total_V_func_eval["11"][j]
                    
                    for jp in range(phonopy_params['num_atoms']):

                        dw_val_jp = np.dot(q_vec, np.matmul(W_tensor[jp], q_vec))
                        pos_phase_jp = (1j)*np.dot(G_XYZ_list[q_index], phonopy_params['eq_positions_XYZ'][jp])

                        q_dot_e_jp = np.dot(q_vec, ph_eigenvectors[0][nu][jp])

                        V00_jp = total_V_func_eval["00"][jp]
                        V01_jp = total_V_func_eval["01"][jp]
                        V10_jp = total_V_func_eval["10"][jp]
                        V11_jp = total_V_func_eval["11"][jp]

                        g0_rate = g0_val*(
                            V00_j*np.conj(V00_jp)
                            + ((spin*(spin+1))/3.0)*(
                                    np.dot(V01_j, np.conj(V01_jp))
                                )
                            )

                        g1_rate = np.dot(g1_vec, 
                                (
                                    V00_j*np.conj(V10_jp) + np.conj(V00_jp)*V10_j
                                    + ((spin*(spin+1))/3.0)*(
                                         np.matmul(np.conj(V11_jp), V01_j) + 
                                         np.matmul(V11_j, np.conj(V01_jp))
                                        )
                                )
                            )

                        g2_rate = (
                            np.dot(V10_j, np.matmul(g2_tens, np.conj(V10_jp)))
                            + ((spin*(spin+1))/3.0)*np.trace(
                                    np.matmul(
                                            V11_j.T, np.matmul(g2_tens, np.conj(V11_jp))
                                        )
                                )
                            )
                       
                        exp_val = - dw_val_j - dw_val_jp + pos_phase_j - pos_phase_jp

                        delta_rate = (
                            0.5*(const.RHO_DM/mass)\
                            *(1.0/m_cell)*(2*const.PI)**(-3)*jacob_list[q_index]\
                            *(1.0/(n_a*n_b*n_c))*(1.0/energy_diff)\
                            *(phonopy_params['atom_masses'][j]*phonopy_params['atom_masses'][jp])**(-0.5)\
                            *np.exp(exp_val)\
                            *q_dot_e_star_j*q_dot_e_jp\
                            *F_prop_val**2\
                            *(g0_rate + g1_rate + g2_rate)
                            )

                        binned_rate[nu] += delta_rate
                        diff_rate[bin_num] += delta_rate

    total_rate = sum(diff_rate)

    return [diff_rate, binned_rate, total_rate]
