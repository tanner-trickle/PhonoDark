"""
    
    Non-relativistic operators

    Notation also descibed in eft_numeric_formulation_notes

    V_j = V^0_j + v_i V^1_{j i}

    V^0_j = V^00_j + S_k V^01_{j k}
    V^1_{j i} = V^10_{j, i} + S_k V^11_{j i k}

    expansion id's (exp_id) = "00", "01", "10", "11"

"""

import numpy as np
from sympy import LeviCivita

import src.constants as const

#############

def V1_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    return mat_properties_dict["N_list"][particle_id]

def V3_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros(num_atoms, dtype=complex)

    q_dir = q_vec/np.linalg.norm(q_vec)

    overall_const = np.linalg.norm(q_vec)**2/mat_properties_dict["mass"][particle_id]**2

    for j in range(num_atoms):

            L_d_S = mat_properties_dict["L_S_list"][particle_id][j]
            L_t_S = mat_properties_dict["L_tens_S_list"][particle_id][j]

            val[j] = overall_const*(
                            (1.0/3.0)*L_d_S - (1.0/2.0)*np.dot(q_dir, np.matmul(L_t_S, q_dir))
                            )

    return val


def V3_10(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3), dtype=complex)

    overall_const = (1j/mat_properties_dict["mass"][particle_id])

    for j in range(num_atoms):

            S_val = mat_properties_dict["S_list"][particle_id][j]

            val[j] = overall_const*(
                            np.cross(S_val, q_vec)
                    )

    return val

def V5_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3), dtype=complex)

    overall_const = 0.5*np.dot(q_vec, q_vec)*(mat_properties_dict["mass"][particle_id])**(-2)

    q_dir = q_vec/np.linalg.norm(q_vec)

    for j in range(num_atoms):

            L_val = mat_properties_dict["L_list"][particle_id][j]

            val[j] = overall_const*(
                            L_val - q_dir*np.dot(q_dir, L_val)
                    )

    return val

def V5_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3, 3), dtype=complex)

    overall_const = (1j/mat_properties_dict["mass"][particle_id]) 

    for j in range(num_atoms):

        N_val = mat_properties_dict["N_list"][particle_id][j]

        for alpha in range(3):
            for beta in range(3):
                for chi in range(3):

                    val[j][alpha][beta] += overall_const*(
                                    N_val*LeviCivita(beta, chi, alpha)*q_vec[chi]
                            )

    return val 

def V6_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = np.dot(q_vec, q_vec)*(mat_properties_dict["mass"][particle_id])**(-2)

        q_dir = q_vec/np.linalg.norm(q_vec)

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                q_dir*np.dot(q_dir, S_val)
                        )

        return val 

def V7_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros(num_atoms, dtype=complex)

        overall_const = -(0.5)*(mass)**(-1.0)

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.dot(q_vec, S_val)
                        )

        return val 

def V7_10(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        return mat_properties_dict["S_list"][particle_id]


def V8_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = 0.5

        for j in range(num_atoms):

                N_val = mat_properties_dict["N_list"][particle_id][j]
                L_val = mat_properties_dict["L_list"][particle_id][j]

                val[j] = overall_const*(
                                -N_val*(q_vec/(2.0*mass)) + 
                                (1j/mat_properties_dict["mass"][particle_id])*np.cross(q_vec, L_val)
                        )

        return val

def V8_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3, 3), dtype=complex)

        overall_const = 1

        for j in range(num_atoms):

                N_val = mat_properties_dict["N_list"][particle_id][j]

                val[j] = overall_const*(
                                N_val*np.identity(3)
                        )

        return val

def V9_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = -(1j/mat_properties_dict["mass"][particle_id])

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.cross(q_vec, S_val)
                        )

        return val 

def V10_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros(num_atoms, dtype=complex)

        overall_const = (1j/mat_properties_dict["mass"][particle_id])

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.dot(q_vec, S_val)
                        )

        return val 

def V11_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = (1j/mat_properties_dict["mass"][particle_id])

        for j in range(num_atoms):

                N_val = mat_properties_dict["N_list"][particle_id][j]

                val[j] = overall_const*(
                                N_val*q_vec
                        )

        return val

def zeros_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    return np.zeros(num_atoms)

def zeros_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    return np.zeros((num_atoms, 3))

def zeros_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    return np.zeros((num_atoms, 3, 3))

def V_terms_func(q_vec, op_id, exp_id, particle_id, num_atoms, mat_properties_dict, 
					mass, spin):
	"""

	op_id : NR operator ID (integer)
	exp_id : expansion ID defined above (str)
	particle_id: type of particle (str)

        Fills a dictionary with functions, then calls the dictionary to the specific 
        function, then evaluates the function at the arguments.

        Avoids evaluating all of the functions every time to build the dictionary.

	"""

	return {
		1:{
			"00": V1_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": zeros_11
		},
		3:{
			"00": V3_00,
			"01": zeros_01,
			"10": V3_10,
			"11": zeros_11
		},
		4:{
			"00": zeros_00,
			"01": V7_10,
			"10": zeros_01,
			"11": zeros_11
		},
		5:{
			"00": zeros_00,
			"01": V5_01,
			"10": zeros_01,
			"11": V5_11
		},
		6:{
			"00": zeros_00,
			"01": V6_01,
            "10": zeros_01,
			"11": zeros_11
		},
		7:{
			"00": V7_00,
			"01": zeros_01,
			"10": V7_10,
			"11": zeros_11
		},
		8:{
			"00": zeros_00,
			"01": V8_01,
			"10": zeros_01,
			"11": V8_11
		},
		9:{
			"00": zeros_00,
			"01": V9_01,
			"10": zeros_01,
			"11": zeros_11
		},
		10:{
			"00": V10_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": zeros_11
		},
		11:{
			"00": zeros_00,
			"01": V11_01,
			"10": zeros_01,
			"11": zeros_11
		}
	}[op_id][exp_id](q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin)


def get_non_zero_c_indicies(c_dict):
	"""
	    Returns the indicies of the non-zero elements in the c_dict
	"""

	indices = []

	op_ids = list(c_dict.keys())

	for i in range(len(op_ids)):

            op_id = op_ids[i]

            particle_ids = list(c_dict[op_id].keys())

            for j in particle_ids:

                if c_dict[op_id][j] != 0:

                    indices.append([op_id, j])

	return indices 



def total_V_func(non_zero_indices, q_vec, num_atoms, mat_properties_dict, 
					mass, spin, c_dict_full):
	"""
            given c_dict, q, generate V_j's which are summed over the non-zero c's
	"""

	total_V_00 = np.zeros(num_atoms, dtype=complex)
	total_V_01 = np.zeros((num_atoms, 3), dtype=complex)
	total_V_10 = np.zeros((num_atoms, 3), dtype=complex)
	total_V_11 = np.zeros((num_atoms, 3, 3), dtype=complex)

	for i in range(len(non_zero_indices)):

            op_id = non_zero_indices[i][0]
            particle_id = non_zero_indices[i][1]

            total_V_00 += c_dict_full[op_id][particle_id]*\
                    V_terms_func(q_vec, op_id, "00", particle_id, num_atoms, 
                                    mat_properties_dict, mass, spin)
            total_V_01 += c_dict_full[op_id][particle_id]*\
                    V_terms_func(q_vec, op_id, "01", particle_id, num_atoms, 
                                    mat_properties_dict, mass, spin)
            total_V_10 += c_dict_full[op_id][particle_id]*\
                    V_terms_func(q_vec, op_id, "10", particle_id, num_atoms, 
                                    mat_properties_dict, mass, spin)
            total_V_11 += c_dict_full[op_id][particle_id]*\
                    V_terms_func(q_vec, op_id, "11", particle_id, num_atoms, 
                                    mat_properties_dict, mass, spin)

	return {
		"00": total_V_00, 
		"01": total_V_01,
		"10": total_V_10,
		"11": total_V_11
	}
