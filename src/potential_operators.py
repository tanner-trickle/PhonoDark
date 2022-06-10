"""
    
    Non-relativistic operators

    Notation also descibed in eft_numeric_formulation_notes

    V_j = V^0_j + v_i V^1_{j i}

    V^0_j = V^00_j + S_k V^01_{j k}
    V^1_{j i} = V^10_{j, i} + S_k V^11_{j i k}

    expansion id's (exp_id) = "00", "01", "10", "11"

"""

import numpy as np
#from sympy import LeviCivita

import src.constants as const

#############

def V1_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    return mat_properties_dict["N_list"][particle_id]

def V3b_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros(num_atoms, dtype=complex)

    q_dir = q_vec/np.linalg.norm(q_vec)

    overall_const = -0.5*np.linalg.norm(q_vec)**2/mat_properties_dict["mass"][particle_id]**2

    for j in range(num_atoms):

            LxS_val = mat_properties_dict["LxS_list"][particle_id][j]

            val[j] = overall_const*(
                            np.trace(LxS_val)- np.dot(q_dir, np.matmul(LxS_val, q_dir))
                            )

    return val


def V3a_10(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3), dtype=complex)

    overall_const = (1j/mat_properties_dict["mass"][particle_id])

    for j in range(num_atoms):

            S_val = mat_properties_dict["S_list"][particle_id][j]

            val[j] = overall_const*(
                            np.cross(S_val, q_vec)
                    )

    return val


def V4_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        return mat_properties_dict["S_list"][particle_id]


def V5b_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3), dtype=complex)

    overall_const = -0.5*np.dot(q_vec, q_vec)*(mat_properties_dict["mass"][particle_id])**(-2)

    q_dir = q_vec/np.linalg.norm(q_vec)

    for j in range(num_atoms):

            L_val = mat_properties_dict["L_list"][particle_id][j]

            val[j] = overall_const*(
                            L_val - q_dir*np.dot(q_dir, L_val)
                    )

    return val

def V5a_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3, 3), dtype=complex)

    overall_const = (1j/mat_properties_dict["mass"][particle_id]) 

    for j in range(num_atoms):

        N_val = mat_properties_dict["N_list"][particle_id][j]
        
        val[j][0][1] = overall_const*N_val*q_vec[2]
        val[j][1][2] = overall_const*N_val*q_vec[0]
        val[j][2][0] = overall_const*N_val*q_vec[1]
        
        val[j][1][0] = -val[j][0][1]
        val[j][2][1] = -val[j][1][2]
        val[j][0][2] = -val[j][2][0]

#        for alpha in range(3):
#            for beta in range(3):
#                for chi in range(3):
#
#                    val[j][alpha][beta] += overall_const*(
#                                    N_val*LeviCivita(beta, chi, alpha)*q_vec[chi]
#                            )

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

def V7a_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros(num_atoms, dtype=complex)

        overall_const = -(0.5)*(mass)**(-1.0)

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.dot(q_vec, S_val)
                        )

        return val 

def V7b_00(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros(num_atoms, dtype=complex)

    overall_const = -(0.5)*(mass)**(-1.0)*1j

    for j in range(num_atoms):

        LxS_val = mat_properties_dict["LxS_list"][particle_id][j]
        
        val[j] = overall_const*(
                            (LxS_val[0][1]-LxS_val[1][0])*q_vec[2]
                            +(LxS_val[1][2]-LxS_val[2][1])*q_vec[0]
                            +(LxS_val[2][0]-LxS_val[0][2])*q_vec[1]
                    )

#        for alpha in range(3):
#            for beta in range(3):
#                for chi in range(3):
#
#                    val[j] += overall_const*(
#                                    LeviCivita(alpha, beta, chi)*LxS_val[alpha][beta]*q_vec[chi]
#                            )

    return val

def V8a_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = 0.5

        for j in range(num_atoms):

                N_val = mat_properties_dict["N_list"][particle_id][j]

                val[j] = overall_const*(
                                -N_val*(q_vec/mass)
                        )

        return val

def V8a_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3, 3), dtype=complex)

        overall_const = 1

        for j in range(num_atoms):

                N_val = mat_properties_dict["N_list"][particle_id][j]

                val[j] = overall_const*(
                                N_val*np.identity(3)
                        )

        return val

def V8b_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = -0.5

        for j in range(num_atoms):

                L_val = mat_properties_dict["L_list"][particle_id][j]

                val[j] = overall_const*(
                                (1j/mat_properties_dict["mass"][particle_id])*np.cross(q_vec, L_val)
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

def V12a_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = -0.5/mass

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.cross(S_val,q_vec)
                        )

        return val

def V12a_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3, 3), dtype=complex)

    overall_const = 1

    for j in range(num_atoms):

        S_val = mat_properties_dict["S_list"][particle_id][j]
        
        val[j][0][1] = overall_const*S_val[2]
        val[j][1][2] = overall_const*S_val[0]
        val[j][2][0] = overall_const*S_val[1]
        
        val[j][1][0] = -val[j][0][1]
        val[j][2][1] = -val[j][1][2]
        val[j][0][2] = -val[j][2][0]

#        for alpha in range(3):
#            for beta in range(3):
#                for chi in range(3):
#
#                    val[j][alpha][beta] += overall_const*(
#                                    LeviCivita(beta, chi, alpha)*S_val[chi]
#                            )

    return val

def V12b_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = -0.5*(1j/mat_properties_dict["mass"][particle_id])

        for j in range(num_atoms):

                LxS_val = mat_properties_dict["LxS_list"][particle_id][j]

                val[j] = overall_const*(
                                np.trace(LxS_val)*q_vec - np.matmul(LxS_val,q_vec)
                        )

        return val

def V13a_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = -0.5*(1j/mat_properties_dict["mass"][particle_id])/mass

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.dot(q_vec, S_val)*q_vec
                        )

        return val

def V13a_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3, 3), dtype=complex)

        overall_const = (1j/mat_properties_dict["mass"][particle_id])

        for j in range(num_atoms):

                S_val = mat_properties_dict["S_list"][particle_id][j]

                val[j] = overall_const*(
                                np.dot(q_vec, S_val)*np.identity(3)
                        )

        return val

def V13b_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

        val = np.zeros((num_atoms, 3), dtype=complex)

        overall_const = -0.5*mat_properties_dict["mass"][particle_id]**(-2)

        for j in range(num_atoms):

                LxS_val = mat_properties_dict["LxS_list"][particle_id][j]

                val[j] = overall_const*(
                                np.cross(np.matmul(LxS_val,q_vec), q_vec)
                        )

        return val

def V14a_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3, 3), dtype=complex)

    overall_const = 1j/mat_properties_dict["mass"][particle_id]

    for j in range(num_atoms):

        S_val = mat_properties_dict["S_list"][particle_id][j]

        for alpha in range(3):
            for beta in range(3):

                val[j][alpha][beta] += overall_const*(
                                S_val[alpha]*q_vec[beta]
                        )

    return val

def V14b_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3), dtype=complex)

    overall_const = 0.5*mat_properties_dict["mass"][particle_id]**(-2)

    for j in range(num_atoms):

        LxS_val = mat_properties_dict["LxS_list"][particle_id][j]
        
        val[j] = overall_const*(
                            (LxS_val[0][1]-LxS_val[1][0])*q_vec[2]
                            +(LxS_val[1][2]-LxS_val[2][1])*q_vec[0]
                            +(LxS_val[2][0]-LxS_val[0][2])*q_vec[1]
                    ) *q_vec

#        for alpha in range(3):
#            for beta in range(3):
#                for chi in range(3):
#
#                    val[j] += overall_const*(
#                                    LeviCivita(alpha, beta, chi)*LxS_val[alpha][beta]*q_vec[chi]*q_vec
#                            )

    return val

def V15a_11(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3, 3), dtype=complex)

    overall_const = -mat_properties_dict["mass"][particle_id]**(-2)

    for j in range(num_atoms):

        S_val = mat_properties_dict["S_list"][particle_id][j]

        val[j][0][1] = overall_const*q_vec[2]*np.dot(q_vec,S_val)
        val[j][1][2] = overall_const*q_vec[0]*np.dot(q_vec,S_val)
        val[j][2][0] = overall_const*q_vec[1]*np.dot(q_vec,S_val)
        
        val[j][1][0] = -val[j][0][1]
        val[j][2][1] = -val[j][1][2]
        val[j][0][2] = -val[j][2][0]

#        for alpha in range(3):
#            for beta in range(3):
#                for chi in range(3):
#
#                    val[j][alpha][beta] += overall_const*(
#                                    LeviCivita(beta, chi, alpha)*q_vec[chi]*np.dot(q_vec,S_val)
#                            )

    return val

def V15b_01(q_vec, particle_id, num_atoms, mat_properties_dict, mass, spin):

    val = np.zeros((num_atoms, 3), dtype=complex)

    overall_const = -0.5*1j*mat_properties_dict["mass"][particle_id]**(-3)

    for j in range(num_atoms):

            LxS_val = mat_properties_dict["LxS_list"][particle_id][j]

            val[j] = overall_const*(
                            np.dot(q_vec,q_vec)*np.matmul(LxS_val,q_vec) - q_vec*np.matmul(q_vec, np.matmul(LxS_val,q_vec))
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

	op_id : NR operator ID (str)
	exp_id : expansion ID defined above (str)
	particle_id: type of particle (str)

        Fills a dictionary with functions, then calls the dictionary to the specific 
        function, then evaluates the function at the arguments.

        Avoids evaluating all of the functions every time to build the dictionary.

	"""

	return {
		"1":{
			"00": V1_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"3a":{
			"00": zeros_00,
			"01": zeros_01,
			"10": V3a_10,
			"11": zeros_11
		},
		"3b":{
			"00": V3b_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"4":{
			"00": zeros_00,
			"01": V4_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"5a":{
			"00": zeros_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": V5a_11
		},
		"5b":{
			"00": zeros_00,
			"01": V5b_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"6":{
			"00": zeros_00,
			"01": V6_01,
            "10": zeros_01,
			"11": zeros_11
		},
		"7a":{
			"00": V7a_00,
			"01": zeros_01,
			"10": V4_01,
			"11": zeros_11
		},
		"7b":{
			"00": V7b_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"8a":{
			"00": zeros_00,
			"01": V8a_01,
			"10": zeros_01,
			"11": V8a_11
		},
		"8b":{
			"00": zeros_00,
			"01": V8b_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"9":{
			"00": zeros_00,
			"01": V9_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"10":{
			"00": V10_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"11":{
			"00": zeros_00,
			"01": V11_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"12a":{
			"00": zeros_00,
			"01": V12a_01,
			"10": zeros_01,
			"11": V12a_11
		},
		"12b":{
			"00": zeros_00,
			"01": V12b_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"13a":{
			"00": zeros_00,
			"01": V13a_01,
			"10": zeros_01,
			"11": V13a_11
		},
		"13b":{
			"00": zeros_00,
			"01": V13b_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"14a":{
			"00": zeros_00,
			"01": V13a_01,
			"10": zeros_01,
			"11": V14a_11
		},
		"14b":{
			"00": zeros_00,
			"01": V14b_01,
			"10": zeros_01,
			"11": zeros_11
		},
		"15a":{
			"00": zeros_00,
			"01": zeros_01,
			"10": zeros_01,
			"11": V15a_11
		},
		"15b":{
			"00": zeros_00,
			"01": V15b_01,
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

def cNe_func(non_zero_indices, q_vec, num_atoms, mat_properties_dict,
					mass, spin, c_dict_full):

    cNe_00 = 0
    cNe_01 = np.zeros(3, dtype=complex)
    cNe_11 = np.zeros((3, 3), dtype=complex)
    
    if ["1","e"] in non_zero_indices:
        
        cNe_00 += c_dict_full["1"]["e"]
    
    if ["5a","e"] in non_zero_indices:
        
        cNe_11[0][1] = c_dict_full["5a"]["e"]*(1j/mat_properties_dict["mass"]["e"])*q_vec[2]
        cNe_11[1][2] = c_dict_full["5a"]["e"]*(1j/mat_properties_dict["mass"]["e"])*q_vec[0]
        cNe_11[2][0] = c_dict_full["5a"]["e"]*(1j/mat_properties_dict["mass"]["e"])*q_vec[1]
        
        cNe_11[1][0] = -cNe_11[0][1]
        cNe_11[2][1] = -cNe_11[1][2]
        cNe_11[0][2] = -cNe_11[2][0]
        
#        for alpha in range(3):
#            for beta in range(3):
#                for chi in range(3):
#
#                    cNe_11[alpha][beta] += c_dict_full["5a"]["e"]*(1j/mat_properties_dict["mass"]["e"])*LeviCivita(beta, chi, alpha)*q_vec[chi]
    
    if ["8a","e"] in non_zero_indices:
        
        cNe_01 += c_dict_full["8a"]["e"]*(-0.5)*q_vec/mass
        cNe_11 += c_dict_full["8a"]["e"]*np.identity(3)
    
    if ["11","e"] in non_zero_indices:
        
        cNe_01 += c_dict_full["11"]["e"]*(1j/mat_properties_dict["mass"]["e"])*q_vec

    return {
        "00": cNe_00,
        "01": cNe_01,
        "11": cNe_11
    }
