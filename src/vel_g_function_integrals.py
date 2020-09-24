"""
    Functions integrating an energy conserving delta function
    with different powers of velocity
"""

import numpy as np

import src.physics as physics
import src.constants as const

def v_star_func(q_vec, omega, m, vE_vec):

    q_mag = np.linalg.norm(q_vec)

    v_star = (1.0/q_mag)*(omega + (q_mag**2/(2.0*m)) + np.dot(q_vec, vE_vec))
    
    return v_star 

def v_minus_func(q_vec, omega, m, vE_vec):

    return np.abs(v_star_func(q_vec, omega, m, vE_vec))

def g0_func(q_vec, omega, m, vE_vec):

    q_mag = np.linalg.norm(q_vec)

    c1 = 2*const.PI**2*const.V0**2/(q_mag*const.N0)

    v_minus = v_minus_func(q_vec, omega, m, vE_vec)

    if v_minus <= const.VESC:

        return c1*( np.exp(-v_minus**2/const.V0**2) - np.exp(-const.VESC**2/const.V0**2) )

    else:

        return 0

def g0_func_opt(q_vec, omega, m, vE_vec, v_minus):

    q_mag = np.linalg.norm(q_vec)

    c1 = 2*const.PI**2*const.V0**2/(q_mag*const.N0)

    if v_minus <= const.VESC:

        return c1*( np.exp(-v_minus**2/const.V0**2) - np.exp(-const.VESC**2/const.V0**2) )

    else:

        return 0

def skew_mat(vec):
    """
        Returns the skew symmetric cross product matrix of vec
    """

    skew = np.array([
            [0, -vec[2], vec[1]], 
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
            ])

    return skew


def rot_v1_to_v2(v1_vec, v2_vec):
    """
        Return the rotation matrix which rotates v1 -> v2, v2 = R . v1

        Algorithm taken from : 
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """

    v1_hat = v1_vec/np.linalg.norm(v1_vec)
    v2_hat = v2_vec/np.linalg.norm(v2_vec)
    c = np.dot(v1_hat, v2_hat)

    if c == -1:

        rot_neg = -1*np.identity(3)

        return rot_neg

    elif c == 1:

        return np.identity(3)

    else:

        v_cross = np.cross(v1_hat, v2_hat)
        s = np.linalg.norm(v_cross)
        c = np.dot(v1_hat, v2_hat)

        sk = skew_mat(v_cross)

        sk_sq = np.matmul(sk, sk)

        rot = np.identity(3) + sk + (1 + c)**(-1)*sk_sq

        return rot 


def rot_z_matrix(q_vec):
    """
        Rotates z_hat -> q_hat, q_hat = R . z_hat

    """

    z_hat = np.array([0, 0, 1])

    q_dir = q_vec/np.linalg.norm(q_vec)

    return rot_v1_to_v2(z_hat, q_dir)

def g1_func(q_vec, omega, m, vE_vec):

    v_star = v_star_func(q_vec, omega, m, vE_vec)

    r_mat = rot_z_matrix(q_vec)

    return (v_star*r_mat[:][2] - vE_vec)*g0_func(q_vec, omega, m, vE_vec)

def g1_func_opt(q_vec, omega, m, vE_vec, g0_val, v_star):

    r_mat = rot_z_matrix(q_vec)

    return (v_star*r_mat[:][2] - vE_vec)*g0_val

def B_func(v_minus):

    if v_minus <= const.VESC:

        return (const.V0**2/2)*(np.exp(-v_minus**2/const.V0**2)*(const.V0**2 + v_minus**2)
                        - np.exp(-const.VESC**2/const.V0**2)*(const.VESC**2 + const.V0**2))
    else:

        return 0


def A_mat_func(q_vec, omega, m, vE_vec):

    v_minus = v_minus_func(q_vec, omega, m, vE_vec)
    g0_val = g0_func(q_vec, omega, m, vE_vec)

    q_mag = np.linalg.norm(q_vec)

    a11 = (2*const.PI**2)/(q_mag*const.N0)*B_func(v_minus) - (v_minus**2/2)*g0_val
    a22 = a11
    a33 = v_minus**2*g0_val

    A_mat = np.array([
            [a11, 0, 0],
            [0, a22, 0],
            [0, 0, a33]
            ])

    return A_mat 

def A_mat_func_opt(q_vec, omega, m, vE_vec, v_minus, g0_val):

    q_mag = np.linalg.norm(q_vec)

    a11 = (2*const.PI**2)/(q_mag*const.N0)*B_func(v_minus) - (v_minus**2/2)*g0_val
    a22 = a11
    a33 = v_minus**2*g0_val

    A_mat = np.array([
            [a11, 0, 0],
            [0, a22, 0],
            [0, 0, a33]
            ])

    return A_mat 

def g2_func(q_vec, omega, m, vE_vec):

    g0_val = g0_func(q_vec, omega, m, vE_vec)
    g1_vec = g1_func(q_vec, omega, m, vE_vec)

    r_mat = rot_z_matrix(q_vec)

    g2_mat = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):

            g2_mat[i][j] = (
                        -vE_vec[i]*vE_vec[j]*g0_val 
                        - vE_vec[i]*g1_vec[j]
                        - vE_vec[j]*g1_vec[i]
                        )

    A_mat = A_mat_func(q_vec, omega, m, vE_vec)

    g2_mat += np.matmul(r_mat, np.matmul(A_mat, r_mat.T))

    return g2_mat

def g2_func_opt(q_vec, omega, m, vE_vec, v_minus, g0_val, g1_vec):

    r_mat =rot_z_matrix(q_vec)

    g2_mat = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):

            g2_mat[i][j] = (
                        -vE_vec[i]*vE_vec[j]*g0_val 
                        - vE_vec[i]*g1_vec[j]
                        - vE_vec[j]*g1_vec[i]
                        )

    A_mat = A_mat_func_opt(q_vec, omega, m, vE_vec, v_minus, g0_val)

    g2_mat += np.matmul(r_mat, np.matmul(A_mat, r_mat.T))

    return g2_mat
