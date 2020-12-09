'''
Set of function used for the Minami & Komatsu birefringence likelihood
Relevant paper is https://arxiv.org/pdf/2006.15982.pdf (here after MK)
And the doc https://github.com/simonsobs/PSpipe/blob/master/project/Planck_pspy/doc/birefringence.pdf
which is mostly a copy of the MK papers.
'''

import pickle
import sys

import numpy as np
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
from pspy import so_dict, so_mcm, so_spectra


def get_R_matrix(alpha_i, alpha_j):
    '''
    Rotation matrix of EE,BB power spectra (Equation 8 of MK)
    '''
    ci = np.cos(2 * alpha_i * np.pi / 180)
    cj = np.cos(2 * alpha_j * np.pi / 180)
    si = np.sin(2 * alpha_i * np.pi / 180)
    sj = np.sin(2 * alpha_j * np.pi / 180)
    return np.array([[ci*cj, si*sj], [si*sj, ci*cj]])

def get_R_vec(alpha_i, alpha_j):
    '''
    Rotation vector from EE,BB power spectra to EB (Equation 9 of MK)
    '''
    ci = np.cos(2 * alpha_i * np.pi / 180)
    cj = np.cos(2 * alpha_j * np.pi / 180)
    si = np.sin(2 * alpha_i * np.pi / 180)
    sj = np.sin(2 * alpha_j * np.pi / 180)
    return np.array([ci*sj, -si*cj])

def get_my_A_vector(alpha_i, alpha_j):
    '''
    Equation 24 of the doc
    '''
    R_vec = get_R_vec(alpha_i, alpha_j)
    R_matrix = get_R_matrix(alpha_i, alpha_j)
    inv_R_matrix = np.linalg.inv(R_matrix)
    A = np.dot(R_vec, inv_R_matrix)
    return A

def get_B_vector(alpha_i, alpha_j, beta):
    '''
    Equation 25 of the doc
    '''
    R_vec = get_R_vec(alpha_i, alpha_j)
    R_matrix = get_R_matrix(alpha_i, alpha_j)
    R_vec_beta = get_R_vec(alpha_i + beta, alpha_j + beta)
    R_matrix_beta = get_R_matrix(alpha_i + beta, alpha_j + beta)
    inv_R_matrix = np.linalg.inv(R_matrix)
    B = R_vec_beta - np.dot(R_vec, np.dot(inv_R_matrix, R_matrix_beta))
    return B
    
def compute_simple_chi2_data(Db, inv_cov, alpha):
    '''
    Not used in the project
    Equation 5 of the doc
    '''
    residual = Db["EB"] - 1 / 2 * (Db["EE"] - Db["BB"]) * np.tan(4 * alpha * np.pi / 180)
    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2

def compute_simple_chi2_theory(Db, Db_th, inv_cov, alpha):
    '''
     Not used in the project
     Equation 6 of the doc
     '''
    residual = Db["EB"] - 1 / 2 * (Db_th["EE"] - Db_th["BB"]) * np.sin(4 * alpha * np.pi / 180)
    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2

def compute_chi2_data(Db, Db_th, inv_cov, alpha, beta):
    '''
     Not used in the project
     Equation 12 of the doc
     '''

    t_alpha = np.tan(4 * alpha * np.pi / 180)
    s_beta = np.sin(4 * beta * np.pi / 180)
    c_alpha = np.cos(4 * alpha * np.pi / 180)
    
    
    residual = Db["EB"] - 1 / 2 * (Db["EE"] - Db["BB"]) * t_alpha
    residual -=  1 / 2 * s_beta / c_alpha *  (Db_th["EE"] - Db_th["BB"])

    chi2 = np.dot(residual, np.dot(inv_cov, residual))
    return chi2

def compute_simple_sigma_alpha(Db, inv_cov):
    '''
     Not used in the project
     Fisher uncertainties corresponding to Equation 6 of the doc
     '''
    vec = Db["EE"] - Db["BB"]
    fisher = 4 * np.dot(vec, np.dot(inv_cov, vec))
    return 1 / np.sqrt(fisher) * 180 / np.pi
