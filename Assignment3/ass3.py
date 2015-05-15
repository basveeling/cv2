from __future__ import division
import math
import os

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import cv2
import pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load pointview matrix (list) from file and convert to matrix
# using trailing zeros
def load_pointview_matrix(filename):
    f = open(filename, "rb")
    matrix = pickle.load(f)
    return matrix

def normalize_point_coordinates(matrix):
    # For every row
    n_rows = np.shape(matrix)[0]
    normal_matrix = np.zeros(np.shape(matrix))
    for row in range(n_rows):
        # Take mean of row
        mean = np.nanmean(matrix[row])
        normal_matrix[row] = matrix[row] - mean
    return normal_matrix

def find_dense_block(matrix):
    n_matches = 2 * 5
    best_full_cols = []
    # Sliding window over 5 rows
    for r in range(np.shape(matrix)[0]-n_matches):
        full_cols = []
        # Look at columns at depth 5
        for c in range(np.shape(matrix[r:r+n_matches])[1]):
            # If no NaN in column
            if not np.any(np.isnan(matrix[r:r+n_matches,c])):
                # Found full column
                full_cols.append(matrix[r:r+n_matches,c])
        if len(full_cols) > len(best_full_cols):
            best_full_cols = full_cols

    return np.array(best_full_cols).T


def derive_structure_motion(dense_matrix):
    # Do singular value decomposition
    U, W, V_T = np.linalg.svd(dense_matrix)
    
    # Take first three rows/columns
    U3 = U[:,:3]
    W3 = np.diag(W[:3])
    V_T3 = V_T[:3,:]
    
    # Compute Motion and Structure matrices
    M = np.dot(U3,sqrtm(W3))
    S = np.dot(sqrtm(W3),V_T3)
    
    # TODO (or in other method): eliminate affine ambiguity
    
    return M,S


def plot_structure_motion(M,S):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(S[0,:], S[1,:], S[2,:])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    norm_matr = normalize_point_coordinates(matr)
    measurement_matrix = find_dense_block(matr)
    M,S = derive_structure_motion(measurement_matrix)
    plot_structure_motion(M,S)
