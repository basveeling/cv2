from __future__ import division
import math
import os

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import cv2
import pickle

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

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
    print "Found a block of %d by %d" % (len(best_full_cols), len(best_full_cols[0]))
    return np.array(best_full_cols).T


def derive_structure_motion(dense_matrix):
    # Do singular value decomposition
    U, W, V_T = np.linalg.svd(dense_matrix)
    
    # Take first three rows/columns
    U3 = U[:,:3]
    W3 = np.diag(W[:3])
    V_T3 = V_T[:3,:] #TODO: Return this to non transpose
    print "shapes UWVT:", U3.shape, W3.shape, V_T3.shape
    # Compute Motion and Structure matrices
    M = np.dot(U3, sqrtm(W3))
    S = np.dot(sqrtm(W3), V_T3)
    
    # TODO (or in other method): eliminate affine ambiguity
    Mnew, Snew = eliminate_affine(M, S)

    # OLD:
    # a01 = M[2,:]
    # a02 = M[3,:]
    #
    # A = np.empty((2,3))
    # B = np.empty((2,3))
    # A[0,:] = a01
    # A[1,:] = a02
    # B = np.eye(3).dot(np.linalg.pinv(A.T).T).T
    # # A[2,:] = a01
    # # B[0,:] = np.linalg.pinv(np.atleast_2d(a01)).T
    # # B[1,:] = np.linalg.pinv(np.atleast_2d(a02)).T
    # # B[1,:] = np.linalg.pinv(np.atleast_2d(a02)).T
    # # B[2,:] = np.zeros((1,3))
    # # B[3,:] = np.zeros((1,3))
    # # L = np.random.rand(3,3)
    # L,_,_,_ = np.linalg.lstsq(A,B)
    # print np.dot(np.dot(a01,L),a02)
    return Mnew,Snew


def eliminate_affine(M, S):
    n_cameras = int(M.shape[0] / 2)
    A = M
    B = np.empty((2 * n_cameras, 3))
    for i in range(n_cameras):
        B[i * 2:i * 2 + 2, :] = np.eye(3).dot(np.linalg.pinv(A[i * 2:i * 2 + 2, :].T).T).T
    L, _, _, _ = np.linalg.lstsq(A, B)
    C = np.linalg.cholesky(L)
    Mnew = M.dot(C)
    Snew = np.linalg.pinv(C).dot(S)
    return Mnew, Snew


def plot_structure_motion(M,S):
    # This now plots an cross-eye sterescopic version of the points
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1], projection='3d')

    ax2.view_init(elev=18, azim=-18-5)
    ax1.view_init(elev=18, azim=-18)

    ax2.scatter(S[0,:], S[1,:], S[2,:],depthshade=False)
    ax1.scatter(S[0, :], S[1, :], S[2, :],depthshade=False)

    meanz = np.mean(S[2, :])
    std = np.std(S[2, :])
    ax2.set_zlim3d(meanz - std, meanz + std)
    ax1.set_zlim3d(meanz - std, meanz + std)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    # ax.set_xlim3d([-10,10])
    # ax.set_ylim3d([-10,10])
    plt.show()
    
if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    norm_matr = normalize_point_coordinates(matr)
    measurement_matrix = find_dense_block(matr)
    M,S = derive_structure_motion(measurement_matrix)
    plot_structure_motion(M,S)
