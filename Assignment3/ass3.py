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

def find_dense_blocks(matrix):
    n_matches = 2 * 4
    sorted_blocks = []
    # Sliding window over 5 rows
    for r in range(np.shape(matrix)[0]-n_matches):
        full_cols = []
        # Look at columns at depth 5
        for c in range(np.shape(matrix[r:r+n_matches])[1]):
            # If no NaN in column
            if not np.any(np.isnan(matrix[r:r+n_matches,c])):
                # Found full column
                full_cols.append(matrix[r:r+n_matches,c])
        if len(full_cols) > len(sorted_blocks):
            sorted_blocks.append((len(full_cols),np.array(full_cols).T))
    sorted_blocks.sort(key=lambda x: x[0],reverse=True)
    print "Found a block of %d by %d" % (len(sorted_blocks), len(sorted_blocks[0]))
    return sorted_blocks


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
    return Mnew,Snew,M,S


def eliminate_affine(M, S):
    n_cameras = int(M.shape[0] / 2)
    A = M[0:n_cameras*2, :]
    B = np.empty((2 * n_cameras, 3))
    for i in range(n_cameras):
        B[i * 2:i * 2 + 2, :] = np.linalg.pinv(A[i * 2:i * 2 + 2, :].T)
    L, _, _, _ = np.linalg.lstsq(A, B)
    C = np.linalg.cholesky(L)
    Mnew = M.dot(C)
    Snew = np.linalg.pinv(C).dot(S)
    print "The following numbers express the learned L matrix, should be approx. 0,0,1,1"
    for i in range(n_cameras):
        print np.dot(np.dot(M[i*2, :], L), M[i*2+1, :].T), np.dot(np.dot(M[i*2+1, :], L), M[i*2, :].T), np.dot(np.dot(M[i*2, :], L), M[i*2, :].T), np.dot(np.dot(M[i*1, :], L), M[i*1, :].T)
    return Mnew, Snew


def plot_structure_motion(S):
    # This now plots an cross-eye sterescopic version of the points
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0], projection='3d')#,aspect='equal')
    # ax2 = fig.add_subplot(gs[1], projection='3d',aspect='equal')

    ax1.view_init(elev=18, azim=-18)
    # ax2.view_init(elev=18, azim=-18-4)

    ax1.scatter(S[0, :], S[1, :], S[2, :],depthshade=True)
    # ax2.scatter(S[0,:], S[1,:], S[2,:],depthshade=True)

    meanz = np.mean(S[2, :])
    stdz = 2*np.std(S[2, :])
#     # ax2.set_zlim3d(meanz - stdz, meanz + stdz)
    # ax1.set_zlim3d(meanz - stdz, meanz + stdz)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')
    # ax.set_xlim3d([-10,10])
    # ax.set_ylim3d([-10,10])
    plt.show()
    
if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    norm_matr = normalize_point_coordinates(matr)
    dense_blocks = find_dense_blocks(matr)
    structures = []
    for _,measurement_matrix in dense_blocks[0:1]:
        M,S,Mam,Sam= derive_structure_motion(measurement_matrix)
        structures.append(S  )

    plot_structure_motion(np.concatenate(structures,axis=1))
    # plot_structure_motion(M,Sam)
