from __future__ import division
from collections import Counter
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
    n_matches = 2 * 3
    best_r = 0
    cols = matrix.shape[1]
    best_indexes = []
    # Sliding window over 5 rows
    for r in range(0, np.shape(matrix)[0]-n_matches, 2):
        full_cols = []
        indexes = []
        # Look at columns at depth 5
        for c in range(cols):
            # If no NaN in column
            if not np.any(np.isnan(matrix[r:r+n_matches,c])):
                # Found full column
                indexes.append(c)
        if len(indexes) > len(best_indexes):
            best_indexes = indexes
            best_r = r
    print "Found a block of %d by %d starting at r %d" % (n_matches/2, len(best_indexes), best_r)
    return matr[best_r:best_r+n_matches,best_indexes], best_indexes, best_r, n_matches

def derive_structure_motion(dense_matrix):
    # Do singular value decomposition
    U, W, V_T = np.linalg.svd(dense_matrix)
    # Take first three rows/columns
    U3 = U[:,:3]
    W3 = np.diag(W[:3])
    V_T3 = V_T[:3,:]
    print "shapes 3UWVT:", U3.shape, W3.shape, V_T3.shape
    # Compute Motion and Structure matrices
    M = np.dot(U3, sqrtm(W3))
    S = np.dot(sqrtm(W3), V_T3)
    
    Mnew, Snew = eliminate_affine(M, S)

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
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0], projection='3d',aspect='equal')#,aspect='equal')
    ax2 = fig.add_subplot(gs[1], projection='3d',aspect='equal')

    ax1.view_init(elev=18, azim=-18)
    ax2.view_init(elev=18, azim=-18-4)

    ax1.scatter(S[0, :], S[1, :], S[2, :])#,depthshade=True)
    ax2.scatter(S[0,:], S[1,:], S[2,:])#,depthshade=True)

    meanz = np.mean(S[2, :])
    stdz = 2*np.std(S[2, :])
    ax2.set_zlim3d(meanz - stdz, meanz + stdz)
    # ax1.set_zlim3d(meanz - stdz, meanz + stdz)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    # ax.set_xlim3d([-10,10])
    # ax.set_ylim3d([-10,10])
    plt.show()


def run():
    global matr
    matr = load_pointview_matrix("../pointview_project.m")
    # cv2.imshow('mtr',matr)
    # cv2.waitKey(0)
    n_cameras = int(matr.shape[0] / 2)
    n_cols = matr.shape[1]
    norm_matr = normalize_point_coordinates(matr)
    measurement_matrix, seen_cols, best_r, n_matches = find_dense_block(norm_matr)
    
    sufficient_coverage=False
    while (sufficient_coverage==False):
        seen_rows = range(best_r, best_r + n_matches)
        unseen_cols = [i for i in range(n_cols) if i not in seen_cols]
        unseen_rows = [i for i in range(n_cameras * 2) if i not in seen_rows]
        best_col, best_col_length = find_best_col(norm_matr, seen_rows, unseen_cols)
        best_row, best_row_length = find_best_row(norm_matr, seen_cols, unseen_rows)
        
        # If rows or columns have been found:
        if best_col != -1 or best_row != -1:
            # Append row or column to measurement matrix
            if best_row_length > best_col_length: # TODO: can we compare these quantities?
                print "Appending row"
                new_row = []
                for r in seen_cols:
                    new_row.append(norm_matr[best_row,r])
                measurement_matrix = np.vstack((measurement_matrix, new_row))
                seen_rows.append(best_row)
                # TODO: Remove row from norm_matr?
            else:
                print "Appending col"
                # Take rows of the column which are in seen_rows,
                # so match with rows in the patch
                new_column = []
                for i in seen_rows:
                    new_column.append(norm_matr[i,best_col])
                measurement_matrix = np.column_stack((measurement_matrix, new_column))
                seen_cols.append(best_col)
                # TODO: Remove column from norm_matr?
            
            # TODO: Do bundle adjustment
            # Possibly http://docs.opencv.org/modules/stitching/doc/motion_estimation.html
        
        # If no new rows and now new columns can be found, coverage is sufficient
        if (best_col == -1) and (best_row == -1):
            sufficient_coverage = True
    
    M, S, Mam, Sam = derive_structure_motion(measurement_matrix)
    plot_structure_motion(S)


def find_best_col(matr, seen_rows, unseen_cols):
    covered_cols = Counter()
    for col in unseen_cols:
        count = np.count_nonzero(~np.isnan(matr[seen_rows, col])) / 2
        if count >= 2: # TODO: Why is the count never more than 2?
            covered_cols[col] = count
    if len(covered_cols) > 0:
        best_col = covered_cols.most_common(1)[0][0]
        best_length = covered_cols.most_common(1)[0][1]
    else:
        # Return -1 if no new columns can be found
        best_col = -1
        best_length = -1
    return best_col, best_length


def find_best_row(matr, seen_cols, unseen_rows):
    covered_rows = Counter()
    for row in unseen_rows:
        count = np.count_nonzero(~np.isnan(matr[row, seen_cols]))
        if count >= 2: # TODO: Why is the count never more than 2? According to description, 3 points should be visible
            covered_rows[row] = count
    if len(covered_rows) > 0:
        best_row = covered_rows.most_common(1)[0][0]
        best_length = covered_rows.most_common(1)[0][1]
    else:
        # Return -1 if no new rows can be found
        best_row = -1
        best_length = -1
    return best_row, best_length


if __name__ == '__main__':
    run()
