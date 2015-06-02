from __future__ import division
from collections import Counter
import pickle

import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mpl_toolkits.mplot3d import Axes3D




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
    n_rows = 2 * 4
    best_r = 0
    cols = matrix.shape[1]
    best_indexes = []
    # Sliding window over 5 rows
    for r in range(0, np.shape(matrix)[0] - n_rows+1, 2):
        full_cols = []
        indexes = []
        # Look at columns at depth 5
        for c in range(cols):
            # If no NaN in column
            if not np.any(np.isnan(matrix[r:r + n_rows, c])):
                # Found full column
                indexes.append(c)
        if len(indexes) > len(best_indexes):
            best_indexes = indexes
            best_r = r
    print "Found a block of %d by %d starting at r %d" % (n_rows / 2, len(best_indexes), best_r)
    return matr[best_r:best_r + n_rows, best_indexes], best_indexes, best_r, n_rows


def derive_structure_motion(dense_matrix):
    # Do singular value decomposition
    U, W, V_T = np.linalg.svd(dense_matrix)
    # Take first three rows/columns
    U3 = U[:, :3]
    W3 = np.diag(W[:3])
    V_T3 = V_T[:3, :]
    print "shapes 3UWVT:", U3.shape, W3.shape, V_T3.shape
    # Compute Motion and Structure matrices
    M = np.dot(U3, sqrtm(W3))
    S = np.dot(sqrtm(W3), V_T3)

    Mnew, Snew = eliminate_affine(M, S)

    return Mnew, Snew, M, S


def eliminate_affine(M, S):
    n_cameras = int(M.shape[0] / 2)
    A = M[0:n_cameras * 2, :]
    B = np.empty((2 * n_cameras, 3))
    for i in range(n_cameras):
        B[i * 2:i * 2 + 2, :] = np.linalg.pinv(A[i * 2:i * 2 + 2, :].T)
    L, _, _, _ = np.linalg.lstsq(A, B)
    C = np.linalg.cholesky(L)
    Mnew = M.dot(C)
    Snew = np.linalg.pinv(C).dot(S)
    print "The following numbers express the learned L matrix, should be approx. 0,0,1,1"
    for i in range(n_cameras):
        print np.dot(np.dot(M[i * 2, :], L), M[i * 2 + 1, :].T), np.dot(np.dot(M[i * 2 + 1, :], L),
                                                                        M[i * 2, :].T), np.dot(np.dot(M[i * 2, :], L),
                                                                                               M[i * 2, :].T), np.dot(
            np.dot(M[i * 1, :], L), M[i * 1, :].T)
    return Mnew, Snew


def plot_structure_motion(S):
    # This now plots an cross-eye sterescopic version of the points
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0], projection='3d', aspect='equal')  # ,aspect='equal')
    ax2 = fig.add_subplot(gs[1], projection='3d', aspect='equal')

    ax1.view_init(elev=18, azim=-18)
    ax2.view_init(elev=18, azim=-18 - 4)

    ax1.scatter(S[0, :], S[1, :], S[2, :])  # ,depthshade=True)
    ax2.scatter(S[0, :], S[1, :], S[2, :])  # ,depthshade=True)

    meanz = np.mean(S[2, :])
    stdz = 2 * np.std(S[2, :])
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


def generate_dummy_matrix():
    d = 10
    s = 3

    matr = np.array([[0,d,0,d,0,d,0,d],[0,0,d,d,0,0,d,d],
            [0-s,d-s,0-s,d-s,0+s,d+s,0+s,d+s],[0,0,d,d,0+s*2,0+s*2,d+s*2,d+s*2],
            [0+s,d+s,0+s,d+s,0-s,d-s,0-s,d-s],[0,0,d,d,0+s*2,0+s*2,d+s*2,d+s*2]
            ])
    return matr

def run():
    """
    Cam index is in steps of 2
    Col index is in steps of 1
    :return:
    """
    global matr
    matr = load_pointview_matrix("../pointview_project.m")
    # matr = generate_dummy_matrix()
    # cv2.imshow('mtr',matr)
    # cv2.waitKey(0)
    n_cameras = int(matr.shape[0] / 2)
    n_cols = matr.shape[1]
    norm_matr = normalize_point_coordinates(matr)
    measurement_matrix, seen_cols, start_cam, n_rows = find_dense_block(norm_matr)
    Mp, Sp, Mam, Sam = derive_structure_motion(measurement_matrix)
    print("printing")
    # plot_structure_motion(S)
    seen_cams = range(start_cam, start_cam + n_rows, 2)
    unseen_cols = [i for i in range(n_cols) if i not in seen_cols]
    unseen_cams = [i for i in range(0, n_cameras * 2, 2) if i not in seen_cams]
    sufficient_coverage = False

    D = norm_matr[range(min(seen_cams), max(seen_cams) + 2), :][:, seen_cols]
    S, _, _, _ = np.linalg.lstsq(Mp, D)
    M, _, _, _ = np.linalg.lstsq(np.linalg.pinv(D), np.linalg.pinv(S))
    while (sufficient_coverage == False):
        best_col, best_col_length = find_best_col(norm_matr, seen_cams, unseen_cols)
        best_cam, best_cam_length = find_best_cam(norm_matr, seen_cols, unseen_cams)

        # If rows or columns have been found:
        if best_col != -1 or best_cam != -1:
            # Append row or column to measurement matrix
            if best_cam_length > best_col_length:  # Adding a camera
                print '\nAdding camera %d' % int(best_cam / 2)
                seen_cams.append(best_cam)
                not_nan_cols = np.where((~np.isnan(norm_matr[best_cam])))
                overlapping_cols = np.intersect1d(seen_cols, not_nan_cols)
                structure_indexes = [i for i, col in enumerate(seen_cols) if col in overlapping_cols]
                
                seen_rows = []
                seen_ms = []
                for m_i,cam_i in enumerate(seen_cams):
                    if ~np.any(np.isnan(norm_matr[cam_i,overlapping_cols])):
                        seen_rows.append(cam_i)
                        seen_rows.append(cam_i+1)
                        seen_ms.append(m_i*2)
                        seen_ms.append(m_i*2+1)
                D2 = norm_matr[seen_rows, :][:, overlapping_cols]
                S2 = S[:, structure_indexes]
                M2, _, _, _ = np.linalg.lstsq(np.linalg.pinv(D2), np.linalg.pinv(S2))

                # TODO: make rank 3?
                # TODO: the new params look a bit weird, check if we need to constrain somehow?

                M.resize((M.shape[0] + 2,3), refcheck=False)
                print M[seen_ms,:] - M2
                M[seen_ms,:] = M2

                unseen_cams.remove(best_cam)
            # measurement_matrix = np.vstack((measurement_matrix, norm_matr[[best_cam, best_cam + 1], seen_cols]))
            # TODO: Remove row from norm_matr?
            elif best_col != -1:  # Adding a point
                print "col %d" % best_col,
                seen_cols.append(best_col)
                covered_cams = [(m_i, cam_i) for m_i, cam_i in enumerate(seen_cams) if
                                ~np.isnan(norm_matr[cam_i, best_col])]
                covered_cols = np.array([(ind,i) for ind,i in enumerate(seen_cols) if np.all(~np.isnan(norm_matr[np.array(covered_cams)[:,1], i]))])
                A = []
                b = []
                for m_i, cam_i in covered_cams:
                    A.append(M[m_i*2])
                    A.append(M[m_i*2 + 1])
                    b.append(norm_matr[cam_i, covered_cols[:,1]])
                    b.append(norm_matr[cam_i + 1, covered_cols[:,1]])
                A, b = np.array(A), np.array(b)

                xyz, _, rnk, sing = np.linalg.lstsq(A,b)
                # Resize S
                S.resize((3, S.shape[1]+1), refcheck=False)
                # Find the erronous translation and remove
                diff = np.median(S[:, covered_cols[:, 0]] - xyz, axis=1)
                xyz[0,:] += diff[0]
                xyz[1,:] += diff[1]
                xyz[2,:] += diff[2]
                S[:,covered_cols[:,0]] = xyz

                unseen_cols.remove(best_col)
                # TODO: Remove column from norm_matr?

                # TODO: Do bundle adjustment
                # Possibly http://docs.opencv.org/modules/stitching/doc/motion_estimation.html

        # If no new rows and now new columns can be found, coverage is sufficient
        if (best_col == -1) and (best_cam == -1):
            sufficient_coverage = True

    plot_structure_motion(S)


def find_best_col(matr, seen_cams, unseen_cols):
    covered_cols = Counter()
    for col in unseen_cols:
        count = np.count_nonzero(~np.isnan(matr[seen_cams, col]))
        if count >= 4:
            covered_cols[col] = count
    if len(covered_cols) > 0:
        best_col = covered_cols.most_common(1)[0][0]
        best_length = covered_cols.most_common(1)[0][1]
    else:
        # Return -1 if no new columns can be found
        best_col = -1
        best_length = -1
    return best_col, best_length


def find_best_cam(matr, seen_cols, unseen_cams):
    covered_cams = Counter()
    for cam in unseen_cams:
        count = np.count_nonzero(~np.isnan(matr[cam, seen_cols]))
        if count >= 20:  # TODO: Why is the count never more than 2? According to description, 3 points should be visible
            covered_cams[cam] = count
    if len(covered_cams) > 0:
        best_cam = covered_cams.most_common(1)[0][0]
        best_length = covered_cams.most_common(1)[0][1]
    else:
        # Return -1 if no new rows can be found
        best_cam = -1
        best_length = -1
    return best_cam, best_length


if __name__ == '__main__':
    run()
