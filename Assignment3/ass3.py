from __future__ import division
import math
import os

import numpy as np
import cv2
import pickle

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
        print np.sum(matrix[row])
        # Take mean of row
        mean = np.nanmean(matrix[row])
        normal_matrix[row] = matrix[row] - mean
    return normal_matrix

def find_dense_block(matrix):
    for c,column in enumerate(matrix.T):
        filled_rows = 0
        for r,val in enumerate(column):
            if val > 0.0:
                filled_rows += 1
            else:
                # When first 0 is encountered after non-0, break
                if (filled_rows > 0):
                    # But first check if there were enough filled rows
                    if (filled_rows > 0):
                        pass

if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    #print matr
    norm_matr = normalize_point_coordinates(matr)
    #print norm_matr
    #print(np.sum(pointview_mtr[:],axis=1)[0])
