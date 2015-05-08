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
        # Take mean of row
        mean = np.nanmean(matrix[row])
        normal_matrix[row] = matrix[row] - mean
    return normal_matrix

def find_dense_block(matrix):
    #filled_rows = {}
    #good_column = {}
    #for c,column in enumerate(matrix.T):
        #filled_rows[c] = []
        #for r,val in enumerate(column):
            #if val is not np.NAN:
                #filled_rows[c].append(r)
            #else:
                ## When first 0 is encountered after non-0, break
                #if (len(filled_rows) > 0):
                    ## Check if there were enough filled rows to call this a dense block
                    #if (len(filled_rows) > 6):
                        #good_column[c] = True
                    #else:
                        #good_column[c] = False
                    
                    #break
    
    #for col in good_column:
        #if good_column[col] == False:
            #print col
    n_matches = 5
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
    
    print best_full_cols
    return np.array(best_full_cols).T
        

if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    norm_matr = normalize_point_coordinates(matr)
    dense_matrix = find_dense_block(matr)
    #print norm_matr
    #print(np.sum(pointview_mtr[:],axis=1)[0])
