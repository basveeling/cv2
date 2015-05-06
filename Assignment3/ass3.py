from __future__ import division
import math
import os

import numpy as np
import cv2
import pickle

# Load pointview matrix (list) from file and convert to matrix
# using trailing zeros
def load_pointview_matrix(filename):
    f = open(filename,"rb")
    pointview_list = pickle.load(f)
    longest_row = 0
    for row in pointview_list:
        row_length = len(row)
        if row_length > longest_row:
            longest_row = row_length
    matrix = np.zeros((len(pointview_list),longest_row))
    
    for i,row in enumerate(pointview_list):
        row_length = len(row)
        matrix[i,:row_length] = row
    matrix = np.nan_to_num(matrix)
    return matrix

if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    print matr
    #print(np.sum(pointview_mtr[:],axis=1)[0])
