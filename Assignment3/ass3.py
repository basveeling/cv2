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

if __name__ == '__main__':
    matr = load_pointview_matrix("../pointview.m")
    print matr
    #print(np.sum(pointview_mtr[:],axis=1)[0])
