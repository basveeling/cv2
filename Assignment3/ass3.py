from __future__ import division
import math
import os

import numpy as np
import cv2
import pickle

if __name__ == '__main__':
    f = open("../pointview.m","rb")
    pointview_mtr = pickle.load(f)
    print pointview_mtr
