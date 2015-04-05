import numpy as np
import cv2
import itertools

# Nice tutorial: https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_tutorials.html

def main():
    # Load images
    img1 = cv2.imread("bus/bus_left.jpg",cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("bus/bus_right.jpg",cv2.IMREAD_GRAYSCALE)

    
    # Detect feature points using SIFT
    detector = cv2.FeatureDetector_create("SIFT")
    kp = detector.detect(img1)
    img1kp=cv2.drawKeypoints(img1,kp)
    cv2.imshow('img1 with keypoints',img1kp)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

