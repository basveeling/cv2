import numpy as np
import cv2
import itertools

# Nice tutorial: https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_tutorials.html

def detect_feature_points(img):
    sift = cv2.SIFT()
    # Detect feature points using SIFT and compute descriptors
    kp,des = sift.detectAndCompute(img, None)
    
    # Show feature points (optional)
    imgkp=cv2.drawKeypoints(img,kp)
    cv2.imshow('img with keypoints',imgkp)
    cv2.waitKey(5000)
    
    # Return keypoints and descriptors
    return kp, des

def find_matches(kp1,des1,kp2,des2):
    # Create brute force matcher
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bfm.match(des1,des2)
    return matches


def get_coordinates(match, keypoints1, keypoints2):
    x1,y1 = keypoints1[match.queryIdx].pt
    x2,y2 = keypoints2[match.trainIdx].pt
    return x1,y1,x2,y2

def perform_ransac(matches, kp1,kp2, n_iterations):
    for i in range(0,n_iterations):
        # Take random subset P from matches
        np.random.shuffle(matches)
        P = matches[:5] # 5 matches?
        A = np.array([])
        b = np.array([])
        for match in P:
            x1,y1,x2,y2 = get_coordinates(match, kp1,kp2)
            A = np.vstack((A,np.array([x1,y1,1,0,0,0, -x2*x1, -x2*y1])))
            A = np.vstack((A,np.array([0,0,0,x1,y1,1,-y2*x1,-y2*y1])))    
            b = np.vstack((b,np.array([x2])))
            b = np.vstack((b,np.array([y2])))
        print A
        print b
        x = np.dot(np.linalg.pinv(A),b)
        print x


def perform_lo_ransac(matches):
    pass

def estimate_new_size(img1,img2,homography):
    pass

def stitch(img1,img2,homography,new_size):
    pass

def main():
    # Load images
    img1 = cv2.imread("bus/bus_left.jpg",cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("bus/bus_right.jpg",cv2.IMREAD_GRAYSCALE)
    
    # Detect feature points and compute descriptors
    kp1, des1 = detect_feature_points(img1)
    kp2, des2 = detect_feature_points(img2)
    # Find matches between descriptors of img1 and img2
    matches = find_matches(kp1,des1,kp2,des2)
    
    # Perform RANSAC
    homography = perform_ransac(matches,kp1,kp2,1)
    
    # Calculate size of new image
    new_size = estimate_new_size(img1,img2,homography)
    
    # Stitch images together
    stitch(img1,img2,homography,new_size)
    
    

if __name__ == "__main__":
    main()

