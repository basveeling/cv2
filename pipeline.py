import numpy as np
import cv2
from skimage import data
from skimage import transform as tf

# Nice tutorial: https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_tutorials.html

def detect_feature_points(img):
    sift = cv2.SIFT()
    # Detect feature points using SIFT and compute descriptors
    kp, des = sift.detectAndCompute(img, None)

    # Show feature points (optional)
    imgkp = cv2.drawKeypoints(img, kp)
    cv2.imshow('img with keypoints', imgkp)
    # cv2.waitKey(5000)

    # Return keypoints and descriptors
    return kp, des


def find_matches(kp1, des1, kp2, des2):
    # Create brute force matcher
    bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bfm.match(des1, des2)
    return matches


def get_coordinates(match, keypoints1, keypoints2):
    x1, y1 = keypoints1[match.queryIdx].pt
    x2, y2 = keypoints2[match.trainIdx].pt
    return x1, y1, x2, y2


def perform_ransac(matches, kp1, kp2, n_iterations):
    best_n_inliers = 0
    best_h = None
    for i in range(0, n_iterations):
        # Take random subset P from matches
        np.random.shuffle(matches)
        P = matches[:4]  # 5 matches?
        A = np.zeros((0, 8))
        b = np.zeros((0, 1))
        for match in P:
            x1, y1, x2, y2 = get_coordinates(match, kp1, kp2)
            A = np.vstack((A, np.array([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])))
            A = np.vstack((A, np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])))
            b = np.append(b, x2)
            b = np.append(b, y2)

        x = np.dot(np.linalg.pinv(A), b)
        x = np.concatenate((x, [1]), axis=0)  # Add 1 to x
        h = np.reshape(x, (3, 3))  # reshape to 3 by 3 matrix
	
        # Check inliers
        kp1_matrix = np.array([[p.pt[0], p.pt[1], 1] for p in kp1]).T
        kp2_matrix = np.array([[p.pt[0], p.pt[1], 1] for p in kp2]).T
        kp2_est = np.dot(h, kp1_matrix)
        kp2_est = kp2_est[:, :] / kp2_est[2, :]

        n_inliers = 0
        for match in matches:
            kp1_id = match.queryIdx
            kp2_id = match.trainIdx
            if kp2_matrix[0, kp2_id] - 10 <= kp2_est[0, kp1_id] <= kp2_matrix[0, kp2_id] + 10 and kp2_matrix[
                1, kp2_id] - 10 <= kp2_est[1, kp1_id] <= kp2_matrix[1, kp2_id] + 10:
                n_inliers += 1
            else:
                pass

        # print n_inliers, kp1_matrix.shape, kp2_matrix.shape
        # print ".",
        if best_n_inliers < n_inliers:
            best_n_inliers = n_inliers
            best_h = h

        pass
    # print "\n", best_n_inliers
    return best_h

        

def show_transformed_kp(img1,img2,kp1,h):
    # Transform keypoints from img1 using homography h
    kp1_matrix = np.array([[p.pt[0], p.pt[1], 1] for p in kp1]).T
    #kp2_matrix = np.array([[p.pt[0], p.pt[1], 1] for p in kp2]).T
    kp2_est = np.dot(h, kp1_matrix)
    kp2_est = kp2_est[:, :] / kp2_est[2, :]
    h1 = img1.shape[0]
    h2 = img2.shape[0]
    w1 = img1.shape[1]
    w2 = img2.shape[1]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    #print kp2_est
    for k in range(0,len(kp1)):
        # Plot img1 keypoints
        kpx = int(kp1[k].pt[0])
        kpy = int(kp1[k].pt[1])
        cv2.circle(vis,(kpx,kpy),1,(255,0,0))
        
        # Plot estimated img2 keypoints
        estx = int(kp2_est[0,k]) + w1
        esty = int(kp2_est[1,k])
        #print kp2_est[:2,k]
        cv2.circle(vis,(estx,esty),1,(255,0,0))
        #cv2.line(vis,(kpx,kpy),(estx,esty),(255,0,0))
    print np.shape(vis)
    plot_image = cv2.imshow("combined", vis)
    
    
    cv2.waitKey(-1)

def perform_lo_ransac(matches):
    pass


def estimate_new_size(img1, img2, homography):

    pass


def stitch(img1, img2, h, new_size):
    r1,c1 = img1.shape
    r2,c2 = img2.shape

    dst = cv2.warpPerspective(img2,h,(c2,r2))
    # TODO: h wordt nog niet goed berekend
    plot_image = cv2.imshow("combined2", dst)
    cv2.waitKey(2000)
    pass


def main():
    # Load images
    img1 = cv2.imread("bus/bus_left.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("bus/bus_right.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect feature points and compute descriptors
    kp1, des1 = detect_feature_points(img1)
    kp2, des2 = detect_feature_points(img2)
    # Find matches between descriptors of img1 and img2
    matches = find_matches(kp1, des1, kp2, des2)

    # Perform RANSAC
    homography = perform_ransac(matches, kp1, kp2, 1)
    
    # Show transformed keypoints
    show_transformed_kp(img1,img2,kp1,homography)
    # Calculate size of new image
    new_size = estimate_new_size(img1, img2, homography)

    # Stitch images together
    stitch(img1, img2, homography, new_size)


if __name__ == "__main__":
    main()

