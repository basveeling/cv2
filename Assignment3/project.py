from __future__ import division
import math
import os

__author__ = 'bas'
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
def sampson_distance(F, h1, h2, i):
    # TODO: verify that this is being computed correctly?
    p = h1[i]
    p_ = h2[i]
    denominator = np.dot(np.dot(p_.T, F), p)**2
    squaredp = np.dot(F, p) ** 2
    squaredp_ = np.dot(F.T, p_) ** 2
    divisor = squaredp[0] + squaredp[1] + squaredp_[0] + squaredp_[1]
    distance = denominator / divisor
    return distance

def drawlines(img1,img2,lines,pts1,pts2):
    # NOTE: from http://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1[0:2]),8,color,-1)
        cv2.circle(img2,tuple(pt2[0:2]),8,color,-1)
    return img1,img2

def drawMatches(img1, kp1, img2, kp2, matches):
    # NOTE: from http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class PointChaining(object):
    def __init__(self, n_iterations, images):
        self.match_dist_threshold = 100 # TODO: find a correct value for this
        self.sift_distance_thres = 160
        self.images = images
        self.n_iterations = n_iterations
        self.n_images = len(images)
        self.list_keypoints = [None] * self.n_images
        self.list_descriptors = [None] * self.n_images
        self.drawing_lvl = 3
        self.use_cv_fund = False
        self.debug_est_fund = True

    def run_pipeline(self):

        # Detect feature points for first image
        self.detect_feature_points(0)
        pointview_mtr = []
        max_row_length = 0
        for i_next in range(1,self.n_images):
            i_prev = i_next - 1
            # Detect feature points for next image
            kp_next, des_next = self.detect_feature_points(i_next)

            # Find matches for this and previous image
            dmatches = self.find_matches(i_prev, i_next)

            # compute fundamental matrix
            F, included = self.compute_fund_matr_ransac(dmatches, i_prev, i_next)

            prev_ind = [dmatches[i].queryIdx for i in included]
            next_ind = [dmatches[i].trainIdx for i in included]
            if len(pointview_mtr) > 0:
                last_kp = pointview_mtr[-1]
                new_pointview_row = self.create_pointview_row(prev_ind, next_ind, last_kp)
                pointview_mtr.append(new_pointview_row)
                max_row_length = max(max_row_length, len(new_pointview_row))  # Keep max row length
            else:
                pointview_mtr.append(prev_ind)
                pointview_mtr.append(next_ind)

        pointview_2m = self.create_pointview_2m(max_row_length, pointview_mtr)

        print "Saving pointview matrix"
        print(np.shape(pointview_2m))
        f = open("../pointview_project.m", "wb")
        pickle.dump(pointview_2m, f)
        print "Done"

    def create_pointview_2m(self, max_row_length, pointview_mtr):
        pointview_2m = np.empty((2 * self.n_images, max_row_length))
        pointview_2m[:] = np.NAN
        for i in range(self.n_images):
            xpoints = [self.list_keypoints[i][id].pt[0] if id is not None else np.NAN for id in pointview_mtr[i]]
            ypoints = [self.list_keypoints[i][id].pt[1] if id is not None else np.NAN for id in pointview_mtr[i]]
            pointview_2m[2 * i, 0:len(pointview_mtr[i])] = xpoints
            pointview_2m[2 * i + 1, 0:len(pointview_mtr[i])] = ypoints
        return pointview_2m


    def create_pointview_row(self, prev_ind, next_ind, last_kp):
        new_pointview_row = [None] * len(last_kp)
        recognized_points = 0
        for p_prev, p_next in zip(prev_ind, next_ind):
            if p_prev in last_kp:
                recognized_points += 1
                new_pointview_row[last_kp.index(p_prev)] = p_next
            else:
                last_kp.append(p_prev)
                new_pointview_row.append(p_next)
        print "Recognized %s" % str(recognized_points / len(last_kp))
        return new_pointview_row

    def add_homogenous(self, coords):
        return np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)

    def compute_d(self, coords, mean_x, mean_y):
        d = np.copy(coords)
        d[:, 0] -= mean_x
        d[:, 1] -= mean_y
        d[:, 0] = d[:, 0] ** 2
        d[:, 1] = d[:, 1] ** 2
        d = np.mean(np.sqrt(np.sum(d, axis=1)))
        return d

    def normalize_coords(self, coords):
        homo_coords = self.add_homogenous(coords)

        mean_x, mean_y = list(np.mean(coords, axis=0))
        d = self.compute_d(coords, mean_x, mean_y)
        sqrt2_d = math.sqrt(2) / d

        T = np.array([[sqrt2_d, 0, -mean_x * sqrt2_d], [0, sqrt2_d, -mean_y * sqrt2_d], [0, 0, 1]])
        norm_coords = np.dot(T, homo_coords.T).T

        return norm_coords[:, 0:2], T

    def find_inliers(self, est_F, homo_coords1, homo_coords2, n_matches):
        cur_inlier_indexes = []
        for m in range(n_matches):
            distance = sampson_distance(est_F, homo_coords1, homo_coords2, m)

            if np.abs(distance) < self.match_dist_threshold:
                cur_inlier_indexes.append(m)
        return cur_inlier_indexes

    def compute_fund_matr_ransac(self,dmatches, i_prev, i_next):
        n_matches = len(dmatches)

        pts_prev, pts_next = self.point_matrixes(dmatches, i_next, i_prev, n_matches)
        # TODO: make own ransac implementation
        # F = self.estimate_fund_matr(rand_matches, i_prev, i_next)
        if self.use_cv_fund:
            F,mask = cv2.findFundamentalMat(pts_prev,pts_next,method=cv2.FM_RANSAC, param1=2., param2=0.99) # TODO: check params
            #pts_prev = pts_prev[mask.ravel() == 1]
            #pts_next = pts_next[mask.ravel() == 1]
            included = [i for i,m in enumerate(mask) if m == 1]
        else:
            best_inlier_indexes = []
            norm_coords1, T1 = pc.normalize_coords(np.array(pts_prev))
            norm_coords2, T2 = pc.normalize_coords(np.array(pts_next))
            homo_coords1, homo_coords2 = self.add_homogenous(pts_prev), self.add_homogenous(pts_next)

            for i in range(self.n_iterations):
                indexes = np.array(range(n_matches))
                np.random.shuffle(indexes)
                indexes = indexes[:8]
                norm_coords1_rand = [norm_coords1[i] for i in indexes]
                norm_coords2_rand = [norm_coords2[i] for i in indexes]

                norm_F = self.estimate_fund_matr(norm_coords1_rand, norm_coords2_rand, i_prev, i_next)
                est_F = np.dot(np.dot(T2.T, norm_F), T1)

                cur_inlier_indexes = self.find_inliers(est_F, homo_coords1, homo_coords2, n_matches)

                if len(cur_inlier_indexes) >= len(best_inlier_indexes):
                    print len(cur_inlier_indexes), n_matches
                    best_inlier_indexes = cur_inlier_indexes

            # Re-estimate with best indexes
            norm_coords1_in = [norm_coords1[i] for i in best_inlier_indexes]
            norm_coords2_in = [norm_coords2[i] for i in best_inlier_indexes]
            norm_F = self.estimate_fund_matr(norm_coords1_in, norm_coords2_in, i_prev, i_next)
            F = np.dot(np.dot(T2.T, norm_F), T1)
            included = best_inlier_indexes

        if self.drawing_lvl > 2:
            self.print_epilines(F, i_next, i_prev, pts_prev[included], pts_next[included])
        return F, included

    def point_matrixes(self, dmatches, i_next, i_prev, n_matches):
        points1, points2 = [], []
        for i in range(n_matches):
            x, y = self.list_keypoints[i_prev][dmatches[i].queryIdx].pt
            x_, y_ = self.list_keypoints[i_next][dmatches[i].trainIdx].pt
            points1.append([x, y])
            points2.append([x_, y_])
        points1, points2 = np.array(points1), np.array(points2)
        return points1, points2

    def estimate_fund_matr(self, norm_coords1_rand, norm_coords2_rand, i_prev, i_next):
        # assuming 8 random matches are already selected (if need be)
        n_matches = len(norm_coords1_rand)

        points1, points2 = [],[]
        A = np.zeros((n_matches, 9))

        for i in range(n_matches):
            x, y = norm_coords1_rand[i][0:2]
            x_, y_ = norm_coords2_rand[i][0:2]
            points1.append([x,y])
            points2.append([x_,y_])
            # TODO: check if above is correct
            A[i,:] = [x*x_, x*y_, x, y*x_, y*y_, y, x_, y_, 1]

        # Fcv, mask = cv2.findFundamentalMat(np.array(points1),np.array(points2),method=cv2.FM_8POINT)
        # Compute F from A
        U, D, V_T = np.linalg.svd(A)
        V = V_T.T
        F_unsingular = np.reshape(V[:,-1], (3,3)).T
        Uf, Df, Vf_T = np.linalg.svd(F_unsingular)
        Df[-1] = 0.0
        F = Uf.dot(np.diag(Df).dot(Vf_T))
        F = F/F[2,2] # Renormalize F so F(3,3) = 1.0

        # if True: #self.debug_est_fund:
        #     print "Debugging estimated fundamental matrix:"
        #     for i in range(n_matches):
        #         points1[i].append(1.)
        #         points2[i].append(1.)
        #         p1 = np.array(points1[i])
        #         p2 = np.array(points2[i])
        #         print i, ":", p2.T.dot(F).dot(p1), p2.T.dot(Fcv).dot(p1), p1.T.dot(Fcv).dot(p2)


        if self.use_cv_fund:
            return Fcv
        else:
            return F

    def print_epilines(self, Fcv, i_next, i_prev, points1, points2):
        lines1 = cv2.computeCorrespondEpilines(np.float32(points2).reshape(-1, 1, 2), 2, Fcv)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(self.images[i_prev], self.images[i_next], lines1, np.int32(points1),
                               np.int32(points2))
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img6)
        plt.show()

    def detect_feature_points(self,ind):
        sift = cv2.SIFT()
        # Detect feature points using SIFT and compute descriptors
        kp, des = sift.detectAndCompute(self.images[ind], None)

        self.list_keypoints[ind] = kp
        self.list_descriptors[ind] = des
        return kp, des

    def find_matches(self, i_prev, i_next):
        # Based on http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        bf = cv2.BFMatcher()
        des_prev, des_next = self.list_descriptors[i_prev], self.list_descriptors[i_next]
        matches = bf.knnMatch(des_prev, des_next, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < .75 * n.distance: # TODO: check ratio (was .75)
                good.append(m)
        if self.drawing_lvl > 3:
            drawMatches(self.images[i_prev], self.list_keypoints[i_prev], self.images[i_next], self.list_keypoints[i_next], good)
        return good

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img
if __name__ == '__main__':
    images = []
    img_dir = "../Assignment2/bear5/"
    for file in os.listdir(img_dir):
        images.append(read_image(img_dir + file))
    pc = PointChaining(200, images)
    pc.run_pipeline()