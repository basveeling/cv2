from __future__ import division
import math
import os

__author__ = 'bas'
import numpy as np
import cv2


def sampson_distance(F, homo_coords1, homo_coords2, i):
    # TODO: verify that this is being computed correctly?
    denominator = np.dot(np.dot(homo_coords1[i].T, F), homo_coords2[i]) 
    squared1 = np.dot(F, homo_coords1[i]) ** 2
    squared2 = np.dot(F.T, homo_coords2[i]) ** 2
    divisor = squared1[0] + squared1[1] + squared2[0] + squared2[1]
    distance = denominator / divisor
    return distance


class PointChaining(object):
    def __init__(self, n_iterations, images):
        self.match_dist_threshold = 5000  # TODO: find a correct value for this
        self.images = images
        self.n_iterations = n_iterations

    def detect_feature_points(self, img):
        sift = cv2.SIFT()
        # Detect feature points using SIFT and compute descriptors
        kp, des = sift.detectAndCompute(img, None)

        # Show feature points (optional)
        # imgkp = cv2.drawKeypoints(img, kp)
        # cv2.imshow('img with keypoints', imgkp)
        # cv2.waitKey(2000)
        # Return keypoints and descriptors
        return kp, des

    def make_match_matrix(self, dmatches, kp1, kp2):
        coords1 = np.zeros((len(dmatches), 2))
        coords2 = np.zeros((len(dmatches), 2))
        for i, dmatch in enumerate(dmatches):
            x1, y1, x2, y2 = self.get_coordinates(dmatch, kp1, kp2)
            coords1[i, :] = [x1, y1]
            coords2[i, :] = [x2, y2]
        return coords1, coords2

    def find_matches(self, kp1, des1, kp2, des2):
        # Create brute force matcher
        bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        dmatches = bfm.match(des1, des2)
        return dmatches

    def get_coordinates(self, match, keypoints1, keypoints2):
        x1, y1 = keypoints1[match.queryIdx].pt
        x2, y2 = keypoints2[match.trainIdx].pt
        return x1, y1, x2, y2

    def estimate_fundamental_matrix(self, matches, kp1, kp2):
        # Take random subset P from matches
        n_matches = matches[0].shape[0]
        indexes = np.array(range(n_matches))
        np.random.shuffle(indexes)
        indexes = indexes[:8]
        A = np.zeros((8, 9))
        for i, match_i in enumerate(indexes):
            x1, y1 = list(matches[0][match_i, :])
            x2, y2 = list(matches[1][match_i, :])
            A[i, :] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

        # Compute F
        U, D, V = np.linalg.svd(A)
        F_unsingular = np.reshape(V[-1], (3, 3))
        Uf, Df, Vf = np.linalg.svd(F_unsingular)
        Df[-1] = 0.0
        F = np.dot(np.dot(Uf, np.diag(Df)), Vf.T)
        return F

    def compute_d(self, coords, mean_x, mean_y):
        d = np.copy(coords)
        d[:, 0] -= mean_x
        d[:, 1] -= mean_y
        d[:, 0] = d[:, 0] ** 2
        d[:, 1] = d[:, 1] ** 2
        d = np.mean(np.sqrt(np.sum(d, axis=1)))
        return d

    def add_homogenous(self, coords):
        return np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)

    def normalize_coords(self, matches, index=0):
        off = index * 2
        coords = matches[index]
        homo_coords = self.add_homogenous(coords)

        mean_x, mean_y = list(np.mean(coords, axis=0))
        d = self.compute_d(coords, mean_x, mean_y)
        sqrt2_d = math.sqrt(2) / d

        T = np.array([[sqrt2_d, 0, -mean_x * sqrt2_d], [0, sqrt2_d, -mean_y * sqrt2_d], [0, 0, 1]])
        norm_coords = np.dot(T, homo_coords.T).T

        return norm_coords[:, 0:2], T

    def dmatches_for_images(self, kp1, kp2, des1, des2):
        dmatches = self.find_matches(kp1, des1, kp2, des2)
        return dmatches

    def compute_fund_matr(self, kp1, kp2, des1, des2):
        dmatches = self.dmatches_for_images(kp1, kp2, des1, des2)
        matches = self.make_match_matrix(dmatches, kp1, kp2)

        norm_coords1, T1 = pc.normalize_coords(matches, 0)
        norm_coords2, T2 = pc.normalize_coords(matches, 1)
        norm_matches = (norm_coords1, norm_coords2)

        norm_F = self.estimate_fundamental_matrix(norm_matches, kp1, kp2)

        F = np.dot(np.dot(T2.T, norm_F), T1)
        return F, dmatches, matches
    
    def compute_fund_matr_ransac(self, kp1, kp2, des1, des2, n_iterations=100):
        dmatches = self.dmatches_for_images(kp1, kp2, des1, des2)
        matches = self.make_match_matrix(dmatches, kp1, kp2)

        norm_coords1, T1 = pc.normalize_coords(matches, 0)
        norm_coords2, T2 = pc.normalize_coords(matches, 1)
        norm_matches = (norm_coords1, norm_coords2)
        
        best_inliers = []
        
        homo_coords1, homo_coords2 = self.add_homogenous(matches[0]), self.add_homogenous(matches[1])
        n_matches = homo_coords1.shape[0]
        
        for i in range(0,n_iterations):
            cur_inliers = []
            norm_F = self.estimate_fundamental_matrix(norm_matches, kp1, kp2)
            F = np.dot(np.dot(T2.T, norm_F), T1)
            for m in range(n_matches):
                distance = sampson_distance(F, homo_coords1, homo_coords2, m)
                
                if np.abs(distance) < self.match_dist_threshold:
                    cur_inliers.append(match)
            if len(cur_inliers) >= len(best_inliers):
                best_inliers = cur_inliers
                
        # compute F_best based on set of best inliers
        best_F = self.estimate_fundamental_matrix(norm_matches, best_inliers[:,0], best_inliers[:,1])
        
        return F_best, dmatches, matches

    def show_matches(self, agreeing_matches, dmatches, img1, img2, kp1, kp1_agree_ind, kp2):
        kp1_agree = [kp1[i] for i in kp1_agree_ind]
        kp2_agree = [kp2[dmatches[i].trainIdx] for i in agreeing_matches]
        self.show_transformed_kp(img1, img2, kp1_agree, kp2_agree)

    def run_pipeline(self):
        n = len(self.images)

        # Start buffer
        img1 = self.images[0]
        kp1, des1 = self.detect_feature_points(img1)

        keypoints = [kp1]
        pointview_mtr = []
        for ind1 in range(n):
            ind2 = (ind1 + 1) % n
            img2 = self.images[ind2]
            kp2, des2 = self.detect_feature_points(img2)
            keypoints.append(keypoints)
            F, dmatches, matches = self.compute_fund_matr(kp1, kp2, des1, des2)
            agreeing_matches = self.find_agreeing_matches(matches, F)
            kp1_agree_ind = [dmatches[i].queryIdx for i in agreeing_matches]
            kp2_agree_ind = [dmatches[i].trainIdx for i in agreeing_matches]

            if len(pointview_mtr) > 0:
                last_kp = pointview_mtr[-1]
                new_pointview_row = [None] * len(pointview_mtr[-1])
                recognized_points = 0
                for p1, p2 in zip(kp1_agree_ind, kp2_agree_ind):
                    if p1 in last_kp:
                        recognized_points += 1
                        new_pointview_row[last_kp.index(p1)] = p2
                    else:
                        last_kp.append(p1)
                        new_pointview_row.append(p2)
                pointview_mtr.append(new_pointview_row)
                print "Recognized %s" % str(recognized_points / len(pointview_mtr[-2]))
                # TODO: almost no points from previous batch are being recognized. Find bug.
            else:
                pointview_mtr.append(kp1_agree_ind)
                pointview_mtr.append(kp2_agree_ind)

            # self.show_matches(agreeing_matches, dmatches, img1, img2, kp1, kp1_agree_ind, kp2)
            print len(agreeing_matches) / len(dmatches)
            # Move buffer forward
            img1, kp1, des1 = img2, kp2, des2
            # TODO: met RANSAC berekenen.
        print "Done"
    def find_agreeing_matches(self, matches, F):
        homo_coords1, homo_coords2 = self.add_homogenous(matches[0]), self.add_homogenous(matches[1])
        n_matches = homo_coords1.shape[0]
        agreeing_matches = []
        for i in range(n_matches):
            distance = sampson_distance(F, homo_coords1, homo_coords2, i)

            if np.abs(distance) < self.match_dist_threshold:
                # print distance, matches[0][i] - matches[0][i], matches[0][i] - matches[1][i]
                agreeing_matches.append(i)

        return agreeing_matches

    def show_transformed_kp(self, img1, img2, kp1, kp2):
        # Transform keypoints from img1 using homography h
        kp1_matrix = np.array([[p.pt[0], p.pt[1], 1] for p in kp1]).T
        kp2_matrix = np.array([[p.pt[0], p.pt[1], 1] for p in kp2]).T
        h1 = img1.shape[0]
        h2 = img2.shape[0]
        w1 = img1.shape[1]
        w2 = img2.shape[1]
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
        # print kp2_est
        for k in range(0, len(kp1)):
            # Plot img1 keypoints
            kpx = int(kp1[k].pt[0])
            kpy = int(kp1[k].pt[1])
            cv2.circle(vis, (kpx, kpy), 1, (255, 0, 0))

            # Plot estimated img2 keypoints
            estx = int(kp2_matrix[0, k]) + w1
            esty = int(kp2_matrix[1, k])
            #print kp2_est[:2,k]
            cv2.circle(vis, (estx, esty), 1, (255, 0, 0))
            cv2.line(vis, (kpx, kpy), (estx, esty), (255, 0, 0))
        print np.shape(vis)
        plot_image = cv2.imshow("combined", vis)
        cv2.waitKey(0)


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == '__main__':
    images = []
    img_dir = "bearsmall/"
    for file in os.listdir(img_dir):
        images.append(read_image(img_dir + file))
    pc = PointChaining(100, images)
    pc.run_pipeline()
