from __future__ import division
import math
import os

__author__ = 'bas'
import numpy as np
import cv2
import pickle

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


class PointChaining(object):
    def __init__(self, n_iterations, images):
        self.match_dist_threshold = 1000 # TODO: find a correct value for this
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

    def estimate_fundamental_matrix(self, matches, use_all=False):
        # Take random subset P from matches
        n_matches = matches[0].shape[0]
        indexes = np.array(range(n_matches))

        if not use_all: # Pick 8 random points
            np.random.shuffle(indexes)
            indexes = indexes[:8]

        A = np.zeros((len(indexes), 9))
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

    def normalize_coords(self, coords):
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

        norm_coords1, T1 = pc.normalize_coords(matches[0])
        norm_coords2, T2 = pc.normalize_coords(matches[1])
        norm_matches = (norm_coords1, norm_coords2)

        norm_F = self.estimate_fundamental_matrix(norm_matches,use_all=False)

        F = np.dot(np.dot(T2.T, norm_F), T1)
        return F, dmatches, matches

    def find_inliers(self, est_F, homo_coords1, homo_coords2, n_matches):
        cur_inlier_indexes = []
        for m in range(n_matches):
            distance = sampson_distance(est_F, homo_coords1, homo_coords2, m)

            if np.abs(distance) < self.match_dist_threshold:
                cur_inlier_indexes.append(m)
        return cur_inlier_indexes

    def compute_fund_matr_ransac(self, kp1, kp2, des1, des2):
        dmatches = self.dmatches_for_images(kp1, kp2, des1, des2)
        matches = self.make_match_matrix(dmatches, kp1, kp2)

        norm_coords1, T1 = pc.normalize_coords(matches[0])
        norm_coords2, T2 = pc.normalize_coords(matches[1])
        norm_matches = (norm_coords1, norm_coords2)
        
        best_inlier_indexes = []
        best_local_inlier_indexes = []
        best_local_F = None
        homo_coords1, homo_coords2 = self.add_homogenous(matches[0]), self.add_homogenous(matches[1])
        n_matches = homo_coords1.shape[0]
        
        for i in range(self.n_iterations):
            norm_F = self.estimate_fundamental_matrix(norm_matches)
            est_F = np.dot(np.dot(T2.T, norm_F), T1)

            cur_inlier_indexes = self.find_inliers(est_F, homo_coords1, homo_coords2, n_matches)

            if len(cur_inlier_indexes) >= len(best_inlier_indexes) and len(cur_inlier_indexes) >= len(best_local_inlier_indexes):
                best_inlier_indexes = cur_inlier_indexes

                inlier_matches = (norm_coords1[best_inlier_indexes], norm_coords2[best_inlier_indexes])
                norm_local_F = self.estimate_fundamental_matrix(inlier_matches, use_all=True)
                local_F = np.dot(np.dot(T2.T, norm_local_F), T1)

                local_inlier_indexes = self.find_inliers(local_F, homo_coords1, homo_coords2, n_matches)

                if len(local_inlier_indexes) >= len(best_local_inlier_indexes):
                    best_local_F = local_F
                    best_local_inlier_indexes = local_inlier_indexes

        print "Num inliers:", len(best_local_inlier_indexes) / n_matches
        # best_matches =
                
        # # compute best_F based on set of best inliers
        return best_local_F, dmatches, matches

    def show_matches(self, agreeing_matches, dmatches, img1, img2, kp1, kp1_agree_ind, kp2):
        kp1_agree = [kp1[i] for i in kp1_agree_ind]
        kp2_agree = [kp2[dmatches[i].trainIdx] for i in agreeing_matches]
        self.show_transformed_kp(img1, img2, kp1_agree, kp2_agree)

    def create_pointview_row(self, kp1_agree_ind, kp2_agree_ind, last_kp):
        new_pointview_row = [None] * len(last_kp)
        recognized_points = 0
        for p1, p2 in zip(kp1_agree_ind, kp2_agree_ind):
            if p1 in last_kp:
                recognized_points += 1
                new_pointview_row[last_kp.index(p1)] = p2
            else:
                last_kp.append(p1)
                new_pointview_row.append(p2)

        print "Recognized %s" % str(recognized_points / len(last_kp))
        return new_pointview_row

    def create_pointview_2m(self, keypoints, max_row_length, n, pointview_mtr):
        pointview_2m = np.empty((2 * n, max_row_length))
        pointview_2m[:] = np.NAN
        for i in range(n):
            xpoints = [keypoints[i][id].pt[0] if id is not None else np.NAN for id in pointview_mtr[i]]
            ypoints = [keypoints[i][id].pt[1] if id is not None else np.NAN for id in pointview_mtr[i]]
            pointview_2m[2 * i, 0:len(pointview_mtr[i])] = xpoints
            pointview_2m[2 * i + 1, 0:len(pointview_mtr[i])] = ypoints
        return pointview_2m

    def run_pipeline(self):
        n = len(self.images)

        max_row_length = 0

        # Start buffer
        img1 = self.images[0]
        kp1, des1 = self.detect_feature_points(img1)

        keypoints = [kp1]
        pointview_mtr = []
        for ind1 in range(n):
            ind2 = (ind1 + 1) % n
            img2 = self.images[ind2]

            kp2, des2 = self.detect_feature_points(img2)
            keypoints.append(kp2)

            F, dmatches, matches = self.compute_fund_matr_ransac(kp1, kp2, des1, des2)
            agreeing_matches = self.find_agreeing_matches(matches, F)
            kp1_agree_ind = [dmatches[i].queryIdx for i in agreeing_matches]
            kp2_agree_ind = [dmatches[i].trainIdx for i in agreeing_matches]

            if len(pointview_mtr) > 0:
                last_kp = pointview_mtr[-1]
                new_pointview_row = self.create_pointview_row(kp1_agree_ind, kp2_agree_ind, last_kp)
                print(len(new_pointview_row))
                pointview_mtr.append(new_pointview_row)
                max_row_length = max(max_row_length, len(new_pointview_row)) # Keep max row length
            else:
                pointview_mtr.append(kp1_agree_ind)
                pointview_mtr.append(kp2_agree_ind)

            # self.show_matches(agreeing_matches, dmatches, img1, img2, kp1, kp1_agree_ind, kp2)
            # if ind1 > 2:
            #     self.show_pointview_mtr(pointview_mtr,img1, img2, kp1, kp2)
            print len(agreeing_matches) / len(dmatches)
            # Move buffer forward
            img1, kp1, des1 = img2, kp2, des2

        pointview_2m = self.create_pointview_2m(keypoints, max_row_length, n, pointview_mtr)

        print "Saving pointview matrix"
        print(np.shape(pointview_mtr))
        f = open("../pointview.m","wb")
        pickle.dump(pointview_2m,f)
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
        cv2.waitKey(1)
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

    def show_pointview_mtr(self, pointview_mtr, img1, img2, kp1, kp2):
        lists = pointview_mtr[-4:]
        kp1_agree = []
        kp2_agree = []
        for i in range(len(lists[0])):
            if lists[0][i] is not None and lists[1][i] is not None and lists[2][i] is not None and lists[3][i] is not None:
                kp1_agree.append(kp1[lists[2][i]])
                kp2_agree.append(kp2[lists[3][i]])
        self.show_transformed_kp(img1, img2, kp1_agree, kp2_agree)
        pass


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == '__main__':
    images = []
    img_dir = "bearsmall/"
    for file in os.listdir(img_dir):
        images.append(read_image(img_dir + file))
    pc = PointChaining(200, images)
    pc.run_pipeline()
