from __future__ import division
import math

__author__ = 'bas'
import numpy as np
import cv2


class PointChaining(object):
    def __init__(self, n_iterations, images):
        self.images = images
        self.n_iterations = n_iterations

    def detect_feature_points(self, img):
        sift = cv2.SIFT()
        # Detect feature points using SIFT and compute descriptors
        kp, des = sift.detectAndCompute(img, None)

        # Show feature points (optional)
        imgkp = cv2.drawKeypoints(img, kp)
        cv2.imshow('img with keypoints', imgkp)
        cv2.waitKey(2000)
        # Return keypoints and descriptors
        return kp, des

    def find_matches(self, kp1, des1, kp2, des2):
        # Create brute force matcher
        bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        dmatches = bfm.match(des1, des2)
        # TODO: maybe speed this up?
        coords1 = np.zeros((len(dmatches), 2))
        coords2 = np.zeros((len(dmatches), 2))
        for i, dmatch in enumerate(dmatches):
            x1, y1, x2, y2 = self.get_coordinates(dmatch, kp1, kp2)
            coords1[i, :] = [x1, y1]
            coords2[i, :] = [x2, y2]
        return coords1, coords2

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

    def normalize_coords(self, matches, index=0):
        off = index * 2
        coords = matches[index]

        mean_x, mean_y = list(np.mean(coords, axis=0))
        d = self.compute_d(coords, mean_x, mean_y)
        sqrt2_d = math.sqrt(2) / d

        T = np.array([[sqrt2_d, 0, -mean_x * sqrt2_d], [0, sqrt2_d, -mean_y * sqrt2_d], [0, 0, 1]])
        homo_coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)
        norm_coords = np.dot(T, homo_coords.T).T

        return norm_coords[:, 0:2], T

    def run_pipeline(self):
        img1 = self.images[0]
        img2 = self.images[1]

        kp1, des1 = self.detect_feature_points(img1)
        kp2, des2 = self.detect_feature_points(img2)

        matches = self.find_matches(kp1, des1, kp2, des2)

        norm_coords1, T1 = pc.normalize_coords(matches, 0)
        norm_coords2, T2 = pc.normalize_coords(matches, 1)

        norm_matches = (norm_coords1, norm_coords2)
        norm_F = self.estimate_fundamental_matrix(norm_matches, kp1, kp2)
        F = np.dot(np.dot(T2.T, norm_F), T1)
        print F
        # TODO: Chaining en met RANSAC berekenen.


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == '__main__':
    images = [read_image("House/frame00000001.png"), read_image("House/frame00000005.png")]
    pc = PointChaining(100, images)
    pc.run_pipeline()