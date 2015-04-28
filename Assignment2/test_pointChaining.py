from unittest import TestCase

__author__ = 'bas'
import ass2
import numpy as np


class TestPointChaining(TestCase):
    def test_sampson_distance(self):
        # pc = PointChaining(100, None)
        F = np.array([[1.17419435e-05, 1.51407381e-06, -1.32022839e-02], [-1.09384537e-05, 3.21830557e-06, 1.03007875e-02],
             [-7.10050606e-03, -3.48840155e-03, 9.09418866e+00]])
        p = np.array([[0, .6, 1]])
        p2 = np.array([[0, .6, 1]])
        p2est = np.dot(p2, np.dot(F, p.T))
        print p2est
        dist_close = ass2.sampson_distance(F, p, p2, 0)
        print dist_close