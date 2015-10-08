import numpy as np
import unittest

from hawkes import StructProbMatrixFromTime, StructProbMatrix, TDMatrix

class Test(unittest.TestCase):


    def setUp(self):
        self.mu, self.a, self.b = .9, .8, .5


    def tearDown(self):
        pass


    def testStructMatricesEquality(self):
        mu, a, b = self.mu, self.a, self.b
        t = np.linspace(0, 1, 100)
        dt = TDMatrix(t)
        p1 = StructProbMatrix(dt, mu, a, b)
        p2 = StructProbMatrixFromTime(t, mu, a, b)
        np.testing.assert_array_almost_equal(p1, p2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()