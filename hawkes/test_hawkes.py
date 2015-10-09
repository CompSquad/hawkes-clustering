import numpy as np
import unittest

from hawkes import * 

class Test(unittest.TestCase):


    def setUp(self):
        self.mu, self.a, self.b = .9, .8, .5
        self.t = np.linspace(0, 1, 100)

    def tearDown(self):
        pass


    def testStructMatricesEquality(self):
        mu, a, b = self.mu, self.a, self.b
        t = self.t
        dt = TDMatrix(t)
        p1 = StructProbMatrix(dt, mu, a, b)
        p2 = StructProbMatrixFromTime(t, mu, a, b)
        np.testing.assert_array_almost_equal(p1, p2)
    
    @unittest.skip("Only useful for debugging.")
    def testVizGradient(self):
        
        def inspect(mu, a, b, t, p, dt, T, S0, S1, S2):
            """ A function called after each iteration of the EM algorithm. """
            
            import matplotlib.pyplot as plt
            
            xx = np.linspace(b - 5, b + 5, 1000)
            fx = [f(t, T, S0, S1, S2)(x) for x in xx]
            plt.plot(xx, fx)
            plt.xlabel(r"$b$")
            plt.ylabel(r"$f(b)$")
            plt.title(r"Graph of $f(b) = \frac{\partial Q(\theta, \theta_{old})}{\partial b}$")
            plt.axhline(0, linestyle='--')
            plt.show()
            ff = lambda k: Q(mu, a, k, t, p, dt)
            fx = [ff(x) for x in xx]
            plt.plot(xx, fx)
            plt.axvline(b, linestyle='--')
            plt.xlabel(r"$b$")
            plt.ylabel(r"$Q(\theta, \theta_{old})$")
            plt.title(r"Graph of $Q(\theta, \theta_{old})$")
            plt.show()
        
        ExpectationMaximization(self.t, 100, callback=inspect)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()