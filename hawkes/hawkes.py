import numpy as np
from scipy.optimize import newton
from numpy import int64

def TDMatrix(t):
    """ Computes the matrix of time differences with entry m_ij = t_i - t_j.
    
    Since negative time (t_i - t_j < 0) are not defined in the model, they are 
    replaced by `-np.inf`. This should be consistent with the exponentiation 
    performed later on.
    
    Parameters
    ----------
    t : array-like
        An array containing the time of the point process events.
    
    Returns
    -------
    dt : ndarray
        The time differences matrix.
    """
    n = len(t)
    dt = np.empty((n, n))
    for i in xrange(n):
        for j in xrange(n):
            dt[i, j] = t[i] - t[j] if t[i] - t[j] >= 0 else -np.inf
    return dt
    
def StructureProbMatrix(dt, b):
    """ Compute the structure matrix with entries m_ij = P(u_i = j).
    
    Parameters
    ----------
    dt : ndarray
        Array of time differences.
    b : float
        Model parameter (in the kernel).
    
    Returns
    -------
    p : ndarray
        Array of parenthood probabilities.
    """
    p = np.exp(b * dt)
    p /= np.sum(p, axis=1)[np.newaxis].T
    return p

def f(t, T, S0, S1, S2):
    """ Function appearing in the third equation of the M-step.
    
    Reference
    ---------
    Guillaume's PDF.
    """
    def fclosed(b):
        fc = S1 + b * S2 \
             + b**2 * S1 * np.sum((T - t) * np.exp(-b * (T - t))) \
             + np.sum(1 - np.exp(-b * (T - t)))
        return fc
    return fclosed

def ExpectationMaximization(time, niter=100, mu0=1, a0=1, b0=.5, callback=None):
    """ Expectation-Maximization as described in Guillaume's PDF. """
    
    # Basic data structures and initialization
    T = time[-1] - time[0]
    dt = TDMatrix(time)
    mu, a, b = mu0, a0, b0
    
    for _ in xrange(niter):
        
        # Expectation
        p = StructureProbMatrix(dt, b)
        
        # Maximization
        dt[dt < 0] = 0  # Small hack
        S0 = np.sum(np.diag(p))
        S1 = p.sum() - S0
        S2 = (p * dt).sum()
        
        mu = S0 / T
        b = newton(f(time, T, S0, S1, S2), b, tol = 0.001)
        a = b * S1 / np.sum(1 - np.exp(-b * (T - time)))
        
        if callback is not None:
            callback(mu, a , b)
        
if __name__ == "__main__":
    
    import pandas as pd
    df = pd.read_csv('GAZPRU.csv', parse_dates=['TradeDateTime'], index_col='TradeDateTime')
    ts = pd.Series(df.index)
    time = sorted(ts.astype(int))
    t = np.array(time)[:10] / 1e12
    t -= t[0]
    
    def inspect(mu, a, b):
        print mu, a, b
        
    ExpectationMaximization(t, 100, callback=inspect)
    