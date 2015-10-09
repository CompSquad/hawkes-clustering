import numpy as np
from scipy.optimize import newton


def TDMatrix(t):
    """ Computes the matrix of time differences with entry m_ij = t_i - t_j.
    
    Since negative time (t_i - t_j < 0) are not defined in the model, they are 
    replaced by 0 by convention.
    
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
            dt[i, j] = max(0, t[i] - t[j])
    return dt

   
def StructProbMatrix(dt, mu, a, b):
    """ Compute the structure matrix with entries m_ij = P(u_i = j).
    
    Parameters
    ----------
    dt : ndarray
        2D array of time differences.
    mu : float
        Immigrant density.
    a : float 
        Kernel parameter.
    b : float
        Kernel parameter.
    
    Returns
    -------
    p : ndarray
        Array of parenthood probabilities.
    """
    p = a * np.exp(-b * dt)
    p[np.triu_indices(p.shape[0])] = 0
    p[np.diag_indices(p.shape[0])] = mu
    p /= np.sum(p, axis=1)[np.newaxis].T
    return p
    

def StructProbMatrixFromTime(t, mu, a, b):
    """ Compute the structure matrix directly from `t`.
    
    Reference
    ---------
    Equation (2) page 7 of Guillaume's pdf.
    
    Parameters
    ----------
    t : ndarray
        1D array of event times.
    mu : float
    a : float 
        Kernel parameter.
    b : float
        Kernel parameter.
    
    Returns
    -------
    p : ndarray
        2D array of parenthood probabilities.
    """
    n = len(t)
    p = np.tile(np.exp(b * t), (n, 1))
    p[np.triu_indices(n)] = 0.
    p[np.diag_indices(n)] = mu / a * np.exp(b * t)
    p /= np.sum(p, axis=1)[np.newaxis].T
    return p
   
def Q(mu, a, b, t, p, dt):
    """ Computes Q(theta, theta_old).
    
    Note
    ----
    The theta_old parameter are implicit in dt.
    
    Parameters
    ----------
    mu : float
        New param.
    a : float
        New param.
    b : float
        New param.
    t : array-like
        Times of events.
    p : ndarray
        2D array of parenthood probabilities.
    dt : ndarray
        2D array of time differences.
    
    Returns
    -------
    q : float
        Value of Q(theta, theta_old).
    """
    T = t[-1] - t[0]
    q = np.log(mu) * np.diag(p).sum() - mu * T \
        + (p * (np.log(a) - b * dt)).sum() \
        + a / b * (np.exp(-b * (T - t)) - 1).sum()
    return q
    

def f(t, T, S0, S1, S2):
    """ Function appearing in the third equation of the M-step.
    
    Reference
    ---------
    Guillaume's PDF.
    
    Returns
    -------
    fclosed : A function
    """
    def fclosed(b):
        fc = S2 + S1 / b - S1 * (np.sum((T - t) * np.exp(-b * (T - t)))
                                 / np.sum(1 - np.exp(-b * (T - t))))
        return fc
    return fclosed

def ExpectationMaximization(t, niter=100, mu0=.9, a0=.8, b0=.5, callback=None):
    """ Expectation-Maximization as described in Guillaume's PDF.
    """
    
    # Basic data structures and initialization
    T = t[-1] - t[0]
    dt = TDMatrix(t)
    mu, a, b = mu0, a0, b0
    
    for _ in xrange(niter):
        
        # Expectation
        p = StructProbMatrix(dt, mu, a, b)
#         p = StructProbMatrixFromTime(t, mu, a, b)
        
        # Maximization
        S0 = np.sum(np.diag(p))
        S1 = len(t) - S0
        S2 = -(p * dt).sum()
        
        mu = S0 / T
        b = newton(f(t, T, S0, S1, S2), b, tol=0.1)
        a = b * S1 / np.sum(1 - np.exp(-b * (T - t)))
        
        if callback is not None:
            callback(mu, a , b, t, p, dt, T, S0, S1, S2)
            
    return mu, a, b, p

def GetTimeSeriesFromCSV(filepath, nbpoints=None):
    """ Retrieve time series data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    nbpoints : int
        The number of points to retrieve.
    
    Returns
    -------
    time : ndarray
        An array of floating point number from 0. to 1. representing time 
        stamps.
    """
    import pandas as pd
    
    df = pd.read_csv(filepath, parse_dates=['TradeDateTime'],
                     index_col='TradeDateTime')
    ts = pd.Series(df.index)
    time = sorted(ts.astype(int))
    time = np.array(time)
    time -= time[0]
    if nbpoints is not None:
        time = time[:nbpoints]
    time = time / float(time[-1])
    return time


if __name__ == "__main__":
    
    time = GetTimeSeriesFromCSV('../GAZPRU.csv', nbpoints=100)
    
    def inspect(mu, a, b, t, p, dt, T, S0, S1, S2):
        """ A function called after each iteration of the EM algorithm. """
        print ("(mu, a, b) = (%.2f, %.2f, %.2f)" % (mu, a, b)).ljust(35),
        print "Q = %s" % Q(mu, a, b, t, p, dt)
        
    mu, a, b, p = ExpectationMaximization(time, 100, callback=inspect)
    