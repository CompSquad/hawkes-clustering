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
        fc = S1 + b * S2 \
             - (b**2 * S1 * np.sum((T - t) * np.exp(-b * (T - t))) /
                np.sum(1 - np.exp(-b * (T - t))))
        return fc
    return fclosed

def ExpectationMaximization(t, niter=100, mu0=.9, a0=.8, b0=.5, callback=None):
    """ Expectation-Maximization as described in Guillaume's PDF. """
    
    # Basic data structures and initialization
    T = t[-1] - t[0]
    dt = TDMatrix(t)
    mu, a, b = mu0, a0, b0
    
    for _ in xrange(niter):
        
        # Expectation
        p = StructProbMatrix(dt, mu, a, b)
#         p = StructProbMatrixFromTime(t, mu, a, b)
        
        # Maximization
        dt[dt < 0] = 0  # Small hack
        S0 = np.sum(np.diag(p))
        S1 = p.sum() - S0
        S2 = (p * dt).sum()
        
        mu = S0 / T
        b = newton(f(t, T, S0, S1, S2), b, maxiter=1000, tol=0.1)
        a = b * S1 / np.sum(1 - np.exp(-b * (T - t)))
        
        if callback is not None:
            callback(mu, a , b, t, p, dt, T, S0, S1, S2)
    
       
if __name__ == "__main__":
    
    import pandas as pd
    
    df = pd.read_csv('../GAZPRU.csv', parse_dates=['TradeDateTime'], index_col='TradeDateTime')
    ts = pd.Series(df.index)
    time = sorted(ts.astype(int))
    time = np.array(time)
    time -= time[0]
    time = time[:100]
    time = time / float(time[-1])
    
    def inspect(mu, a, b, t, p, dt, T, S0, S1, S2):
        """ A function called after each iteration of the EM algorithm. """
        
        import matplotlib.pyplot as plt
        np.set_printoptions(1)
        
        print ("(mu, a, b) = (%.2f, %.2f, %.2f)" % (mu, a, b)).ljust(35),
        print "Q = %s" % Q(mu, a, b, t, p, dt)
        print dt
        print p
        xx = np.linspace(-1, 50, 1000)
        fx = [f(t, T, S0, S1, S2)(x) for x in xx]
        plt.plot(xx, fx)
        plt.xlabel("b")
        plt.ylabel("f(b)")
        plt.title("Find a root with Newton's method...")
        plt.axvline(b, color='k', linestyle='--')
#         plt.plot(xx, S1 + S2 * xx, '--')  # Asymptotic line +inf
        plt.show()
        
    ExpectationMaximization(time, 100, callback=inspect)
    