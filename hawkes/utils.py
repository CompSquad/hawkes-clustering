from collections import defaultdict

import numpy as np
import pandas as pd


def GetTimeSeriesFromDF(df, nb_points=None):
    
    ds = df.sort_index()
    ts = pd.Series(ds.index)
    time = ts.astype(int)
    time = np.array(time)
    time -= time[0]
    if nb_points is not None:
        time = time[:nb_points]
    time = time / float(time[-1])
    return time, ds.ix[:nb_points, :]

def GetTimeSeriesFromCSV(filepath, nb_points=None):
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
    
    df = pd.read_csv(filepath, parse_dates=['TradeDateTime'],
                     index_col='TradeDateTime')
    return GetTimeSeriesFromDF(df, nb_points)


def GetInfluenceMatrix(p, customer_ids, stat_func=np.mean):
    """ Builds influencer matrix.
    
    By default entry u_{i, j} = E(i is an influencer of j). Custom influence
    scores can be passed to `stat_func`.
    
    Parameters
    __________
    p : ndarray[ndim=2]
        matrix of (P[u_i=j]) where P[u_i=j] is the probability that jump_i was
        triggered by jump_j.
    customer_ids: ndarray[ndim=1]
        A mark vector with client ids.
    
    Returns
    _______
    adjancy_matrix: ndarray
        Matrix with entries u_{i, j} corresponding to the influence of i on j.
    """
    assert(p.shape[0] == len(customer_ids == p.shape[1]))
    unique_ids = np.unique(customer_ids)
    n_customers = len(unique_ids)
    adjancy_matrix = np.zeros((n_customers, n_customers), dtype=np.float32)
    for i, child in enumerate(unique_ids):
        cids = np.nonzero(customer_ids == child)[0]
        pc = p[cids, :]
        for j, parent in enumerate(unique_ids):
            pids = np.nonzero(customer_ids == parent)[0]
            pp = pc[:, pids]
            adjancy_matrix[j, i] = stat_func(pp)
    return adjancy_matrix, unique_ids

def InfluenceMatrixToGDF(g, unique_ids, filename="influence.gdf"):
    with open(filename, 'w') as f:
        f.write('nodedef> name VARCHAR\n')
        for node in unique_ids:
            f.write("{}\n".format(node))
        f.write('edgedef> node1 VARCHAR, node2 VARCHAR, weight DOUBLE\n')
        N = g.shape[0]
        for i in xrange(N):
            for j in xrange(N):
                f.write("{}, {}, {}\n".format(unique_ids[i], unique_ids[j], g[i, j]))


def GetLeadersFollowers(p):
    """ Retrieve a dictionary of the leaders and followers.
    
    Parameters
    ----------
    p: ndarray
        matrix of (P[u_i=j]) where P[u_i=j] is the probability that jump_i was
        triggered by jump_j.
    
    Returns
    -------
    dict_leaders_followers : dict
        dictionary where keys are the indices of the jumps and values are 
        dictionary with keys "leader" and "followers" with values are sets of
        leader of the jumps and its followers. A jump with a set for key
        "leader" containing its number is an immigrant event.
    """
    dict_leaders_followers = defaultdict(dict)
    ind_max = np.argmax(p, axis=1)
    for i, ind in enumerate(ind_max):
        #print i, ind
        dict_leaders_followers[i] = {"leader": [], "followers": []}
        if i == ind:
            #print '----', i, ind, i == ind
            dict_leaders_followers[i]["leader"].append(i)
        else:
            #print i == ind
            dict_leaders_followers[ind]["followers"].append(i)
            dict_leaders_followers[i]["leader"].append(ind)
    return dict_leaders_followers

