from collections import defaultdict

import numpy as np

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

