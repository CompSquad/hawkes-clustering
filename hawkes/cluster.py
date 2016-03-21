import numpy as np
import pandas as pd

import nimfa
from sklearn.cluster import KMeans

from scipy.sparse import csr_matrix, coo_matrix


def FormatAndSplit(df, cut_date):
    """ Preprocess data into Customer, Ticker, Rating dataframe.
    
    Arguments
    ---------
    df: pd.DataFrame
        The input data
    cut_date: pd.DateTime
        A cut date for train/test splitting
    
    Returns
    -------
    A (train, test) tuple of coo sparse matrices.
    """
    
    ticker_idx = {tck: i for i, tck in enumerate(set(df['Ticker']))}
    df['TradeDateKey'] = pd.to_datetime(df['TradeDateKey'], format='%Y%m%d')
    train = df.ix[df[u'TradeDateKey'] <= cut_date, :]
    test = df.ix[df[u'TradeDateKey'] > cut_date, :]
    
    ddata = df.ix[df["BuySell"] == "Buy", [u'Customer', u'NotionalEUR', u'Ticker', u'TradeDateKey']]

    data_train = pd.DataFrame.copy(ddata)
    data_train.ix[data_train[u'TradeDateKey'] <= cut_date, "NotionalEUR"] = 0
    
    data_test = pd.DataFrame.copy(ddata)
    data_test.ix[data_test[u'TradeDateKey'] > cut_date, "NotionalEUR"] = 0
    
    # Bucket ratings into quantiles.
    cut_count = 10
    n_bins = 4
    labels = range(1, n_bins + 1)
    
    train_dense = data_train.groupby(['Customer', 'Ticker'])
    train_dense = (train_dense['NotionalEUR']
                   .agg({'NotionalSum' : np.sum, 'count' : 'count'})
                   .reset_index())
    train_dense = (train_dense
                   .groupby('Customer')
                   .filter(lambda x: sum(x['count']) >= cut_count))
    train_dense[u'NotionalRating'], bins = pd.qcut(
            train_dense[u'NotionalSum'], n_bins, labels=labels, retbins=True)
    # train_dense[u'NotionalRating'] = train_dense[u'NotionalSum']
    
    test_dense = data_test.groupby(['Customer', 'Ticker'])
    test_dense = (test_dense['NotionalEUR']
                  .agg({'NotionalSum' : np.sum, 'count' : 'count'})
                  .reset_index())
    test_dense = (test_dense.groupby('Customer')
                  .filter(lambda x: sum(x['count']) >= cut_count))
    test_dense[u'NotionalRating'] = pd.cut(
            test_dense[u'NotionalSum'], bins, labels=labels)
    # test_dense[u'NotionalRating'] = test_dense[u'NotionalSum']
    
    train_dense['Ticker'] = train_dense['Ticker'].map(lambda x: ticker_idx[x])
    test_dense['Ticker'] = test_dense['Ticker'].map(lambda x: ticker_idx[x])
    
    train_dense.drop(['count', 'NotionalSum'], axis=1, inplace=True)
    test_dense.drop(['count', 'NotionalSum'], axis=1, inplace=True)
    
    # Remove empty rows
    idx = ~np.isnan(np.array(train_dense['NotionalRating']))
    train_dense = train_dense[idx]
    idx = ~np.isnan(np.array(test_dense['NotionalRating']))
    test_dense = test_dense[idx]
    
    nb_customers = len(set(df['Customer']))
    nb_tickers = len(set(df['Ticker']))
    
    return (train, 
            ToSparse(train_dense, nrow=nb_customers, ncol=nb_tickers),
            test, 
            ToSparse(test_dense, nrow=nb_customers, ncol=nb_tickers), 
            ticker_idx)
    

def ToSparse(df, nrow, ncol):
    
    data = df['NotionalRating']
    row = df['Customer']
    col = df['Ticker']
    train_coo = coo_matrix((data, (row, col)), shape=(nrow, ncol))
    return train_coo


def GetLatentSpace(train, rank=10):
    
    model = nimfa.Nmf(train.todense(), seed='random_vcol', rank=rank, max_iter=100)
    mfit = model()
    M = np.array(mfit.coef())
    N = np.array(mfit.basis())
    
    # GetLatentSpace customers
    M = M.T
    M_normalized = M / np.linalg.norm(M, axis=1)[:, np.newaxis]
    
    # GetLatentSpace tickers
    N = N
    N_normalized = N / np.linalg.norm(N, axis=1)[:, np.newaxis]
    
    return M_normalized, N_normalized


def GetClosestCustomers(ticker_id, train, rank=10):
    
    M, N = GetLatentSpace(train)
    
    centroidN = N[ticker_id]
    
    # Get closest customers
    d = M.dot(centroidN)
    idx = d.argsort()[-20:][::-1]
    return idx
    

if __name__ == "__main__":
    
    df = pd.read_csv('/Users/arnaud/cellule/data/bnpp/ETSAnonymousPricesFull.csv')
    
    cut_date = pd.to_datetime('20131106', format='%Y%m%d')
    train_df, train_coo, test_df, test_coo, ticker_idx = FormatAndSplit(df, cut_date=cut_date)
    cust = GetClosestCustomers(ticker_idx['GAZPRU'], train_coo, train_df, rank=10)  
    
    print train.head()
