import pkgutil
import csv
import io

import pandas as pd


def load_data(data_path, fn, chunksize=1e6, nrows='all', verbose=False, n_print=5):
    load_str = data_path + '/' + fn
    if nrows == 'all':
        chunks = pd.read_csv(load_str, header=0, chunksize=chunksize)
    else:
        chunks = pd.read_csv(load_str, header=0, chunksize=chunksize, nrows=nrows)
    df = pd.concat(chunks)
    
    if verbose:
        print(fn)
        print(df.info())
        print(df.head(n_print))
    
    return df


def load_test_data():
    data_path = '../tests/test_df.csv'
    df = pd.read_csv(io.StringIO(
        pkgutil.get_data('lumos_ncpt_tools', data_path).decode('utf-8')))
    return df