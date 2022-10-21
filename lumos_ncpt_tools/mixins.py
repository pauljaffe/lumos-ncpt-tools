import pickle
import os
import functools
import pkgutil

import pandas as pd
import numpy as np


class OutliersMixin:
    supported_methods = {'MAD'}
    
    def filter_outliers_by_subtest(self, score_col, thresh, subtests, method='MAD', df=None):
        """Identify and remove test runs with outlier scores, return the filtered
        data. Outliers are identified for each subtest in df. Note that a given
        test run will be removed if any of the subtests are identified as outliers.

        Args
        ----        
        score_col (str): Name of the column with scores to screen for outliers. 
        thresh (float): Outlier threshold (details provided in 'method' argument).
        subtests (list): List of specific subtest IDs to check for outliers. 
        method (str, optional): Method used to identify outliers. Currently
            only 'MAD' is supported. For the 'MAD' method, the median absolute
            deviation from the median (MAD) of the scores is calculated.
            Scores whose absolute deviation from the median exceeds 
            'thresh' x the MAD are identified as outliers. 
        df (DataFrame, optional): DataFrame containing scores to be screend for outliers.
            If set to None (the default), self.df is used (i.e., the df attribute of the NCPT
            class instance). 
                              
        Returns
        -------
        df_filt (DataFrame): DataFrame with outliers removed (all test runs
            with outlier scores are removed). 
        """ 
        
        if df is None:
            df2filt = self.df.copy(deep=True)
        else:
            df2filt = df.copy(deep=True)
        exclude_ids = []
        for sub in subtests:
            sub_df = df2filt.query('specific_subtest_id == @sub')
            exclude_ids.extend(
                self.find_outliers(sub_df, score_col, thresh, method, sub))
        df_filt = df2filt.query('test_run_id not in @exclude_ids')
        return df_filt 
    
    def find_outliers(self, df, score_col, thresh, method, subtest_id):
        """Return the test run IDs for which the subtest scores in df
        were outliers. 
        Notes: df should only contain data from a single subtest. 

        Args
        ----
        df (DataFrame): DataFrame containing data from a single subtest. 
        score_col (str): Name of the column with scores to screen for outliers. 
        thresh (float): Outlier threshold.
        method (str, optional): Method used to identify outliers. 
            See filter_outliers_by_subtest for details. 
                  
        Returns
        -------
        outlier_ids (set): Set containing the test run IDs with outlier scores.
        """       

        assert method in self.supported_methods, 'Outlier method not supported!' 
        outlier_df = df.copy(deep=True)
        scores = outlier_df[score_col].values        
        if method == 'MAD':
            mad, devs = self._median_absolute_dev(scores)
            outlier_df.loc[:, 'devs'] = devs
            outlier_ids = set(outlier_df.query('devs >= @thresh * @mad')['test_run_id'].unique())        
        print(f'Subtest ID {subtest_id}: N outliers = {len(outlier_ids)}')
        return outlier_ids   
                                
    def _median_absolute_dev(self, data, median=None):
        if median is None:
            median = np.median(data)
        devs = np.abs(data - median)
        return np.median(devs), devs                                              