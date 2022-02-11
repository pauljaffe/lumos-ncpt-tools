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
    
                                            
class ScoreLookupMixin:
    subtests_to_negate = {39, 40, 48, 49}
    census_INT_norm_dir = '/config/census_rank_INT_norm_tables/'
    INT_norm_dir = '/config/rank_INT_norm_tables/'
    
    def lookup_normed_scores(self, method):
        """Lookup normalized scores and add them as a column to self.df.
        The new column will be named '[method]_normed_score'.

        Args
        ----
        method (str): The method used for normalization. 
            Can be either 'rank_INT' or 'census_rank_INT'. 
            rank_INT is a rank-based inverse normal transformation;
            census_rank_INT is a rank-based INT reweighted to match the demographics
            of the 2019 US census (see data descriptor manuscript). Both transformations
            are scaled to have a mean of 100 and a SD of 15. 
        """
        
        print('Looking up norms...')
        subtest_normed_df = pd.concat([df for df in self._lookup_subtest_norms(method)])
        self.df = subtest_normed_df.drop(columns=['negated_score'])
        self.df.sort_values(by=['user_id', 'test_run_id'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print(f'Done! Added normalized scores in column {method}_normed_score')
        
    def add_grand_index(self):
        """Calculate a composite score for each NCPT assessment;
        add this as column 'grand_index' to self.df. The grand index
        is only calculated for completed assessments (i.e., all subtests completed).
        
        Args
        ----
        norm_parent_dir (str): Parent directory containing the folders with the norm tables.
        """
        
        print('Adding grand index...')
        if 'census_rank_INT_normed_score' not in self.df.columns:
            self.lookup_normed_scores('census_rank_INT') 
        
        # Calculate the mean subtest score for completed tests     
        complete_df, incomplete_df = self.filter_by_completeness()
        mean_scores = complete_df.groupby(
            'test_run_id')['census_rank_INT_normed_score'].transform('mean')
        complete_df = complete_df.assign(mean_normed_score=mean_scores)
        incomplete_df = incomplete_df.assign(mean_normed_score=np.nan)
        self.df = pd.concat([complete_df, incomplete_df])
        
        # Look up the grand index by battery
        battery_normed_df = pd.concat(
            [df for df in self._lookup_battery_norms('census_rank_INT')])                               
        GI_df = battery_normed_df[['test_run_id', 'grand_index']].drop_duplicates()
        GI_map = dict(zip(GI_df['test_run_id'], GI_df['grand_index']))        
        self.df.loc[:, 'grand_index'] = self.df['test_run_id'].map(GI_map)
        
        self.df.drop(columns=['mean_normed_score'], inplace=True)
        self.df = self.df.sort_values(by=['user_id', 'test_run_id'])
        self.df = self.df.reset_index(drop=True)
        print('Done! Added composite score in column grand_index') 
                       
    def _lookup_subtest_norms(self, method):
        if method == 'census_rank_INT':
            norm_dir = self.census_INT_norm_dir
        elif method == 'rank_INT':
            norm_dir = self.INT_norm_dir
        subtests = self.df['specific_subtest_id'].unique()
        norm_col_name = f'{method}_normed_score'
        for sub in subtests:
            print(f'Looking up norms for subtest ID {sub}')
            table_fn = f'subtest_{int(sub)}.pkl'
            subtest_df = self.df.query('specific_subtest_id == @sub').copy(deep=True)          
            # Negate the scores of select subtests so that higher score = better
            neg_scores = -1 * subtest_df['raw_score']
            subtest_df.loc[:, 'negated_score'] = neg_scores
            if sub in self.subtests_to_negate:                
                col_to_norm = 'negated_score'
            else:
                col_to_norm = 'raw_score'

            try:
                yield self._lookup_normed_scores(subtest_df, col_to_norm, 
                                                 norm_col_name, table_fn, norm_dir)
            except FileNotFoundError:
                print(f'No norm table for {table_fn}!')
                continue
         
    def _lookup_battery_norms(self, method):
        if method == 'census_rank_INT':
            norm_dir = self.census_INT_norm_dir
        elif method == 'rank_INT':
            norm_dir = self.INT_norm_dir
        batteries = self.df['battery_id'].unique()
        for bat in batteries:            
            table_fn = f'battery_{int(bat)}.pkl'
            battery_df = self.df.query('battery_id == @bat')    
            battery_df = battery_df.groupby('test_run_id', as_index=False).nth(0)
            try:
                yield self._lookup_normed_scores(battery_df, 'mean_normed_score', 
                                                 'grand_index', table_fn, norm_dir)
            except FileNotFoundError:
                print(f'No norm table for {table_fn}!')
                continue                
            
    def _lookup_normed_scores(self, subdf, raw_col, normed_col, table_fn, norm_dir):
        table_path = os.path.join(norm_dir, table_fn)
        lookup_table = pickle.loads(pkgutil.get_data('lumos_ncpt_tools', table_path))
        norm_function = functools.partial(self._round_to_nearest, lookup_table=lookup_table)
        norms = subdf[raw_col].map(norm_function, na_action='ignore')
        subdf.loc[:, normed_col] = norms
        return subdf
        
    def _round_to_nearest(self, raw, lookup_table=None):
        keys, vals = np.array(list(lookup_table.keys())), list(lookup_table.values())
        return vals[np.argmin(np.abs(keys - raw))]