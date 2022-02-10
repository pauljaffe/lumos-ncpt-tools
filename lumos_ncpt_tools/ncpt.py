import pkgutil
import pdb

import pandas as pd
import numpy as np
import seaborn as sns
import yaml
import matplotlib.pyplot as plt

from .mixins import OutliersMixin, ScoreLookupMixin


class NCPT(OutliersMixin,
           ScoreLookupMixin):
    """Methods for filtering and analyzing a NCPT dataset.
    
    Args
    ----
    df (DataFrame): DataFrame containing NCPT data. 
    """
    
    config_path = '/config/ncpt_config.yaml'
    
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        
    def report_stats(self):
        """Display simple summary statistics for the dataset."""
        n_users = len(self.df['user_id'].unique())
        n_assessments = len(self.df['test_run_id'].unique())
        n_subtests = len(self.df)
        print('Data summary')
        print('------------')
        print(f'N users: {n_users}')
        print(f'N tests: {n_assessments}') 
        print(f'N subtests: {n_subtests}')
        print(f'DataFrame columns: {self.df.columns.tolist()}')
        print('')
        
    def get_subtest_info(self):
        """Display some basic information on the subtests in self.df."""
        print('Subtest information')
        print('-------------------')
        subtests = np.sort(self.df['specific_subtest_id'].unique())
        for sub in subtests:
            subtest_df = self.df.query('specific_subtest_id == @sub')
            name = self.config['subtests'][sub][0]
            v = self.config['subtests'][sub][2]
            N = len(subtest_df)
            print(f'Subtest ID {sub}: {name}, {v}, N scores = {N}')
        print('')
    
    def get_education_info(self):
        """Display the meaning of the numeric education levels."""
        edu = self.config['education']
        for key, val in edu.items():
            print(f'{key}: {val}')
        print('')

    def filter_by_completeness(self, ids=None, inplace=False, df=None):
        """Retain only users that have completed all of the subtests for 
        a given battery (i.e. no other subtests and no missing subtests).
        """
        
        df2filt = self.df if df is None else df
        ids2filt = df2filt['battery_id'].unique() if ids is None else ids     
        keep_run_ids = []
        for bi in ids2filt:
            b_subtests = self.config['batteries'][bi][1]
            # Remove incorrect subtests
            battery_df = df2filt.query('battery_id == @bi and specific_subtest_id in @b_subtests')
            # Check each run ID has correct number of subtests
            subtest_counts = battery_df.groupby('test_run_id')['specific_subtest_id'].apply(len)
            correct_num = subtest_counts[subtest_counts == len(b_subtests)].index.tolist()
            # Check subtests for each test run ID are unique
            unique_subtests = battery_df.groupby('test_run_id')['specific_subtest_id'].nunique()
            correct_unique = unique_subtests[unique_subtests == len(b_subtests)].index.tolist()
            keep_run_ids.extend(list(set(correct_num).intersection(set(correct_unique))))

        if inplace:
            df2filt.query('test_run_id in @keep_run_ids', inplace=True)
            filt_df, exclude_df = None, None
        else:
            filt_df = df2filt.query('test_run_id in @keep_run_ids', inplace=False)
            exclude_df = df2filt.query('test_run_id not in @keep_run_ids', inplace=False)
            
        return filt_df, exclude_df
 
    def plot_score_dists_new_seaborn(self, subtests='all', save_dir=None):
        if subtests == 'all':
            plot_df = self.df
        else:
            plot_df = self.df.query('specific_subtest_id in @subtests')
        raw_score_fig = sns.displot(data=plot_df, x='raw_score', col='specific_subtest_id', kind="kde")
        normed_score_fig = sns.displot(data=plot_df, x='normed_score', col='specific_subtest_id', kind="kde")
        if save_dir is not None:
            raw_score_fig.savefig(save_dir + '/' + 'raw_score_dists.png',
                    bbox_inches='tight', dpi=150)
            normed_score_fig.savefig(save_dir + '/' + 'normed_score_dists.png',
                    bbox_inches='tight', dpi=150)
                        
    def plot_score_dists(self, subtests, save_dir=None, figsize=(12, 6)):
        if subtests == 'all':
            plot_tests = self.df['specific_subtest_id'].unique()
        else:
            plot_tests = subtests
        score_fig, score_ax = plt.subplots(2, len(plot_tests), figsize=figsize)
        for ind, st in enumerate(plot_tests):
            st_df = self.df.query('specific_subtest_id == @st')
            sns.distplot(st_df['raw_score'], hist=False, ax=score_ax[0, ind])
            score_ax[0, ind].set_title(f'test id {st}')
            sns.distplot(st_df['normed_score'], hist=False, ax=score_ax[1, ind])           
        if save_dir is not None:
            score_fig.savefig(save_dir + '/' + 'score_dists.png',
                    bbox_inches='tight', dpi=150)
        plt.show()
           
    def save_df(self, save_path):        
        self.df.to_csv(save_path, sep=',', index=False)