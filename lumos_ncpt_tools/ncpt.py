import pandas as pd
import numpy as np
import seaborn as sns
import yaml
import pkgutil
import matplotlib.pyplot as plt
from .mixins import MapConfigInfoMixin, OutliersMixin, ScoreLookupMixin


class NCPT(MapConfigInfoMixin,
           OutliersMixin,
           ScoreLookupMixin):
    """
    Methods for filtering and analyzing a NCPT dataset.
    """

    config_path = '/config/ncpt_config.yaml'
    
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.config = yaml.safe_load(pkgutil.get_data('lumostools', self.config_path))
               
    def add_subtest_names(self):
        """Add columns with descriptive names for the the general and specific subtest IDs."""
        self.add_column_from_config_map(
                'alternate_ids', 'general_subtest_name', 'specific_subtest_id', 0)
        self.add_column_from_config_map(
                'subtests', 'specific_subtest_name', 'specific_subtest_id', 0)
             
    def get_battery_info(self):
        """Determine all batteries and their subtests in the dataset,
        and the prevalence of each battery."""
        
        all_batteries = self.df['battery_id'].unique()
        total_n = len(self.df['test_run_id'].unique())
        battery_prevalence = np.array([])
        battery_n = np.array([])
        battery_subtests = np.array([])
        for bid in all_batteries:
            battery_df = self.df.query('battery_id == @bid')
            this_n = len(battery_df['test_run_id'].unique())
            battery_n = np.append(battery_n, this_n)
            battery_prevalence = np.append(battery_prevalence, np.round(100 * (this_n / total_n), 3))
            battery_subtests = np.append(battery_subtests, battery_df['specific_subtest_id'].unique())
            
        # Sort by prevalence
        sort_inds = np.flip(np.argsort(battery_prevalence))
        print('Total number of assessments: {0}'.format(total_n))
        print('Battery ID: N, % of total')
        for si in sort_inds:
            print(f'{all_batteries[si]}: {battery_n[si]}, {battery_prevalence[si]}%')
                        
    def filter_by_col(self, filt_col, filt_vals, inplace=False, df=None):
        # Filter by battery, subtest ID, nth_take, etc.
        df2filt = self.df if df is None else df            
        if inplace:
            df2filt.query(f'{filt_col} in @filt_vals', inplace=True)
        else:
            return df2filt.query(f'{filt_col} in @filt_vals', inplace=False)  
        
    def filter_by_baseline_gameplays(self, n_gameplays, inplace=False, df=None):
        # TODO: remove? 
        # Filters for NCPT takes in which the number of gameplays before the test was <= n_gameplays
        df2filt = self.df if df is None else df           
        if inplace:
            df2filt.query('num_plays_before <= @n_gameplays', inplace=True)
        else:
            return df2filt.query('num_plays_before <= @n_gameplays', inplace=False) 

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
        
        
    def drop_nan_rows(self, cols, inplace=False, df=None):
        df2filt = self.df if df is None else df
        if inplace:
            df2filt.dropna(subset=cols, inplace=True)
        else:
            return df2filt.dropna(subset=cols, inplace=False)
        
        
    def report_stats(self):
        """Print simple summary statistics for the dataset."""
        n_users = len(self.df['user_id'].unique())
        n_assessments = len(self.df['test_run_id'].unique())
        print('Summary Stats')
        print(f'N users: {n_users}')
        print(f'N assessments: {n_assessments}') 
        print(f'N rows: {len(self.df)}')
        print('')
        
        
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