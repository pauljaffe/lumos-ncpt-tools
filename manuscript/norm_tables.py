import pkgutil
from collections import OrderedDict
import os
from itertools import product
import copy

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data

class NormTables():
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    age_bins = [[18, 29], [30, 39], [40, 49], [50, 59], [60, 69], [70, 99]]
    edu_bins = [[1, 2], [3, 4, 8], [5, 6, 7]] 
    genders = ['m', 'f']
    pctiles = [10, 25, 50, 75, 90]
    cols = ['Subtest', 'Subtest ID', 'Age (years)', 'Education', 'Gender', 'N', 
            'Mean', 'SD', '10th perc.', '25th perc.', '50th perc.', '75th perc.', '90th perc.']
    batteries = [17, 32, 39, 50, 60]
    
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = os.path.join(save_dir, 'norm_tables')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.figsize = figsize
        
    def make_tables(self):
        for bat_id in self.batteries:
            print(f'Battery {bat_id}')
            bat_data = self._get_battery_data(bat_id)
            bat_df = pd.DataFrame(data=bat_data, columns=self.cols)
            save_path = os.path.join(self.save_dir, f'battery{bat_id}_norms.csv')
            bat_df.to_csv(save_path, sep=',', index=False)
    
    def _get_battery_data(self, bat_id):
        bat_data = []
        data_fn = f'battery{bat_id}_df.csv'
        bat_df = load_data(self.data_dir, data_fn) 
        subtests = self.config['batteries'][bat_id][1]
        for sub in subtests:
            sub_df = bat_df.query('specific_subtest_id == @sub')
            subtest_data = self._get_subtest_data(sub_df, sub)
            bat_data.extend(subtest_data)
        return bat_data

    def _get_subtest_data(self, df, sub_id):
        subtest_data = []
        n_pre = len(df)
        df_filt = df.dropna(subset=['raw_score', 'gender', 'education_level', 'age'])
        sub_name = self.config['subtests'][sub_id][0]
        bin_vec = [sub_name, sub_id]
        for dbin in product(self.age_bins, self.edu_bins, self.genders):
            bin_data = copy.copy(bin_vec)
            bin_data.extend(self._get_bin_data(df_filt, dbin))
            subtest_data.append(bin_data)
        return subtest_data

    def _get_bin_data(self, df, dbin):
        bin_data = []
        age, edu, gen = dbin[0], dbin[1], dbin[2]
        age_lo, age_hi = age[0], age[1]
        bin_df = df.query('@age_lo <= age <= @age_hi and education_level in @edu and gender == @gen')
        bin_data.append(f'{age_lo}-{age_hi}')
        if edu == [1, 2]:
            bin_data.append('HS/Some HS')
        elif edu == [3, 4, 8]:
            bin_data.append("College/Some college/Associate's")
        else:
            bin_data.append("Professional deg./Ph.D./Master's")
        if gen == 'm':
            bin_data.append('Male')
        else:
            bin_data.append('Female')
        bin_data.append(len(bin_df))
        bin_data.extend(self._get_bin_stats(bin_df))
        return bin_data

    def _get_bin_stats(self, df):
        stats = []
        scores = df['raw_score'].values
        stats.append(np.round(np.mean(scores), 2))
        stats.append(np.round(np.std(scores), 2))
        for p in self.pctiles:
            stats.append(int(np.percentile(scores, p)))
        return stats
