import pkgutil
from collections import OrderedDict
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table1():
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    age_map = {'Ages 18-39': [18, 39],
               'Ages 40-59': [40, 59],
               'Ages 60-99': [60, 99]}
    age_bins = [[18, 29], [30, 39], [40, 49], [50, 59], [60, 69], [70, 99]]
    edu_map = {'Some high school': [1], 'High school': [2], 'Some college': [3],
               "Associate's degree": [8], 'College degree': [4], 
               'Professional degree': [5], "Master's degree": [6],
               'Ph.D.': [7], 'Other': [99]}
    
    def __init__(self, data_dir, save_dir, figsize):
        self.data_dir = data_dir
        self.png_path = os.path.join(save_dir, 'table1.png')
        self.svg_path = os.path.join(save_dir, 'table1.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.figsize = figsize
        
    def make_table(self):
        data = []
        for bat_id in self.batteries:
            data.append(self._get_battery_data(bat_id))            
        table = self._format_table(data)
        table.savefig(self.png_path, bbox_inches='tight')
        table.savefig(self.svg_path, transparent=True, bbox_inches='tight')                
    
    def _get_battery_data(self, bat_id):
        bat_data = {}
        data_fn = f'battery{bat_id}_df.csv'
        bat_df = load_data(self.data_dir, data_fn) 
        bat_df.drop_duplicates(subset=['test_run_id'], inplace=True)
        return self._get_demog_stats(bat_df, bat_id)
        
    def _get_demog_stats(self, df, bat_id):
        name = f'Battery {bat_id}'
        n_users = len(df)

        # Gender
        n_male = len(df.loc[df['gender'] == 'm'])
        n_female = len(df.loc[df['gender'] == 'f'])
        n_nan_gender = len(df.loc[df['gender'].isnull()])
        perc_male = round(100 * n_male / n_users, 1)
        perc_female = round(100 * n_female / n_users, 1)
        perc_nan_gender = round(100 * n_nan_gender / n_users, 1)

        # Age
        mean_age = round(df['age'].mean(), 1)
        std_age = round(df['age'].std(), 1)
        age_stats = OrderedDict()
        for age_name, ab in self.age_map.items():
            age_n = len(df.query('@ab[0] <= age <= @ab[1]'))
            age_perc = round(100 * age_n / n_users, 1)
            age_stats[age_name] = f'{age_n} ({age_perc}%)'

        # Education
        edu_stats = OrderedDict()
        for ed_name, ed_levels in self.edu_map.items():
            ed_n = len(df.query('education_level in @ed_levels'))
            ed_perc = round(100 * ed_n / n_users, 1)
            edu_stats[ed_name] = f'{ed_n} ({ed_perc}%)'

        stats = {'n_users': n_users, 'n_male': n_male, 'n_female': n_female, 
                 'perc_male': perc_male, 'perc_female': perc_female,
                 'mean_age': mean_age, 'std_age': std_age, 'edu_stats': edu_stats, 
                 'age_stats': age_stats, 'name': name}

        return stats  
    
    def _format_table(self, data):
        cols = [f'{d["name"]} \n N = {d["n_users"]}' for d in data]
        gen_rows = ['N, Male', 'N, Female'] 
        age_rows = list(data[0]['age_stats'].keys())
        mean_age = 'Age (mean {0} s.d. years)'.format(u"\u00B1")
        edu_rows = list(data[0]['edu_stats'].keys())
        rows = gen_rows + [''] + age_rows + [mean_age] + [''] + edu_rows
    
        table_data = []
        for d in data:
            this_data = [f'{d["n_male"]} ({d["perc_male"]}%)',
                         f'{d["n_female"]} ({d["perc_female"]}%)']
            this_data.append('')
            this_data.extend(list(d['age_stats'].values()))
            this_data.append('{0} {1} {2}'.format(d["mean_age"], u"\u00B1", d["std_age"]))
            this_data.append('')
            this_data.extend(list(d['edu_stats'].values()))
            table_data.append(this_data)
        
        data_array = np.array(table_data).T
        table = Table(self.figsize, scale=(1, 1.25))
        table_fig = table.plot_table(cols, rows, data_array)
        return table_fig
