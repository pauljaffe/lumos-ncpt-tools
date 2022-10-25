import pkgutil
from collections import OrderedDict
import os
import csv

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table1():
    
    config_path = './config/ncpt_config.yaml'
    age_map = {'Ages 18-39': [18, 39],
               'Ages 40-59': [40, 59],
               'Ages 60-99': [60, 99]}
    age_bins = [[18, 29], [30, 39], [40, 49], [50, 59], [60, 69], [70, 99]]
    edu_map = {'Some high school': [1], 'High school': [2], 'Some college': [3],
               "Associate's degree": [8], 'College degree': [4], 
               'Professional degree': [5], "Master's degree": [6],
               'Ph.D.': [7], 'Other': [99]}
    acs_path = '../manuscript/US_ACS_2019_data.csv'
    
    def __init__(self, data_dir, save_dir, figsize):
        self.data_dir = data_dir
        self.png_path = os.path.join(save_dir, 'table1.png')
        self.svg_path = os.path.join(save_dir, 'table1.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.figsize = figsize
        
    def make_table(self):
        data = []
        acs_data = self._get_acs_data()
        for bat_id in self.batteries:
            data.append(self._get_battery_data(bat_id))            
        table = self._format_table(data, acs_data)
        table.savefig(self.png_path, bbox_inches='tight')
        table.savefig(self.svg_path, transparent=True, bbox_inches='tight')                

    def _get_acs_data(self):
        # Add data from the 2019 United States Census Bureau’s 
        # American Community Survey (ACS) 1-year Public Use Microdata Sample
        # rather klugish...
        csvdata = pkgutil.get_data('lumos_ncpt_tools', self.acs_path)
        reader = csv.DictReader(csvdata.decode('utf-8').splitlines(), delimiter=',')
        acs_list = [row['reformatted'] for row in reader]
        return acs_list
    
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
        perc_male = round(100 * n_male / n_users, 2)
        perc_female = round(100 * n_female / n_users, 2)
        perc_nan_gender = round(100 * n_nan_gender / n_users, 2)
        gender_sum = n_male + n_female + n_nan_gender
        assert gender_sum == n_users, "Gender sum doesn't sum to n_users!"

        # Age
        mean_age = round(df['age'].mean(), 2)
        std_age = round(df['age'].std(), 2)
        age_stats = OrderedDict()
        age_sum = 0
        for age_name, ab in self.age_map.items():
            age_n = len(df.query('@ab[0] <= age <= @ab[1]'))
            age_perc = round(100 * age_n / n_users, 2)
            age_sum += age_n
            age_stats[age_name] = f'{age_n} ({age_perc}%)'
        assert age_sum == n_users, "Age sum doesn't sum to n_users!"

        # Education
        edu_stats = OrderedDict()
        edu_sum = 0
        for ed_name, ed_levels in self.edu_map.items():
            ed_n = len(df.query('education_level in @ed_levels'))
            ed_perc = round(100 * ed_n / n_users, 2)
            edu_sum += ed_n
            edu_stats[ed_name] = f'{ed_n} ({ed_perc}%)'
        n_nan_edu = len(df.loc[df['education_level'].isnull()])
        nan_edu_perc = round(100 * n_nan_edu / n_users, 2)
        edu_sum += n_nan_edu
        edu_stats['Not reported'] = f'{n_nan_edu} ({nan_edu_perc}%)'
        assert edu_sum == n_users, "Edu sum doesn't sum to n_users!"

        stats = {'n_users': n_users, 'n_male': n_male, 'n_female': n_female, 
                 'n_nan_gender': n_nan_gender, 'perc_male': perc_male, 'perc_female': perc_female,
                 'perc_nan_gender': perc_nan_gender, 'mean_age': mean_age, 'std_age': std_age, 
                 'edu_stats': edu_stats, 'age_stats': age_stats, 'name': name}

        return stats  
    
    def _format_table(self, data, acs_data):
        cols = [f'{d["name"]} \n N = {d["n_users"]}' for d in data]
        cols.append(f'2019 ACS \n N = {acs_data[0]}')
        gen_rows = ['N, Male', 'N, Female', 'N, Not reported'] 
        age_rows = list(data[0]['age_stats'].keys())
        mean_age = 'Age (mean {0} s.d. years)'.format(u"\u00B1")
        edu_rows = list(data[0]['edu_stats'].keys())
        rows = gen_rows + [''] + age_rows + [mean_age] + [''] + edu_rows
    
        table_data = []
        for d in data:
            this_data = [f'{d["n_male"]} ({d["perc_male"]}%)',
                         f'{d["n_female"]} ({d["perc_female"]}%)',
                         f'{d["n_nan_gender"]} ({d["perc_nan_gender"]}%)']
            this_data.append('')
            this_data.extend(list(d['age_stats'].values()))
            this_data.append('{0} {1} {2}'.format(d["mean_age"], u"\u00B1", d["std_age"]))
            this_data.append('')
            this_data.extend(list(d['edu_stats'].values()))
            table_data.append(this_data)
        table_data.append(acs_data[1:])
        
        data_array = np.array(table_data).T
        table = Table(self.figsize, scale=(1, 1.25))
        table_fig = table.plot_table(cols, rows, data_array)
        return table_fig
