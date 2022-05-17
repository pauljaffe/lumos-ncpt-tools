import pkgutil
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data

class SummaryStats():
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    
    def __init__(self, data_dir):
        self.data_dir = data_dir       
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.N_users = 0
        self.N_scores = 0

    def get_summary_stats(self): 
        for bat_id in self.batteries:
            data_fn = f'battery{bat_id}_df.csv'
            bat_df = load_data(self.data_dir, data_fn) 
            bat_users = bat_df['user_id'].unique()
            self.N_users += len(bat_users)
            self.N_scores += len(bat_df) # All NAN scores have been removed
        print(f'N unique users: {self.N_users}, N total scores: {self.N_scores}')
            