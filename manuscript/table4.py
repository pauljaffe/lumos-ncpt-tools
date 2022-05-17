import pkgutil
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table4():
    
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    
    def __init__(self, save_dir, figsize):
        self.png_path = os.path.join(save_dir, 'table4.png')
        self.svg_path = os.path.join(save_dir, 'table4.svg')   
        self.figsize = figsize
        self.data = [['user_id', 'Participant (user) identifier'],
                     ['age', 'Age of participant (years)'],
                     ['gender', 'Gender of participant'],
                     ['education_level', "Numeric identifier for participant's education"],
                     ['country', 'Home country of participant'],
                     ['test_run_id', 'Identifier for the entire test taken by the participant'],
                     ['battery_id', 'NCPT battery identifier'],
                     ['specific_subtest_id', 'NCPT subtest identifier'],
                     ['raw_score', 'Raw score for the subtest'],
                     ['rank_INT_normed_score', 'Normalized subtest score calculated using a rank-based INT'],
                     ['grand_index', 'Composite score for the entire test (Grand Index)']]
        
    def make_table(self):
        cols = ['Variable name', 'Description']        
        data_array = np.array(self.data)
        table = Table(self.figsize, scale=(1, 1))
        table_fig = table.plot_table_no_rows(cols, data_array)
        table_fig.savefig(self.png_path, bbox_inches='tight')
        table_fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')   