import pkgutil
from collections import OrderedDict
import os

import yaml
import numpy as np
import pandas as pd

from lumos_ncpt_tools.utils import load_data
from .manuscript_utils import Table

class Table2():
    """Make a table with the subtest IDs for each battery."""
    config_path = '../lumos_ncpt_tools/config/ncpt_config.yaml'
    age_map = {'Ages 18-39': [18, 39],
               'Ages 40-59': [40, 59],
               'Ages 60-99': [60, 99]}
    age_bins = [[18, 29], [30, 39], [40, 49], [50, 59], [60, 69], [70, 99]]
    edu_map = {'Some high school': [1], 'High school': [2], 'Some college': [3],
               "Associate's degree": [8], 'College degree': [4], 
               'Professional degree': [5, 6, 7], 'Other': [99]}
    
    def __init__(self, data_dir, save_dir, figsize):
        self.data_dir = data_dir
        self.png_path = os.path.join(save_dir, 'table1.png')
        self.svg_path = os.path.join(save_dir, 'table1.svg')        
        self.config = yaml.safe_load(pkgutil.get_data('lumos_ncpt_tools', self.config_path))
        self.batteries = self.config['batteries'].keys()
        self.figsize = figsize
        
    def make_table(self):
        cols = [f'Battery {bat_id}' for bat_id in self.batteries]
        rows = ['Subtest IDs']
    
        table_data = []
        for bat_id in self.batteries:
            bat_subtests = self.config['batteries'][bat_id][1]
            if len(bat_subtests) > 5:
                s1 = ', '.join(str(s) for s in bat_subtests[:5])
                s2 = ', '.join(str(s) for s in bat_subtests[5:])
                subtest_str = s1 + ',\n' + s2
            else:
                subtest_str = ', '.join(str(s) for s in bat_subtests)
            table_data.append([subtest_str])
        
        data_array = np.array(table_data).T
        table = Table(self.figsize)
        table_fig = table.plot_table(cols, rows, data_array)
        table_fig.savefig(self.png_path, bbox_inches='tight')
        table_fig.savefig(self.svg_path, transparent=True, bbox_inches='tight')    